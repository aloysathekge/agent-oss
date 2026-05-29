# Quarq Agent v0.4.0

import os
import json
import asyncio
import re
import numpy as np
import faiss
from datetime import datetime
from typing import TypedDict, Sequence
from dotenv import load_dotenv
import shutil
import time

from pydantic import SecretStr
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
import tools.tool_manager as tool_manager
import uuid

from functools import wraps

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
load_dotenv()

raw_api_key = os.getenv("OPENAI_API_KEY")



AGENT_ID = os.getenv("AGENT_ID") or "local_agent"
USER_ID = os.getenv("USER_ID")

if not raw_api_key:
    raise ValueError("Missing critical environment variable: OPENAI_API_KEY.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_AGENT_ID = re.sub(r"[^a-zA-Z0-9_.-]", "_", AGENT_ID)
LOCAL_MEMORY_ROOT = os.getenv("LOCAL_MEMORY_ROOT", os.path.join(BASE_DIR, "local_memory"))
LOCAL_AGENT_MEMORY_DIR = os.path.join(LOCAL_MEMORY_ROOT, LOCAL_AGENT_ID)

if not all([raw_api_key, AGENT_ID]):
    raise ValueError(
        "Missing critical environment variables (OPENAI_API_KEY, AGENT_ID)."
    )



# LLM for Text Generation
retrieval_llm = ChatOpenAI(
   api_key=SecretStr(raw_api_key),
    temperature=0,
    model="gpt-4o-mini",
    timeout=30,      
    max_retries=3    

)

gen_llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=SecretStr(raw_api_key),
    temperature=0,
    max_retries=3   
)


learn_llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=SecretStr(raw_api_key),
    temperature=0,
    timeout=60,      
    max_retries=3    
)

EMBED_MODEL = "text-embedding-3-large"

# OpenAI Client for Vector Embeddings
embed_client = OpenAIEmbeddings(
    model=EMBED_MODEL,
    api_key=SecretStr(raw_api_key),
    dimensions=1536,
    timeout=20,     
    max_retries=3    
)


# ==========================================
# GLOBAL CACHE & CONCURRENCY LIMITERS
# ==========================================
AGENT_CONFIG_CACHE = None
LEARNING_SEMAPHORE = asyncio.Semaphore(4)  # 🛠️ NEW: Max 4 concurrent DB saves
PENDING_LEARNING_TASKS = set()  # 🛠️ NEW: Tracks active background tasks



# ==========================================
# ROBUST NETWORK RETRY DECORATOR
# ==========================================
def network_retry(max_retries=4, initial_delay=2.0, timeout=15.0): # 🛠️ ADDED TIMEOUT
    """Automatically retries an async function and prevents infinite thread hanging."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    # 🛠️ FORCE TIMEOUT ON THE FUNCTION CALL
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                except asyncio.TimeoutError:
                    print(f"⚠️ [Network Timeout] {func.__name__} hung for {timeout}s (Attempt {attempt+1}/{max_retries}).")
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"❌ [Fatal Error] {func.__name__} failed after {max_retries} attempts: {e}")
                        raise e 
                    print(f"⚠️ [Network Retry] {func.__name__} failed ({e}). Retrying in {delay}s...")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2 
            raise TimeoutError(f"{func.__name__} completely failed after {max_retries} attempts.")
        return wrapper
    return decorator


def persistent_network_retry(initial_delay=2.0, max_delay=60.0, timeout=None): # 🛠️ Changed default to None
    """
    An infinite retry loop for background tasks. 
    It will NEVER drop the task. It keeps trying forever until it succeeds.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            attempt = 1
            while True:  # 🛠️ INFINITE LOOP
                try:
                    if timeout:
                        return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                    else:
                        return await func(*args, **kwargs) # 🛠️ Waits forever without killing the nested DB loop
                except asyncio.TimeoutError:
                    print(f"⚠️ [Persistent Queue] {func.__name__} hung for {timeout}s (Attempt {attempt}). Retrying...")
                except Exception as e:
                    print(f"⚠️ [Persistent Queue] {func.__name__} failed ({e}) (Attempt {attempt}). Retrying in {delay}s...")
                
                # Exponential backoff, capped at `max_delay`
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                attempt += 1
        return wrapper
    return decorator


# ==========================================
# HELPER: ROBUST CONTENT EXTRACTION (Gemini Fix)
# ==========================================
def extract_pure_text(response) -> str:
    """Safely extracts raw text from Gemini's complex list/dict response structures."""
    content = response.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = ""
        for part in content:
            if isinstance(part, dict):
                text += part.get("text", "")
            else:
                text += str(part)
    else:
        text = str(content)
        
    # Strip <thinking> tags if the model returned them
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()

def extract_json_block(text: str, is_array: bool = False) -> str:
    """Extracts the last outermost valid JSON object/array from model text."""
    if not text:
        return ""

    text = str(text)

    text = re.sub(
        r"<thinking>.*?</thinking>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    decoder = json.JSONDecoder()
    expected_start = "[" if is_array else "{"
    expected_type = list if is_array else dict

    candidates = []

    for idx, char in enumerate(text):
        if char != expected_start:
            continue

        try:
            parsed, end_offset = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, expected_type):
            candidates.append((idx, idx + end_offset, text[idx : idx + end_offset]))

    if not candidates:
        return ""

    outermost = []
    for start, end, block in candidates:
        is_nested = any(
            other_start < start and end <= other_end
            for other_start, other_end, _ in candidates
        )
        if not is_nested:
            outermost.append((start, end, block))

    return outermost[-1][2] if outermost else candidates[0][2]

def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        clean = re.sub(r"\s+", " ", item).strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


MAX_STRUCTURED_ARTIFACT_MEMORIES = 60
MAX_ARTIFACT_LIST_ITEMS_PER_BLOCK = 10


def _table_cells(line: str) -> list[str]:
    line = line.strip().strip("|")
    return [c.strip() for c in line.split("|")]


def _is_table_separator(cells: list[str]) -> bool:
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells
    )


def extract_table_rows(text: str, current_time: str, artifact_context: str) -> list[str]:
    lines = text.splitlines()
    memories = []
    i = 0

    while i < len(lines) - 1:
        if "|" not in lines[i] or "|" not in lines[i + 1]:
            i += 1
            continue

        headers = _table_cells(lines[i])
        sep = _table_cells(lines[i + 1])

        if len(headers) < 2 or not _is_table_separator(sep):
            i += 1
            continue

        i += 2
        while i < len(lines) and "|" in lines[i]:
            row = _table_cells(lines[i])
            if len(row) >= 2 and not _is_table_separator(row):
                row_label = row[0] or "row"
                pairs = []
                for h, v in zip(headers[1:], row[1:]):
                    if h and v:
                        pairs.append(f"{h} = {v}")

                if pairs:
                    memories.append(
                        f"On {current_time}, in {artifact_context}, table row {row_label}: "
                        + "; ".join(pairs)
                        + "."
                    )
            i += 1

    return memories


def extract_heading_sections(text: str, current_time: str, artifact_context: str) -> list[str]:
    lines = text.splitlines()
    memories = []

    heading_re = re.compile(
        r"^\s*(#{1,6}\s+.+|chapter\s+\d+[:.\-].+|\d+[.)]\s+.+|[A-Z][^.!?]{3,80}:)\s*$",
        re.IGNORECASE,
    )

    current_heading = None
    buffer = []

    def flush():
        if current_heading and buffer:
            content = " ".join(x.strip() for x in buffer if x.strip())
            content = re.sub(r"\s+", " ", content).strip()
            if content:
                memories.append(
                    f"On {current_time}, in {artifact_context}, section {current_heading}: {content}"
                )

    for line in lines:
        clean = line.strip()
        if heading_re.match(clean):
            flush()
            current_heading = clean.lstrip("#").strip().rstrip(":")
            buffer = []
        elif current_heading:
            buffer.append(clean)

    flush()
    return memories


def extract_list_items(text: str, current_time: str, artifact_context: str) -> list[str]:
    memories = []
    current_heading = "the artifact"

    heading_re = re.compile(r"^\s*(#{1,6}\s+.+|[A-Z][^.!?]{3,80}:)\s*$")
    item_re = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$")

    for line in text.splitlines():
        clean = line.strip()

        if heading_re.match(clean):
            current_heading = clean.lstrip("#").strip().rstrip(":")
            continue

        match = item_re.match(clean)
        if match:
            memories.append(
                f"On {current_time}, in {artifact_context}, under {current_heading}, list item: {match.group(1).strip()}"
            )

    return memories


def extract_labeled_lines(text: str, current_time: str, artifact_context: str) -> list[str]:
    memories = []

    patterns = [
        re.compile(r"^::\s*(?P<label>[^:\n]+?)\s*::\s*==\s*(?P<value>.+)$"),
        re.compile(r"^(?P<label>[A-Za-z][^:\n]{2,80})\s*[:=]\s*(?P<value>.+)$"),
    ]

    for line in text.splitlines():
        clean = line.strip()
        for pattern in patterns:
            match = pattern.match(clean)
            if match:
                label = match.group("label").strip()
                value = match.group("value").strip()
                if label.lower() in {"user", "assistant", "ai", "human", "system"}:
                    continue
                memories.append(
                    f"On {current_time}, in {artifact_context}, labeled item {label}: {value}"
                )
                break

    return memories


def extract_explicit_artifact_block_memories(
    text: str,
    current_time: str,
    artifact_context: str,
) -> list[str]:
    """Extract only explicit artifact blocks like `::title:: == description`."""
    memories = []
    marker_re = re.compile(r"::\s*(?P<label>[^:\n]{1,120}?)\s*::\s*==")

    for line in text.splitlines():
        matches = list(marker_re.finditer(line))
        if not matches:
            continue

        for idx, match in enumerate(matches):
            label = re.sub(r"\s+", " ", match.group("label")).strip()
            value_start = match.end()
            value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
            value = re.sub(r"\s+", " ", line[value_start:value_end]).strip()

            if not label or not value:
                continue

            memories.append(
                f"On {current_time}, in {artifact_context}, explicit artifact {label}: {value}"
            )

    return _dedupe_keep_order(memories)


def _strip_inline_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    return text.strip()


def _split_named_list_item(body: str) -> tuple[str, str] | None:
    body = _strip_inline_markdown(re.sub(r"\s+", " ", body).strip())
    if not body:
        return None

    match = re.match(
        r"(?P<title>.{2,120}?)(?:\s+[-–—]\s+|:\s+)(?P<desc>.+)$",
        body,
    )
    if not match:
        return None

    title = _strip_inline_markdown(match.group("title")).strip(" .:-")
    desc = _strip_inline_markdown(match.group("desc")).strip()

    if not title or len(desc) < 8:
        return None

    return title, desc


def _has_named_list_anchor(title: str) -> bool:
    """Return true for likely named entities, not generic step labels."""
    clean = re.sub(r"[^A-Za-z0-9&'+ ]", " ", title)
    words = [w for w in clean.split() if w]
    if not words:
        return False

    generic_starters = {
        "add", "assemble", "bake", "blend", "bring", "brush", "check",
        "choose", "consider", "create", "cut", "do", "don't", "dont",
        "enjoy", "explore", "factor", "have", "knead", "make", "map",
        "order", "pace", "pick", "plan", "preheat", "roll", "share",
        "stay", "try", "use", "visit", "watch", "wear", "whisk",
    }

    capitalized = [
        w for w in words
        if re.match(r"^[A-Z][A-Za-z0-9&'+-]*$", w) and w.lower() not in {"a", "an", "the"}
    ]
    acronyms_or_numbers = [
        w for w in words
        if re.search(r"\d", w) or (len(w) > 1 and w.isupper())
    ]

    if len(capitalized) >= 2 or acronyms_or_numbers:
        return True

    first = words[0].lower()
    if len(capitalized) == 1 and first not in generic_starters:
        return True

    return False


def extract_artifact_list_item_memories(
    text: str,
    current_time: str,
    artifact_context: str,
) -> list[str]:
    """
    Extract high-signal bullet/numbered list items without turning every list
    into memory. This is meant for recommendation/artifact lists like
    `1. The Sugar Factory - ...`, not generic step-by-step instructions.
    """
    memories = []
    item_re = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$")
    context_signal_re = re.compile(
        r"\b(recommend|suggest|option|idea|place|spot|restaurant|shop|activity|"
        r"things to do|where to|itinerary|list|example|dessert|dining|"
        r"family-friendly|museum|exhibition|attraction|venue|operator|trail|"
        r"recipe)\b",
        re.IGNORECASE,
    )

    lines = text.splitlines()
    current_context = artifact_context
    block_count = 0

    for line in lines:
        clean = line.strip()
        match = item_re.match(clean)

        if not match:
            if clean:
                current_context = _clean_table_context(clean)[:180] or artifact_context
            block_count = 0
            continue

        context_blob = f"{artifact_context} {current_context} {clean}"
        if not context_signal_re.search(context_blob):
            continue

        if block_count >= MAX_ARTIFACT_LIST_ITEMS_PER_BLOCK:
            continue

        split = _split_named_list_item(match.group(1))
        if not split:
            continue

        title, desc = split
        if not _has_named_list_anchor(title):
            continue

        memories.append(
            f"On {current_time}, in {artifact_context}, under {current_context}, "
            f"list item {title}: {desc}"
        )
        block_count += 1

    return _dedupe_keep_order(memories)


def extract_structured_artifact_memories(
    text: str,
    current_time: str,
    artifact_context: str,
) -> list[str]:
    memories = []
    memories.extend(extract_markdown_table_row_memories(text, current_time))
    memories.extend(
        extract_explicit_artifact_block_memories(
            text,
            current_time,
            artifact_context,
        )
    )
    memories.extend(extract_artifact_list_item_memories(text, current_time, artifact_context))

    # Keep deterministic extraction high-signal only. Generic headings, numbered
    # sections, and `Label: value` lines are intentionally left to the learning
    # model because extracting all of them floods Episodic memory. List extraction
    # is restricted to named recommendation/artifact items and capped per block.
    return _dedupe_keep_order(memories)[:MAX_STRUCTURED_ARTIFACT_MEMORIES]


def _split_markdown_table_row(line: str) -> list[str]:
    """Split a simple markdown table row into cells."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _is_markdown_table_line(line: str) -> bool:
    return "|" in line and len(_split_markdown_table_row(line)) >= 2


def _is_markdown_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False

    for cell in cells:
        compact = cell.replace(" ", "")
        if not re.fullmatch(r":?-{3,}:?", compact):
            return False
    return True


def _clean_table_context(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(User|AI|Assistant):\s*", "", text, flags=re.IGNORECASE)
    return text.strip(" :-")


def extract_markdown_table_row_memories(text: str, current_time: str = "") -> list[str]:
    """
    Convert markdown tables into row-level memory strings.

    The learning LLM may summarize a table and drop row/cell mappings. These
    deterministic row memories preserve the artifact shape before compression.
    """
    lines = text.splitlines()
    row_memories: list[str] = []
    i = 0

    while i < len(lines):
        if not _is_markdown_table_line(lines[i]):
            i += 1
            continue

        if i + 1 >= len(lines):
            break

        headers = _split_markdown_table_row(lines[i])
        separator = _split_markdown_table_row(lines[i + 1])

        if not _is_markdown_separator_row(separator):
            i += 1
            continue

        table_start = i
        data_rows: list[list[str]] = []
        i += 2

        while i < len(lines) and _is_markdown_table_line(lines[i]):
            row = _split_markdown_table_row(lines[i])
            if not _is_markdown_separator_row(row):
                data_rows.append(row)
            i += 1

        if not data_rows:
            continue

        context_lines: list[str] = []
        context_idx = table_start - 1
        while context_idx >= 0 and len(context_lines) < 3:
            candidate = lines[context_idx].strip()
            if candidate and not _is_markdown_table_line(candidate):
                context_lines.append(candidate)
            context_idx -= 1

        context = _clean_table_context(" ".join(reversed(context_lines)))
        if not context:
            context = "the table"

        for row_num, row in enumerate(data_rows, start=1):
            max_len = max(len(headers), len(row))
            padded_headers = headers + [""] * (max_len - len(headers))
            padded_row = row + [""] * (max_len - len(row))

            first_header = padded_headers[0].strip() or "row"
            first_value = padded_row[0].strip()

            cell_pairs: list[str] = []
            for idx in range(1, max_len):
                header = padded_headers[idx].strip() or f"Column {idx + 1}"
                value = padded_row[idx].strip()
                if not value:
                    continue
                cell_pairs.append(f"{header} = {value}")

            if not cell_pairs:
                for idx in range(max_len):
                    header = padded_headers[idx].strip() or f"Column {idx + 1}"
                    value = padded_row[idx].strip()
                    if value:
                        cell_pairs.append(f"{header} = {value}")

            if not cell_pairs:
                continue

            row_label = f"{first_header} {first_value}".strip()
            if row_label == "row":
                row_label = f"row {row_num}"

            prefix = f"On {current_time}, " if current_time else ""
            row_memories.append(
                f"{prefix}in {context}, table row {row_label}: "
                + "; ".join(cell_pairs)
                + "."
            )

    return row_memories
# ==========================================
# HELPER: TOKEN EXTRACTION (Breakdown)
# ==========================================
def get_token_metrics(response) -> dict:
    """Extracts input, output, and total token usage from LangChain AIMessage."""
    metrics = {"input": 0, "output": 0, "total": 0}

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        metrics["input"] = response.usage_metadata.get("input_tokens", 0)
        metrics["output"] = response.usage_metadata.get("output_tokens", 0)
        metrics["total"] = response.usage_metadata.get("total_tokens", 0)
    elif (
        hasattr(response, "response_metadata")
        and "token_usage" in response.response_metadata
    ):
        usage = response.response_metadata["token_usage"]
        # Safe check: ensures 'usage' is a dict before calling .get()
        if isinstance(usage, dict):
            metrics["input"] = usage.get("prompt_tokens", 0)
            metrics["output"] = usage.get("completion_tokens", 0)
            metrics["total"] = usage.get("total_tokens", 0)

    return metrics




class VectorMemoryManager:
    """Manages Semantic and Episodic memories locally with FAISS + JSON."""

    def __init__(self, memory_type: str):
        self.memory_type = memory_type
        self.dim = 1536
        self.folder = os.path.join(
            LOCAL_AGENT_MEMORY_DIR, f"{memory_type.lower()}_memory"
        )
        self.index_file = os.path.join(self.folder, "index.faiss")
        self.store_file = os.path.join(self.folder, "memories.json")
        self.lock = asyncio.Lock()

        os.makedirs(self.folder, exist_ok=True)
        self.memories = self._load_memories()
        self.index = self._load_or_rebuild_index()

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _load_memories(self) -> list[dict]:
        if not os.path.exists(self.store_file):
            return []
        try:
            with open(self.store_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:
            print(f"⚠️ [Memory] Failed to load {self.memory_type} store: {e}")
            return []

    def _save_memories(self):
        os.makedirs(self.folder, exist_ok=True)
        with open(self.store_file, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

        faiss.write_index(self.index, self.index_file)

    def _vector_from_embedding(self, embedding: list[float]) -> np.ndarray:
        vector = np.array([embedding], dtype="float32")
        faiss.normalize_L2(vector)
        return vector

    def _new_index(self):
        return faiss.IndexFlatIP(self.dim)

    def _rebuild_index(self):
        self.index = self._new_index()
        vectors = []

        for memory in self.memories:
            embedding = memory.get("embedding")
            if embedding:
                vectors.append(np.array(embedding, dtype="float32"))

        if vectors:
            matrix = np.array(vectors, dtype="float32")
            faiss.normalize_L2(matrix)
            self.index.add(matrix)

    def _load_or_rebuild_index(self):
        if os.path.exists(self.index_file):
            try:
                index = faiss.read_index(self.index_file)
                if index.ntotal == len(self.memories):
                    return index
            except Exception as e:
                print(f"⚠️ [Memory] Failed to load FAISS index: {e}")

        self.index = self._new_index()
        self._rebuild_index()
        self._save_memories()
        return self.index

    def _format_memory(self, memory: dict) -> str:
        return (
            f"[STORED_AT: {memory.get('created_at', self._now())}] "
            f"[ID: {memory['id']}] {memory['content']}"
        )

    @persistent_network_retry(initial_delay=2.0, max_delay=30.0, timeout=30.0)
    async def execute_action(self, action: dict):
        act_type = action.get("action", "").upper()
        content = action.get("content", "")
        raw_id = action.get("id")

        record_id = None
        if raw_id:
            match = re.search(r"([0-9a-fA-F\-]{36})", raw_id)
            if match:
                record_id = match.group(1)

        async with self.lock:
            if act_type == "DELETE" and record_id:
                before = len(self.memories)
                self.memories = [m for m in self.memories if m.get("id") != record_id]
                if len(self.memories) != before:
                    self._rebuild_index()
                    self._save_memories()
                    print(f"🗑️ [Memory] DELETED {self.memory_type} memory: {record_id}")
                return

            if not content.strip():
                return

            embedding = await embed_client.aembed_query(text=content)
            if not embedding:
                return

            vector = self._vector_from_embedding(embedding)

            if act_type == "ADD":
                if self.index.ntotal > 0:
                    scores, _ = self.index.search(vector, 1)
                    if scores[0][0] >= 0.95:
                        print(f"🔄 [Memory] Skipped duplicate {self.memory_type} ADD.")
                        return

                self.memories.append(
                    {
                        "id": str(uuid.uuid4()),
                        "agent_id": AGENT_ID,
                        "memory_type": self.memory_type,
                        "content": content,
                        "embedding": embedding,
                        "created_at": self._now(),
                        "updated_at": self._now(),
                    }
                )
                self.index.add(vector)
                self._save_memories()
                print(f"✅ [Memory] ADDED {self.memory_type}: {content[:30]}...")

            elif act_type == "UPDATE" and record_id:
                for memory in self.memories:
                    if memory.get("id") == record_id:
                        memory["content"] = content
                        memory["embedding"] = embedding
                        memory["updated_at"] = self._now()
                        self._rebuild_index()
                        self._save_memories()
                        print(f"✏️ [Memory] UPDATED {self.memory_type} memory: {record_id}")
                        return

    @network_retry(max_retries=4, initial_delay=4.0)
    async def search(self, query: str, top_k: int = 10, threshold: float = 0.38) -> str:
        if not query.strip() or self.index.ntotal == 0:
            return ""

        embedding = await embed_client.aembed_query(text=query)
        if not embedding:
            return ""

        vector = self._vector_from_embedding(embedding)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < threshold:
                continue
            if idx < len(self.memories):
                results.append(self._format_memory(self.memories[idx]))

        return "\n".join(results)

    @network_retry(max_retries=3, initial_delay=1.0)
    async def keyword_search(self, keywords: list[str], top_k: int = 10) -> str:
        if not keywords:
            return ""

        results = {}
        for kw in keywords:
            clean_kw = kw.strip().lower()
            if len(clean_kw) < 3:
                continue

            for memory in self.memories:
                if clean_kw in memory.get("content", "").lower():
                    results[memory["id"]] = self._format_memory(memory)
                    if len(results) >= top_k:
                        break

        return "\n".join(results.values())

    @network_retry(max_retries=3, initial_delay=2.0)
    async def clear(self):
        async with self.lock:
            self.memories = []
            self.index = self._new_index()
            if os.path.exists(self.folder):
                shutil.rmtree(self.folder)
            os.makedirs(self.folder, exist_ok=True)
            self._save_memories()
            print(f"🧹 [Memory] Wiped all {self.memory_type} memories for agent {AGENT_ID}.")


semantic_db = VectorMemoryManager("Semantic")
episodic_db = VectorMemoryManager("Episodic")


def get_formatted_rules_with_ids(rules: list, limit: int = 15) -> str:
    """Formats a specific list of rule objects with their IDs and DB Timestamps for LLM context."""
    if not rules:
        return ""
    limited_rules = rules[:limit]

    formatted = []
    for r in limited_rules:
        raw_time = r.get("created_at", datetime.now().isoformat())
        dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        formatted.append(f"[STORED_AT: {dt}] [ID: {r['id']}] {r['rule']}")

    return "\n".join(formatted)


PROCEDURAL_DIR = os.path.join(LOCAL_AGENT_MEMORY_DIR, "procedural_memory")
PROCEDURAL_FILE = os.path.join(PROCEDURAL_DIR, "rules.json")


def _load_rules_file() -> list:
    if not os.path.exists(PROCEDURAL_FILE):
        return []
    try:
        with open(PROCEDURAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"⚠️ [Rules] Failed to load rules: {e}")
        return []


def _save_rules_file(rules: list):
    os.makedirs(PROCEDURAL_DIR, exist_ok=True)
    with open(PROCEDURAL_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)


@network_retry(max_retries=4, initial_delay=2.0, timeout=15.0)
async def load_procedural_rules() -> list:
    rules = await asyncio.to_thread(_load_rules_file)
    return sorted(
        rules,
        key=lambda r: r.get("created_at", ""),
        reverse=True,
    )


def save_procedural_rules(valid_rules: list):
    rules = _load_rules_file()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for rule_obj in valid_rules:
        rules.append(
            {
                "id": str(uuid.uuid4()),
                "agent_id": AGENT_ID,
                "rule": rule_obj.get("rule"),
                "reasoning": rule_obj.get("reasoning", ""),
                "target_entity": rule_obj.get("target_entity", "global"),
                "tags": rule_obj.get("tags", []),
                "created_at": now,
                "updated_at": now,
            }
        )

    _save_rules_file(rules)


@network_retry(max_retries=4, initial_delay=2.0, timeout=15.0)
async def clear_procedural_rules():
    await asyncio.to_thread(_save_rules_file, [])

async def wipe_all_memories():
    """Wipes all vectors and rules for this agent from the DB."""
    try:
        await semantic_db.clear()
        await episodic_db.clear()
        
        # 🛠️ USE THE NEW PROTECTED FUNCTION
        await clear_procedural_rules()
        
        print(f"🧹 [Rules] Wiped all procedural rules for agent {AGENT_ID}.")
        print(f"✅ Agent {AGENT_ID} is now completely blank and ready for the next test.")
    except Exception as e:
        print(f"❌ [Error] Failed to wipe all memories: {e}")

def sort_memories_by_recency(memory_block: str,max_lines: int = 15) -> str:
    """Parses timestamps in a block of text and sorts lines newest-to-oldest."""
    if not memory_block or memory_block == "None":
        return "None"

    lines = memory_block.strip().split("\n")

    def extract_timestamp(line):
        match = re.search(
        r"\[(?:STORED_AT:\s*)?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]",
        line,
    )
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            except Exception:
                return datetime.min
        return datetime.min

    # Sort lines by timestamp descending
    sorted_lines = sorted(lines, key=extract_timestamp, reverse=True)
    # 🛠️ TOKEN PROTECTION: Keep only the most recent N lines
    capped_lines = sorted_lines[:max_lines]
    
    return "\n".join(capped_lines)


# ==========================================
# 4. GRAPH STATE
# ==========================================
class AgentState(TypedDict):
    user_prompt: str
    chat_history: Sequence[BaseMessage]
    semantic_context: str
    episodic_context: str
    procedural_context: str
    hyde_queries: list[str]         # 🛠️ NEW: Pass queries to the next node
    selected_skills: list[str]  # UPDATE: Now a list of strings
    skill_markdown: str  # NEW: Documentation for active tools
    final_response: str
    skip_learning: bool
    user_id: str  # NEW: Unique identifier for the user
    channel_type: str  # NEW: e.g., 'telegram', 'whatsapp', 'terminal'
    metrics: dict
    current_date: str  # 🛠️ ADDED: To explicitly pass the benchmark date


# ==========================================
# 5. RETRIEVAL NODE (Robust Tagging)
# ==========================================
async def retrieve_memories_node(state: AgentState):
    start_time = time.time()  # START TIMER

    # 🛠️ BENCHMARK SYNC: If this is a final benchmark question, wait for all background memories to save FIRST!
    global PENDING_LEARNING_TASKS
    if state.get("skip_learning", False) and PENDING_LEARNING_TASKS:
        print(
            f"⏳ [Benchmark Sync] Waiting for {len(PENDING_LEARNING_TASKS)} background chunks to finish saving to DB..."
        )
        await asyncio.gather(*PENDING_LEARNING_TASKS, return_exceptions=True)
        print("✅ [Benchmark Sync] All memories saved. Proceeding with retrieval.")

    in_tokens = 0
    out_tokens = 0

    user_prompt = state["user_prompt"]
    history_text = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state["chat_history"][-5:]]
    )

    # 🛠️ NEW: Use the provided state date, fallback to system clock if none provided
    if state.get("current_date"):
        current_time_str = state["current_date"]
    else:
        current_time_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    hyde_prompt = f"""
    You are an AI Search Query Optimizer. 
    Analyze the recent chat history and the user's latest prompt.
    You must generate distinct search queries to maximize the chances of finding the right memory in a hybrid database (Vector + Keyword).



    Query Definitions:
    - Query 1 (Comprehensive Baseline): A concise, 3rd-person factual statement capturing the core subject of the user's intent. (This is the primary direct search).
    - Query 2 (Entity Focus / Relational Anchor):  A keyword list of specific objects, tools, brands, or places. If the prompt is relational (e.g., 'X before Y'), focus this query strictly on the anchor (Y).
    - Query 3 (Action Focus / Relational Target):  A keyword list of verbs, milestones, or thematic concepts. If the prompt is relational (e.g., 'X before Y'), focus this query strictly on the target detail (X). 
    - Query 4 (Literal Unit & Noun Net): A raw, comma-separated list of ONLY the exact nouns, numbers, and quantitative units (e.g., hours, dollars, three) from the prompt. DO NOT add the word "User". Keep it strictly to the user's literal vocabulary. DO NOT include verbs. DO NOT use synonyms. Keep it under 6 words.
    - Element 5 (Exact Keywords): A raw, comma-separated string of 2 to 4 highly specific proper nouns, names, or rare keywords from the prompt for direct text matching.
    

    DO NOT answer the user's question. Just state the context for a database search.
    
    CRITICAL RULES:
    1. PERSPECTIVE: Convert 1st-person ("I", "my") into 3rd-person ("User", "User's") for Queries 1, 2, and 3. Do NOT add "User" to Query 4.
       Queries 1, 2 MUST literally start with either:
        - "User"
        - "User's"
        This is mandatory.
        Query 3 FORMAT RULES:
        - MUST start with "User's "
        - MUST contain EXACTLY 1-2 comma-separated concepts after "User's"
        - Each concept MUST be 1-3 words maximum
        - NO additional commas
        - NO explanations
        - NO expansions
        - Example valid outputs:
        - "User's leadership, management"
        - "User's purchases, spending"
    2. AGGREGATION & MONEY (CRITICAL): If the user asks for a total sum, count, or cost (e.g., "how much money", "total hours", "how many"):
       - For money/expenses: Query 4 MUST include literal financial symbols and terms: "$, cost, paid, price, bought, purchased, spent".
       - For time/counts: Query 4 MUST include unit words: "hours, times, total, count, days".
       Prefer verb forms like "attended", "visited" over abstract nouns like "attendance".
    3. RELATIONAL DECONSTRUCTION & NEGATIVE CONSTRAINTS (CRITICAL): 
       - If the prompt relates two different things (e.g., 'What did I do before X?', 'Who was at Y with me?', 'X during Y'), Queries 2 and 3 MUST search for those two components INDEPENDENTLY. 
         * Query 2: Search for the context/anchor event (The meeting, the doctor, the concert).
         * Query 3: Search for the specific detail or action (The food, the bedtime, the companion).
       - If the user asks for advice, recommendations, suggestions, or ideas (e.g., "Can you suggest...", "Any advice?", "Can you recommend..."), Query 3 MUST ALSO explicitly search for the user's past struggles, dislikes, negative constraints, or explicit non-interests related to the topic to ensure the agent knows what to avoid (e.g., "User's dislikes, struggles, avoids, not interested in, strictly prefers").
    4. CONDITIONAL TIME RESOLUTION (CRITICAL): The current system date is {current_time_str}. 
       - ONLY append dates if the user's intent is temporally bound (e.g., specific past events, action times, time comparisons, durations).
       - If the user's prompt contains relative time (e.g., "last weekend", "yesterday", "recently"), you MUST explicitly calculate the exact target date or year (Month and Day/Year) and put THAT DATE in Query 1 and Query 2. 
       - NEVER use the actual words "last weekend" or "yesterday" in your generated queries. Replace them completely with the calculated absolute date (e.g., "May 9").
       - HOW-LONG-AGO QUERY GUARD: If the user asks "how long ago", "how many weeks ago", "how many days ago", or similar, this rule overrides the relative-date calculation rule above. DO NOT invent or calculate the target event date inside the search query. Search for the named event/entity itself. The current date is only the calculation anchor after retrieval, not a guessed event date.
       - DO NOT append dates or timestamps if the user is asking about permanent facts, timeless attributes, demographics, or general preferences (e.g., "What is my ethnicity?", "Do I like dogs?").
    5. CATEGORICAL , GEOGRAPHICAL & HYPONYMS EXPANSION (CRITICAL): If the user asks about a broad category, an action, or a geographical region, your queries MUST include the category/region name PLUS at least 4 to 5 specific physical examples or sub-locations to catch exact matches in the database. 
       - Noun Example: "electronics" -> "electronics, phone, laptop, TV, headphones, monitor"
       - Verb Example: "read" -> "read, book, novel, article, magazine, textbook"
       - Geography Example: "Europe" -> "Europe, Paris, Rome, London, Germany, Spain"
       - Geography Example: "Hawaii" -> "Hawaii, Maui, Oahu, Kauai, Honolulu"
       Do not just use abstract synonyms; you MUST list specific physical items or sub-locations.
    6. NO ANSWERS: Do not try to answer the question. Only provide the search context.
    7. DYNAMIC SEARCH MODE & JSON FORMAT ONLY (CRITICAL): You must classify the required search depth based on the fundamental nature of the user's question. This determines how wide the database search net will be cast.
       
       Use "deep" (Wide Net / High Recall) IF the prompt involves ANY of the following:
       - Aggregations & Lists: Questions asking for totals, counts, calculations, or exhaustive lists (e.g., "how many", "total cost", "list all").
       - Temporal Spans & Histories: Questions requiring timelines, durations, or finding the origin/transition point of a current state (e.g., "how long have I", "when did I start", "history of").
       - Broad Categories: Questions asking about a general class of items where the database might hold highly specific variants (e.g., asking about "vehicles" when the database might hold "sedan" or "bicycle").
       - Exploratory & Advisory: Open-ended requests for recommendations, advice, or suggestions, which require pulling historical dislikes, past constraints, and broad preferences.
       
       Use "standard" (Strict Net / High Precision) ONLY IF the prompt is:
       - Point-Fact Retrieval: Direct, highly specific questions looking for a single established fact, name, or location (e.g., "What is the name of my pet?", "Where did I leave my keys?", "Who is my manager?").
       
       You MUST return a valid JSON object matching this exact schema. Do not include markdown blocks, just the raw JSON:
       {{
           "vector_queries": ["Query 1", "Query 2", "Query 3", "Query 4"],
           "keywords": "exact, proper, nouns",
           "search_mode": "standard" // or "deep"
       }}

    Recent Chat History:
    {history_text if history_text else "None"}
    
    User's Latest Prompt: "{user_prompt}"
    
    ---
    EXAMPLES OF OPTIMIZED QUERIES:

    Input: "What did I eat before the meeting?" (Relational / Point-Fact)
    Output: {{
        "vector_queries": [
            "User's food consumption prior to the meeting",
            "User's meeting history and schedule",
            "User's meals, diet, and what they ate",
            "eat, meeting"
        ],
        "keywords": "eating, ate, meeting",
        "search_mode": "standard"
    }}

    Input: "How many hours in total did I spend driving?" (Aggregation)
    Output: {{
        "vector_queries": [
            "User total driving duration and travel history",
            "User vehicle, road trip, transit records",
            "User travel milestones, driving time calculation",
            "hours, road trip, destinations"
        ],
        "keywords": "driving, hours, total",
        "search_mode": "deep"
    }}

    Input: "Can you recommend a new hobby for me?" (Exploratory & Advisory)
    Output: {{
        "vector_queries": [
            "User's current hobbies, interests, and recreational activities",
            "User's pastimes, skills, free time",
            "User's dislikes, struggles, quit doing, avoids, negative constraints",
            "hobby, interests, activities, free time"
        ],
        "keywords": "hobby, interests, activities",
        "search_mode": "deep"
    }}
    
    """


    hyde_response = await retrieval_llm.ainvoke([HumanMessage(content=hyde_prompt)])
    # 🛠️ FIXED: Use robust extractor
    content = extract_pure_text(hyde_response)
    
    

    # Track HyDE tokens
    m_hyde = get_token_metrics(hyde_response)
    in_tokens += m_hyde["input"]
    out_tokens += m_hyde["output"]

    
    # Clean JSON
    if content.startswith("```json"): content = content[7:]
    if content.startswith("```"): content = content[3:]
    if content.endswith("```"): content = content[:-3]
    content = content.strip()

    # ---------------------------------------------------------
    # 🛠️ BULLETPROOF JSON PARSING & DYNAMIC THRESHOLD LOGIC
    # ---------------------------------------------------------
    search_queries = []
    keywords = []
    search_mode = "standard"

    try:
        # Isolate just the JSON object explicitly
        json_str = extract_json_block(content, is_array=False)
        parsed_data = json.loads(json_str)
        
        if isinstance(parsed_data, dict):
            search_queries = parsed_data.get("vector_queries", [])
            
            # Handle keywords (safely parsing string or list)
            keyword_data = parsed_data.get("keywords", "")
            if isinstance(keyword_data, list):
                keyword_data = ",".join(keyword_data)
            keywords = [k.strip() for k in keyword_data.split(",") if len(k.strip()) >= 3]
            
            # Extract Mode
            search_mode = str(parsed_data.get("search_mode", "standard")).lower()
            
        elif isinstance(parsed_data, list):
            # Safe Fallback in case LLM generates the old array format
            if len(parsed_data) == 5:
                keyword_str = str(parsed_data.pop())
                keywords = [k.strip() for k in keyword_str.split(",") if len(k.strip()) >= 3]
            search_queries = parsed_data[:4]
            
    except Exception as e:
        print(f"⚠️ [HyDE] JSON parse failed ({e}). Attempting text fallback.")
        
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.lower().startswith("here are") and not line.lower().startswith("output:"):
                clean_line = re.sub(r"^[\d\.\-\*\s]+", "", line).strip().strip('"').strip("'")
                if clean_line:
                    cleaned_lines.append(clean_line)
        
        if cleaned_lines:
            search_queries = cleaned_lines[:4]
            if len(cleaned_lines) > 4:
                keywords = [k.strip() for k in cleaned_lines[4].split(",") if len(k.strip()) >= 3]
        else:
            print("⚠️ [HyDE] Text fallback failed. Reverting to raw user prompt.")
            search_queries = [user_prompt]

    # Ensure we don't have empty queries
    search_queries = [sq for sq in search_queries if sq.strip()]
    if not search_queries:
        search_queries = [user_prompt]

    # 🛠️ DYNAMIC THRESHOLD APPLICATION
    current_threshold = 0.28 if search_mode == "deep" else 0.38

    print(f"HYDE Mode: [{search_mode.upper()}] (Threshold: {current_threshold})")
    print("HYDE Vector Queries:", search_queries)
    if keywords:
        print("HYDE Direct Keywords:", keywords)


    # 🛠️ CHANGED: CONCURRENT SEARCH FOR ALL QUERIES USING DYNAMIC THRESHOLD
    semantic_tasks = [semantic_db.search(sq, top_k=20, threshold=current_threshold) for sq in search_queries]
    episodic_tasks = [episodic_db.search(sq, top_k=20, threshold=current_threshold) for sq in search_queries]

    if keywords:
        # Add the Keyword Search tasks to the concurrent pool
        semantic_tasks.append(semantic_db.keyword_search(keywords, top_k=10))
        episodic_tasks.append(episodic_db.keyword_search(keywords, top_k=10))

    all_semantic_results = await asyncio.gather(*semantic_tasks)
    all_episodic_results = await asyncio.gather(*episodic_tasks)

    # 🛠️ NEW: DEDUPLICATE RESULTS BASED ON DATABASE ID
    def deduplicate_memories(results_list):
        unique_memories = {}
        for result_block in results_list:
            if not result_block: continue
            lines = result_block.split('\n')
            for line in lines:
                # Extract ID to use as a unique key
                match = re.search(r"\[ID: ([0-9a-fA-F\-]{36})\]", line)
                if match:
                    unique_memories[match.group(1)] = line
        return "\n".join(unique_memories.values())

    combined_semantic = deduplicate_memories(all_semantic_results)
    combined_episodic = deduplicate_memories(all_episodic_results)

    # Apply Temporal Sorting
    semantic_result = sort_memories_by_recency(combined_semantic,max_lines=70)
    episodic_result = sort_memories_by_recency(combined_episodic,max_lines=70)

    all_rules=[]
      # 🛠️ UPDATED: Procedural Tag Routing with CoT and Fallback
    try:
        all_rules = await load_procedural_rules()
    except Exception:
        print("⚠️ [Warning] Procedural rules completely failed to load after 4 retries. Moving on without them.")
        all_rules = []


    procedural_result = ""

    if all_rules:

        known_tags = list(
            set(
                tag
                for rule in all_rules
                if isinstance(rule, dict)
                for tag in rule.get("tags", [])
            )
        )
        tag_prompt = f"""
        You are an intelligent Routing AI. Your task is to determine which behavioral rules the agent needs to answer the user's prompt correctly.
        
        Current User Prompt: "{user_prompt}"
        Recent Semantic Context: {semantic_result}
        Recent Episodic Context: {episodic_result}
        
        Available Rule Tags: {known_tags}
        
        CHAIN OF THOUGHT REQUIREMENTS:
        1. Analyze the user's intent. Is it a greeting? A coding request? A technical architecture question?
        2. Select tags that apply to this intent.
        3. ALWAYS include the "global" tag, as it contains universal personality traits.
        
        You MUST respond EXACTLY with a valid JSON object matching this schema:
        {{
            "reasoning": "Briefly explain what the user wants and why you selected these tags.",
            "tags": ["global", "tag1", "tag2"]
        }}
        """

        response = await retrieval_llm.ainvoke([HumanMessage(content=tag_prompt)])
        # TRACK BREAKDOWN
        m = get_token_metrics(response)
        in_tokens += m["input"]
        out_tokens += m["output"]

        # 🛠️ FIXED: Use robust extractor
        content = extract_pure_text(response)
        

        try:
            # Isolate just the JSON object
            json_str = extract_json_block(content, is_array=False)
            parsed_data = json.loads(json_str)

            # CRITICAL FIX: Verify parsed_data is a dictionary
            if isinstance(parsed_data, dict):
                requested_tags = [
                    str(tag).lower() for tag in parsed_data.get("tags", [])
                ]
            else:
                # If LLM returned a list or string, fallback to global
                requested_tags = ["global"]

            matched_rules = []
            for rule in all_rules:
                # ARMOR: Skip if the JSON file got corrupted with raw strings
                if not isinstance(rule, dict):
                    continue

                rule_tags = [t.lower() for t in rule.get("tags", [])]
                if (
                    any(tag in rule_tags for tag in requested_tags)
                    or "global" in rule_tags
                ):
                    matched_rules.append(rule)

            # 🛠️ CHANGED: Use the formatter to attach IDs and apply the Hard Limit
            procedural_result = get_formatted_rules_with_ids(matched_rules, limit=8)

        except Exception as e:
            print(f"[Warning] Failed to parse procedural tags: {e}")
            # Fallback: Just show the 10 most recent rules if routing fails
            procedural_result = get_formatted_rules_with_ids(all_rules, limit=8)

    procedural_result = sort_memories_by_recency(procedural_result)

    end_time = time.time()  # END TIMER

    print("\n--- Memory Retrieval Complete ---")
    if semantic_result:
        print(f"Semantic Found:\n{semantic_result}")
    if episodic_result:
        print(f"Episodic Found:\n{episodic_result}")
    if procedural_result:
        print(f"Procedural Found:\n{procedural_result}")
    else:
        print("Procedural Found: None (No relevant tags found)")

    print(
        f"⏱️ [Metrics] Time: {end_time - start_time:.2f}s | Tokens: In({in_tokens}) Out({out_tokens})"
    )
    print("---------------------------------\n")

    return {
        "semantic_context": semantic_result,
        "episodic_context": episodic_result,
        "procedural_context": procedural_result,
        "hyde_queries": search_queries, # 🛠️ NEW
        "metrics": {"retrieval_in": in_tokens, "retrieval_out": out_tokens},
    }


# ==========================================
# 5b. TOOL ROUTING NODE
# ==========================================
async def route_tools_node(state: AgentState):
    """Pick skills for this turn using the tool_manager."""
    start_time = time.time()

    # NEW: Disable tool calling entirely during benchmarks
    if state.get("channel_type") == "benchmark":
        print("--- Tool Routing: Skipped (Benchmark Mode) ---")
        return {"selected_skills": [], "skill_markdown": ""}

    # We pass the history to the tool router for better intent detection
    history_text = "\n".join(
        [f"{m.type}: {m.content}" for m in state["chat_history"][-4:]]
    )

    # --- NEW: Compile memory context for the router ---
    memory_context = f"""
    [Semantic]: {state.get('semantic_context', 'None')}
    [Episodic]: {state.get('episodic_context', 'None')}
    [Procedural]: {state.get('procedural_context', 'None')}
        """.strip()

    chosen_skills = await tool_manager.select_skills(
        user_prompt=state["user_prompt"],
        recent_history=history_text,
        memory_context=memory_context,
    )

    if not chosen_skills:
        print("--- Tool Routing: No skill selected ---")
        return {"selected_skills": [], "skill_markdown": ""}

    print(f"--- Tool Routing: Skills Selected -> '{chosen_skills}' ---")

    combined_markdown = ""
    for skill in chosen_skills:
        loaded = tool_manager.load_skill(skill)
        combined_markdown += f"\n### {skill.upper()} SKILL\n{loaded['markdown']}\n"

    end_time = time.time()

    print(f"⏱️ [Metrics] Time: {end_time - start_time:.2f}s")
    print("---------------------------------\n")
    return {"selected_skills": chosen_skills, "skill_markdown": combined_markdown}


# ==========================================
# 6. GENERATION NODE (Robust CoT Generation & ReAct Loop)
# ==========================================
async def generate_response_node(state: AgentState):
    start_time = time.time()

    # 1. INITIALIZE variables at the top of the function scope
    in_tokens = 0
    out_tokens = 0

    # 🛠️ TRACK CONTEXT FOR STATE UPDATES & LEARNING
    sem_ctx_to_save = state.get("semantic_context", "")
    epi_ctx_to_save = state.get("episodic_context", "")

    global AGENT_CONFIG_CACHE  # 🛠️ Pull in the global cache

    if AGENT_CONFIG_CACHE is None:
        print("📁 [Local Config] Loading Agent Config from environment/defaults...")
        AGENT_CONFIG_CACHE = {
            "agent_name": os.getenv("AGENT_NAME") or "Quarq Agent",
            "agent_personality": os.getenv("AGENT_PERSONALITY") or "professional and helpful",
            "agent_use_cases": [
                item.strip()
                for item in os.getenv("AGENT_USE_CASES", "general assistance").split(",")
                if item.strip()
            ],
            "agent_custom_prompt": os.getenv("AGENT_CUSTOM_PROMPT") or "",
        }

    cfg = AGENT_CONFIG_CACHE
    name = cfg.get("agent_name") or "Quarq Agent"
    personality = cfg.get("agent_personality") or "professional and helpful"
    use_cases = ", ".join(cfg.get("agent_use_cases") or ["general assistance"])
    custom_prompt = cfg.get("agent_custom_prompt") or ""



    # 🛠️ NEW: Pass current date down to the LLM
    if state.get("current_date"):
        current_time_str = state["current_date"]
    else:
        current_time_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    temporal_hack = ""
    if state.get("channel_type") == "benchmark":
        temporal_hack = """
        BENCHMARK TEMPORAL GROUNDING:
        - The CURRENT SYSTEM DATE/TIME may come from the benchmark scenario, not the real wall clock. Use it only to resolve relative date phrases in the user's prompt or conversation.
        - Bracketed timestamps at the start of memory lines are database storage timestamps, not event dates.
        - Never use bracketed storage timestamps as event dates, sequence anchors, or date-gap values.
        - Never infer an event date from neighboring memories unless the neighboring memory explicitly names the same event/entity or an exact alias.
        - If a named event lacks an explicit narrative date in the memory content, trigger REQUIRED_DATA with targeted queries for that event and its date.
        """


    identity_instruction = f"""
    [IDENTITY & PERSONA]
    Your Name: {name}
    Personality/Tone: {personality}
    Core Objectives: You are specifically optimized for: {use_cases}.
    {f'Custom User Directives: {custom_prompt}' if custom_prompt else ''}
    CURRENT SYSTEM DATE/TIME: {current_time_str} 
    

    Your responses must strictly align with this identity and tone.
                """.strip()

    system_instruction = f"""You are a highly advanced, disciplined AI assistant created by QuarqLabs Team.

    {identity_instruction}


    TEMPORAL TRUTH PROTOCOL (CRITICAL):
    1. The memories above are provided in RECENCY ORDER (Newest information at the top). The STORED_AT field at the start of each memory line is database storage metadata. It is used only for recency and contradiction resolution, not as the event date . If the text of the memory explicitly states a different date (e.g., "User bought a car in April 2023"), you MUST use the date in the text as the true historical date of the event, NOT the bracketed database log.
    2. EXACT EVENT DATE ATTRIBUTION:
        If the user asks for a date, duration, gap, order, or number of days between named events/entities, each date must be explicitly attached to the exact named event/entity in retrieved memory content.
        A date attached only to a broader category, generic description, nearby memory, related topic, or semantically similar event is not valid evidence for the named event.
        You may merge a date from another retrieved memory only if that memory explicitly names the same event/entity or an exact alias.
        If a required named event lacks an explicit narrative date after exact-alias merging, trigger REQUIRED_DATA instead of guessing.
    3. RELATIVE ORDERING SUFFICIENCY:
        For "which came first/earlier/later" questions, an exact absolute date is not always required.

        If both candidates have enough temporal evidence to determine ordering using explicit relative phrases, durations, session dates, or anchored mention dates, you MUST answer the ordering instead of triggering uncertainty.

        Accept evidence such as:
        - "about a month ago"
        - "recently"
        - "started X days/weeks ago"
        - "finished in X days"
        - "started on [weekday]"
        - "as of [session/date], had already started/finished"

        Use exact dates when available, but for ordering questions, approximate temporal ranges are valid if they do not overlap in a way that changes the ordering.

        Only say the comparison is impossible if the relative ranges overlap or one candidate has no temporal anchor at all.

    4. MENTION-DATE VS EVENT-DATE DISAMBIGUATION:
        If the user asks what they mentioned, discussed, said, recalled, shared, talked about, participated in, or referred to "a week ago" / "last week" / on a relative conversation date, the resolved date may refer to the conversation/session date, not necessarily the historical date of the underlying event.

        For such questions, prioritize memories whose narrative line is anchored to the resolved conversation date and contains the requested event/entity, even if the underlying event happened earlier or is described as recent/before that date.

        Do not reject a correct answer merely because the underlying event has no exact historical date, if the memory explicitly says that on the resolved conversation date the user discussed, recalled, shared, or drew from that event.

        Only require the underlying event date when the user explicitly asks when the event happened, how long since the event happened, or for a date gap between actual events.
    5. CONTRADICTIONS: If two memories explicitly contradict each other (e.g., "User's favorite color is blue" vs "User's favorite color is red"), the NEWER memory (higher timestamp) is the ABSOLUTE TRUTH. Ignore the older memory.
    6. DIRECT CURRENT-STATE PRECEDENCE (OVERRIDES DERIVED MATH):
        If retrieved context contains a direct explicit answer to a current/latest/as-of state question, use that value.
        This includes durations, counts, frequencies, locations, jobs, subscriptions, routines, goals, preferences, possessions, schedules, and personal records.
        Do not recompute from older start dates, event dates, or background facts when a newer direct current/as-of value exists.
        If direct current/as-of statements conflict, the newest direct statement wins unless the user asks for history, chronology, or the original start date.
    7. COMPLEMENTARY FACTS (MERGE RULE): If multiple memories describe the SAME past event, role, entity, or item without directly contradicting (e.g., "Previous job was marketing specialist" and "Previous job involved managing interns"), you MUST synthesize and combine all of them to provide a complete, highly detailed picture. Do not discard details just because they are slightly older, unless they are explicitly corrected.
    {temporal_hack}

    CONFIDENCE & SYNTHESIS PROTOCOL:
    1. ZERO WORLD KNOWLEDGE (STRICT GROUNDING): You are strictly forbidden from using your internal pre-trained knowledge to fill in missing prices, dates, names, or facts. If the user asks for a price, cost, calculation, or comparison, and the exact numbers are NOT explicitly written in the [RETRIEVED CONTEXT], you MUST NOT guess or estimate based on real-world averages. You must trigger the REQUIRED_DATA flag.
    2. DIRECT FACTUAL ANSWER FORMAT:
        For factual recall questions, answer only the requested fact. Do not add offers, advice, praise, or follow-up suggestions.
        If the requested answer is a scalar, return the normalized scalar plus unit/category when useful.
        Keep approximation or uncertainty only when the user asks for exact precision, or when the question requires arithmetic, date gaps, prices, payments, or other exact calculations.

    3. STRICT MATH RULE & QUANTITATIVE EXTRACTION (CRITICAL): 
       - If calculating totals from multiple events, you MUST write out the step-by-step arithmetic inside your <thinking> block. 
       - When scanning memories for numbers, you MUST treat hyphenated or descriptive numerical adjectives (e.g., "a 10-pound weight", "a 3-mile run") as exact mathematical quantities (10 pounds, 3 miles). Do not claim the exact duration, cost, or amount is unspecified if a numerical adjective is present in the text.
       - EXACT NUMBER ENFORCEMENT & RANGE AVOIDANCE: If the user asks a mathematical calculation question (e.g., "how much", "total", "difference") and does not explicitly use the words "range", "minimum", or "maximum", you are STRICTLY FORBIDDEN from outputting a numerical range. 
    - You are STRICTLY FORBIDDEN from splitting a range (e.g., taking the lower or upper bound of "5-15") to invent a definitive number. 
    - If the retrieved context contains both a numerical range and a distinct singular exact number for a requested variable, you MUST unconditionally use the singular exact number for your final calculation, even if that singular number is described as an assumption, a special condition, or a quote. Never calculate using a range if an exact singular integer is present in the text.

    NUMERIC CALCULATION RULE:
        For any total, count, duration, price, cost, quantity, or money question, evaluate candidate numbers before using them.
        For each candidate number, determine:
        actor/entity, measured action/property, event/item, exactness.
        Actor/entity must come from the grammatical subject of the number-bearing clause. Example: in "User helped organize EVENT, which raised X", X belongs to EVENT, not User. Participation in or association with an entity does not transfer that entity's numbers to User.
        Include a number only when its actor/entity and measured action/property match the user's requested target.
        For calculations, totals, differences, exact costs, and date gaps, use only exact unqualified values unless the user asks for an estimate, minimum, maximum, or range.
        For direct scalar recall questions about a current/latest state, prefer the latest same-target scalar evidence over older exact values, while preserving any approximation or uncertainty present in that evidence.
        Do not canonicalize approximate numbers when the user asks "exactly", asks for arithmetic, or asks for a payment/cost/date-gap calculation.

    EVENT STATE DISAMBIGUATION (CRITICAL):
        When the user asks what they did, used, took, visited, ate, wore, received, attended, bought, or experienced, prefer memories where the action actually happened or was experienced by the user.
        Do NOT answer with memories whose state is only planned, booked, scheduled, considered, compared, researched, intended, potential, upcoming, or hypothetical unless the user specifically asks what they planned, booked, scheduled, considered, compared, or researched.
        A record date for planning, booking, discussing, or researching is not automatically the event date for the underlying action.
        If the same date contains both:
        - an actual or experienced event,
        - and a planned, booked, scheduled, considered, compared, researched, intended, potential, upcoming, or hypothetical event,
        choose the actual or experienced event for questions asking what the user did.

    ACQUISITION VERB DISAMBIGUATION:
        For factual recall questions asking what item the user got/acquired/bought/ordered/purchased/received/obtained on a resolved date, retrieval may contain a different acquisition verb.
        If the question only asks for the item identity, and the retrieved memory has exactly one date-matched acquisition event for a category-compatible item, you may answer with that item while preserving the memory's actual verb.
        For date, duration, or ordering questions, a dated acquisition/transfer/gift event can be used as the event anchor when the question is not specifically asking about payment, seller, cost, or purchase method.
        Do NOT rewrite the event as a purchase unless the memory says bought, purchased, paid, ordered, or gives a price/seller transaction.
        If the user asks specifically about payment, cost, seller, purchase method, or whether it was bought versus gifted/received/found, strict verb/source matching is required.
        If the user's verb conflicts with the memory's verb, answer transparently:
        "The memory says you [actual verb] [item] on [date]. It does not say you bought it."

    4. TARGET BINDING & ANSWER GRANULARITY:
        Before answering, bind every target field present in the user question: entity/name, role/title/status, organization, relationship label, artifact/event label, relation, time/scope, and answer type.
        Treat each bound field as required evidence, not as a soft hint.
        Candidate evidence is usable only when every bound field is supported by the same evidence chain or explicitly equated in retrieved context.
        Do not normalize, rename, or substitute identity-bearing fields. Job titles, roles, employers, programs, subscriptions, relationship labels, event names, and artifact names are identity-bearing fields.
        If retrieved context supports the requested relation but under a different identity-bearing field, reject that candidate and answer that the information is not enough, naming the supported field when useful.
        Trigger REQUIRED_DATA only when the needed target field is missing and not contradicted by retrieved context.
        Preserve the requested answer type and granularity. Do not require a narrower subtype than the question asks for, and do not answer with a broader type when a same-target narrower answer is available.
        For travel where/stay/planning/visiting questions, treat "stay" by itself as a destination/location request, not a lodging request. A city, island, region, neighborhood, venue, or named place is sufficient unless the question explicitly asks for lodging-level details such as hotel, resort, accommodation, room, booking, address, or property name. If both broad and narrower same-trip locations are present, answer with the narrower same-trip location. If multiple same-broad-destination candidates exist, prefer the latest/direct destination statement tied to the requested trip purpose and time/scope.
        If multiple candidates still match after target binding, choose the one with the closest relation and time/scope. If candidates remain unresolved, trigger REQUIRED_DATA.

    5. ENUMERATION, CATEGORY BOUNDARY & CARDINALITY:
        If the user asks for a count, ordered list, or "what are all", first identify the requested category noun and answer type.
        The requested category noun is binding. Include only candidates explicitly matching that category, explicitly described as belonging to that category, or clearly a subtype of that same category.
        Do not include sibling or nearby entity types merely because they are topically similar, appear in the same domain, or occurred in the same storyline.
        If the user asks for exactly N items of a category, filter candidates by category before counting, ordering, or challenging the count.
        If more than N retrieved candidates exist, exclude category-mismatched candidates first.
        Only say the user actually has more than N items if more than N candidates remain after category filtering.
        For ordered-list questions, order only the accepted category-matching candidates by their event dates.
        Deduplicate repeated memories of the same event before counting.

        CATEGORY MISMATCH DISCLOSURE FOR SOURCE QUESTIONS:
            If the user asks a source/giver question such as "from whom?", "who gave me?", or "who did I receive it from?", and the retrieved context does not contain an exact category match, do not silently substitute a different category.

            However, if there is exactly one received/got/acquired/was-given event on the resolved date with an explicit source person, answer transparently by naming the actual item and the source.

            Format:
            "The retrieved memory says you received [actual item] from [source person] on [date]. It is not [user's requested category], but that is the only received item found for that date."

            Do not use this rule for counts, lists, ordering, calculations, or object-identity questions.
    
    6. CHRONOLOGICAL INFERENCE (CRITICAL): If the user asks a sequence or date-gap question:
        - You MUST identify the anchor event and its exact narrative date from the memory content.
        - Bracketed STORED_AT timestamps are only storage recency metadata and must never be used as event chronology.
        - Scan retrieved context for events with explicit narrative dates before or after the anchor event.
        - If a required named event has no explicit narrative date, trigger REQUIRED_DATA instead of using storage time.

    7. PREFERENCE META-ANALYSIS & NEGATIVE CONSTRAINTS (CRITICAL): If the user asks an open-ended request for advice, recommendations, ideas, or suggestions (e.g., "Any advice?", "Can you recommend...", "Can you suggest some activities?", "What should I do?", "might find interesting"), you MUST structure your response as a meta-analysis of their preferences. 
       - Do NOT just list suggestions directly to the user.
       - You MUST format your answer in the 3rd person like this: "The user would prefer suggestions related to... They would not be interested in..." 
       - You must explicitly state what they want to avoid or dislike based on the retrieved context.

    8. HYPER-PERSONALIZATION & STATE TRANSITIONS (CRITICAL): If the user asks for advice, inspiration, or ideas (e.g., "Any advice?", "What should I look for?"), you MUST explicitly name-drop specific elements, projects, tools, or experiences from the retrieved context in your response. Do not give generic advice. Furthermore, if the context reveals the user is transitioning from an existing state to a new state (e.g., upgrading an asset, changing a routine, swapping a component), your advice MUST explicitly mention BOTH the old state/item and the target state/item. You must draw a direct comparison between the two to ground your advice in their specific personal history.


    CRITICAL EXECUTION PROTOCOL (CHAIN OF THOUGHT):
    Before answering the user, you MUST plan your response using a <thinking> block.
    
    1. Inside <thinking> ... </thinking>:
       - Analyze the user's core intent.

       - TARGET-FIRST FACT GATHERING:
        First bind the requested target fields from the user question: entity/name, role/title/status, organization, relationship label, event/artifact label, relation, time/scope, and answer type.
        Then scan retrieved memories line-by-line and extract only candidate facts that match those bound fields, or explicitly mark them as mismatched.
        If a number exists only under a mismatched target field, treat it as unusable for the answer.
        Never answer with a number just because it is present in retrieved context; the number must belong to the bound target.

       - NUMERIC EVIDENCE TABLE:
        For calculation questions, write one row per real-world event/item:
        Number | Actor/Entity | Action/Property | Event/Item | Exactness | Include/Exclude | Reason.
        Merge duplicate memories first. Actor/Entity must come from the number-bearing clause, not default to User. Include only rows whose actor, action/property, and exactness match the user's question.

       - TARGET MATCHING & CONTEXTUAL SKEPTICISM:
         Match the user's requested entity and relation at the intended granularity.
         Do not require a narrower subtype than the question asks for.
         For travel where/stay/planning/visiting questions, apply the same destination-vs-lodging distinction from TARGET BINDING. Do not reject a destination answer merely because no hotel/accommodation is named.
         Require hotel, resort, address, or exact accommodation only if the user explicitly asks for that narrower subtype.
         Reject a candidate only when it belongs to a different entity, trip, date scope, or storyline.

       - FALLBACK EVALUATION (MANDATORY): 
         1. Does the user's prompt ask "How many", "Total", "Count", "List all", or "What are all"?
            If YES: apply TARGET BINDING before counting.
            Count only values whose identity-bearing fields, relation, and time/scope match the requested target.
            If the only available value belongs to a different role, title, employer, relationship label, event, artifact, or storyline, answer that the information is not enough.
            Trigger REQUIRED_DATA only if no target-matching value was found and retrieval does not already show a mismatched target:
            - no directly relevant item/date/value was found,
            - the question asks for an exhaustive list across a broad category and retrieval appears incomplete,
            - a required variable is missing for the exact named target,
            - or multiple distinct entities could match the user's reference.
         2. Does the user's prompt ask for open-ended "recommendations", "suggestions", or things they "might find interesting"?
            If YES: You MUST trigger the REQUIRED_DATA JSON flag. Open-ended recommendation requests require a deep-dive fallback search to ensure all historical preferences (even older ones) are retrieved.

       - COMPREHENSIVE SYNTHESIS: If multiple memories mention different aspects of the same topic, you MUST merge ALL listed details into your final answer. Do not drop specific nouns, proper names, or titles assuming the newest memory is "good enough". Combine them to form a complete picture.

       - ADVICE & COMPARISONS: If the user asks for advice, recommendations, tips, or suggestions (e.g., "what to look for", "any ideas", "can you recommend"), you MUST check if the retrieved context mentions what they CURRENTLY own, use, or do. If they are upgrading or changing, you MUST physically write down: "Current State: [Item/Habit] -> Target State: [Item/Habit]" inside this thinking block. Your final response to the user MUST explicitly name-drop BOTH items/states and draw a direct comparison between them.

       - EVIDENCE SYNTHESIS:
        Merge semantic and episodic facts only after the requested target has been matched.
        The requested target includes entity/name, role/title/status, relation, time/scope, and answer type.
        Deduction may connect compatible facts for the same target and relation, but must not create, rename, or substitute identity-bearing fields.
        If a candidate answers the right relation but belongs to a different identity-bearing field, reject it and return insufficiency or trigger REQUIRED_DATA.
       
       - If it is a simple greeting, acknowledge it naturally without over-explaining technical facts.
       
       - Read the [PROCEDURAL] rules. State explicitly how you will alter your formatting or tone to obey them.

    2. MULTI-STEP TOOL USE (ReAct): If you have tools available, you can call them sequentially. If the result of a tool reveals that you need more information (e.g., an email thread references a different email), you should immediately call the tool again with the new parameters.
    3. After the </thinking> block and after ALL necessary tool calls are complete, provide your final response to the user in the required JSON format.
    4. TOOL USE: If you use a tool, you MUST provide a helpful response to the user AFTER the tool execution is complete (e.g., "I found your email..." or "I've updated your calendar"). NEVER respond with only tool calls; always speak to the user.
    5. PRESUPPOSITION CHECK:
        If the question assumes a current/past state, but retrieved memory contains a conflicting current state, explicitly correct the false premise before answering. Do not answer only as missing data.

    [REQUIRED_DATA] PROTOCOL & HYDE QUERY GENERATION (CRITICAL):
    You must maintain strict skepticism about your retrieved context. You MUST trigger the REQUIRED_DATA flag in your final JSON if you encounter ANY of the following gaps:
    - Missing Variables & Scope Mismatches: The user asks for a calculation or comparison for a SPECIFIC target. If your context contains the necessary numbers, but they belong to a DIFFERENT target or storyline than the one the user requested, you cannot use them. You must trigger REQUIRED_DATA to search for the correct target's exact variables.
    - Broad vs. Specific Gap: The user asks for a highly specific duration, phase, or item, but your context only provides the broad, overarching total (meaning a transition or sub-phase is missing).
    - Recommendation Blindspots: The user asks for open-ended recommendations, but you lack data on their explicit dislikes or past constraints.
    - Unresolved References & Scope Mismatch: The user asks about a specific reference using generic pronouns/terms (e.g., "my latest vehicle", "my current project") but the context only contains overarching categorical data (e.g., "has been driving for 10 years", "works in software").
    - Ambiguous References & Cross-Contamination (CRITICAL): The user asks about a generic location, event, or item (e.g., "the airport", "the hotel", "the wedding") and your retrieved context contains MULTIPLE distinct entities of that type (e.g., a trip to Mumbai AND a trip to Tokyo). You are FORBIDDEN from mixing facts, prices, or timelines between different storylines to force an answer. You must trigger REQUIRED_DATA to clarify which specific entity they mean.

    When triggering this protocol:
    - Leave "agent_response" empty.
    - Add "REQUIRED_DATA" to the "flags" array.
    - Provide exactly 3 highly specific keyword queries in the "hyde_queries" array (maximum 2-3 words per query).

    HOW TO CONNECT THE DOTS FOR QUERY GENERATION:
    1. Identify the logical gap between the user's prompt and your current knowledge.
    2. Deduce the exact transition, opposite, or missing component required to bridge that gap.
    3. Distill that deduction into targeted keywords.

    Examples of connecting the dots:
    - Gap (Entity Hijacking Prevention): User asks for the total cost of the "Marketing software". Context shows the "HR software" costs $500, but mentions no price for the Marketing software.
      Dot-Connecting: I cannot substitute the HR software price for the Marketing software just because they are both software. The exact data for the target entity is missing.
      Queries: ["marketing software cost", "marketing software price", "paid marketing software"]

    - Gap (Ambiguous Reference): User asks about "the airport". Context mentions Mumbai Airport and Narita Airport.
      Dot-Connecting: I cannot mix Mumbai prices with a Tokyo hotel. I need to know the specific airport or if the specific bus price exists for the target hotel.
      Queries: ["airport bus", "bus fare", "hotel transit"]

    - Gap (Exact Noun Mismatch): User asks about "my leased apartment". Context shows they lived in a "purchased condo", but a lease and a purchase are legally distinct.
      Dot-Connecting: I cannot assume the purchased condo is the leased apartment. I must search specifically for the leased apartment.
      Queries: ["leased apartment", "rented apartment", "leasing"]

    - Gap (Unresolved Reference): User asks about "my latest car". Context shows they have been driving for 10 years, but does not name a specific car model.
      Dot-Connecting: I cannot apply the overarching 10-year driving history to the "latest car" without knowing what the latest car is and when it was purchased. I need to find the specific car acquisition.
      Queries: ["new car", "bought vehicle", "latest vehicle"]
    
    - Gap (Missing Entity): User asks for total cost of Item A and Item B. Context only shows Item A.
      Dot-Connecting: I cannot do the math. I need the exact purchase event for Item B.
      Queries: ["bought [Item B]", "cost [Item B]", "paid [Item B]"]

    - Gap (Negative Constraints): User asks for new hobby recommendations. Context only shows what they currently do.
      Dot-Connecting: To give good advice, I need to know what they dislike, abandoned, or actively avoid so I don't recommend the wrong thing.
      Queries: ["dislikes", "quit doing", "avoids"]

    FINAL OUTPUT FORMAT (CRITICAL):
    After your <thinking> block, your final response MUST be a valid JSON object matching this schema. Do not output raw text outside of this JSON block.
    {{
        "agent_response": "Your final conversational answer to the user.",
        "flags": ["REQUIRED_DATA"], 
        "hyde_queries": ["keyword one", "keyword two", "keyword three"] 
    }}

    Example structure:
    <thinking>
    User intent: Wants to know about their previous vacation.
    Fact Gathering: 
    - Episodic Memory 1: User stayed in a cabin.
    - Semantic Memory 1: The vacation was in Lake Tahoe.
    - Episodic Memory 2: User went kayaking with their brother Dan.
    Comprehensive Synthesis: Merge all details. Vacation was in a cabin in Lake Tahoe, and involved kayaking with brother Dan.
    Procedural Rule: "Tone must be enthusiastic".
    </thinking>
    {{
        "agent_response": "Your last vacation was amazing! You stayed in a cabin in Lake Tahoe and went kayaking with your brother Dan.",
        "flags": [],
        "hyde_queries": []
    }}

    or 

    <thinking>
    User intent: Seeking advice on a new coffee machine.
    Fact Gathering: 
    - Semantic Memory: User currently uses a basic drip coffee maker.
    - Episodic Memory: User mentioned wanting to start making lattes at home.
    Advice & Comparisons: Current State: Basic drip coffee maker -> Target State: Espresso machine. I need to explicitly compare these two.
    </thinking>
    {{
        "agent_response": "Since you are looking to upgrade from your basic drip coffee maker, an entry-level espresso machine would be perfect. It will require a bit more manual work than your old drip machine, but it will finally allow you to make the lattes you've been wanting to try at home.",
        "flags": [],
        "hyde_queries": []
    }}

    or 

    <thinking>
    User intent: How many days did I spend in London and Tokyo?
    Fact Gathering: 
    - Episodic Memory: User spent 10 days in London.
    - Tokyo: Zero evidence found.
    Synthesis & Deduction: I am missing the data for Tokyo. I need to connect the dots: the user is asking about a specific travel duration to Tokyo. I will generate targeted keyword queries to find the missing Tokyo trip.
    </thinking>
    {{
        "agent_response": "",
        "flags": ["REQUIRED_DATA"],
        "hyde_queries": ["tokyo trip", "travel tokyo", "visited tokyo"]
    }}
    """

    final_user_prompt = f"""
    
    You have three types of contextual memory available:
        [RETRIEVED CONTEXT]
        [SEMANTIC - User Facts]: {state.get("semantic_context", "None")}
        [EPISODIC - Past Events]: {state.get("episodic_context", "None")}
        [PROCEDURAL - Strict Rules]: {state.get("procedural_context", "None")}

        [USER QUESTION] :
        {state["user_prompt"]}
        """

    if state.get("skill_markdown"):
        system_instruction += (
            f"\n\n[ACTIVE SKILL INSTRUCTIONS]:\n{state['skill_markdown']}"
        )

    messages = (
        [SystemMessage(content=system_instruction)]
        + list(state["chat_history"])
        + [HumanMessage(content=final_user_prompt)]
    )

    selected_skills = state.get("selected_skills", [])
    last_response = None

    if selected_skills:

        tools_list = []
        for skill in selected_skills:
            skill_data = tool_manager.load_skill(skill)
            tools_list.extend(skill_data["tools"])

        llm_with_tools = gen_llm.bind_tools(tools_list)

        # --- NEW: ReAct Loop ---
        MAX_ITERATIONS = 5  # Prevent infinite loops if the LLM gets stuck
        iteration = 0

        while iteration < MAX_ITERATIONS:

            # --- Pass 1: Intent & Initial Tool Call ---
            response = await llm_with_tools.ainvoke(messages)

            # Track Tokens
            m = get_token_metrics(response)
            in_tokens += m["input"]
            out_tokens += m["output"]
            last_response = response

            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                # If the LLM didn't call any tools, it means it has formulated its final answer. Break the loop.
                break

            # CRITICAL: Must append the LLM's request to history before executing
            messages.append(response)

            tool_msgs = []
            for call in tool_calls:
                fn = next((t for t in tools_list if t.name == call["name"]), None)
                if fn:
                    try:
                        print(
                            f"🔧 [Loop {iteration+1}] Executing Tool: {call['name']}..."
                        )

                        # 🚀 NEW: CACHE INVALIDATION INTERCEPTOR
                        # If the agent uses the identity update tool, wipe the cache!
                        if call["name"] == "update_agent_identity":
                            print(
                                "🔄 [Cache] Agent identity updated. Invalidating config cache."
                            )
                            AGENT_CONFIG_CACHE = None

                        # --- NEW: SECURE CONTEXT INJECTION ---
                        # Pass backend variables securely without exposing them to the LLM
                        run_config = {
                            "configurable": {
                                "user_id": state.get("user_id"),
                                "channel_type": state.get("channel_type"),
                            }
                        }

                        result = fn.invoke(call["args"], config=run_config)
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = "Tool not found."

                tool_msgs.append(
                    ToolMessage(content=str(result), tool_call_id=call["id"])
                )

            messages.extend(tool_msgs)

            iteration += 1

        # --- FIX FOR PROBLEM 2: FORCE TEXT GENERATION ON TIMEOUT ---
        if iteration == MAX_ITERATIONS:
            print(
                "⚠️ ReAct loop reached maximum iterations. Forcing final text generation."
            )
            # Unbind tools by using the plain `llm` so it is FORCED to output text
            messages.append(
                SystemMessage(
                    content="SYSTEM INSTRUCTION: You have reached the maximum allowed tool execution limit. You must immediately provide a final text response to the user summarizing what you accomplished and what you couldn't finish. Do NOT attempt to call any more tools."
                )
            )

            final_response = await gen_llm.ainvoke(messages)

            m_final = get_token_metrics(final_response)
            in_tokens += m_final["input"]
            out_tokens += m_final["output"]
            last_response = final_response

    else:
        # Normal execution
        response = await gen_llm.ainvoke(messages)
        m = get_token_metrics(response)
        in_tokens = m["input"]
        out_tokens = m["output"]
        last_response = response

    # =====================================================================
    # 🛠️ NEW: JSON PARSING & DYNAMIC REQUIRED_DATA FALLBACK
    # =====================================================================
    raw_text = str(last_response.content) 
    json_str = extract_json_block(raw_text)


     # 🐞 ADD THIS DEBUG BLOCK 🐞
    print("\n" + "="*50)
    print("🐞 [DEBUG] RAW PASS 1 OUTPUT (WITH THINKING):")
    print(last_response.content)
    print("="*50 + "\n")
    
    final_output = ""
    flags = []
    dynamic_queries = []

    try:
        if json_str:
            data = json.loads(json_str)
            final_output = data.get("agent_response", "")
            flags = data.get("flags", [])
            dynamic_queries = data.get("hyde_queries", []) or []
        else:
            final_output = extract_pure_text(last_response)
    except Exception as e:
        print(f"⚠️ [JSON Parse Error] Falling back to raw text. {e}")
        final_output = extract_pure_text(last_response)

    print("New dynamic Queries:")
    print(dynamic_queries)

    if "REQUIRED_DATA" in flags and dynamic_queries:
        print(f"🔄 [Self-Correction] Agent requested specific data. Queries: {dynamic_queries}")
        
        # Run targeted search using the LLM's exact keyword pairs
        fb_sem_tasks = [semantic_db.search(sq, top_k=20, threshold=0.28) for sq in dynamic_queries]
        fb_epi_tasks = [episodic_db.search(sq, top_k=20, threshold=0.28) for sq in dynamic_queries]
        
        fb_sem_results = await asyncio.gather(*fb_sem_tasks)
        fb_epi_results = await asyncio.gather(*fb_epi_tasks)
        
        def fallback_dedupe(results_list):
            unique = {}
            for block in results_list:
                if not block: continue
                for line in block.split('\n'):
                    match = re.search(r"\[ID: ([0-9a-fA-F\-]{36})\]", line)
                    if match: unique[match.group(1)] = line
            return "\n".join(unique.values())
        
        # 🛠️ FIXED: Pass both the new results AND the old context into the deduplicator!
        combined_sem = fallback_dedupe(fb_sem_results + [sem_ctx_to_save])
        combined_epi = fallback_dedupe(fb_epi_results + [epi_ctx_to_save])

        new_sem_ctx = sort_memories_by_recency(combined_sem, max_lines=70)
        new_epi_ctx = sort_memories_by_recency(combined_epi, max_lines=70)

        # Update context variables for state tracking
        sem_ctx_to_save = new_sem_ctx
        epi_ctx_to_save = new_epi_ctx


        # Force the LLM to answer using the expanded context
        override_command = """
CRITICAL OVERRIDE: You are currently in FALLBACK mode. You have been provided with the expanded database search you requested. You are FORBIDDEN from using the "REQUIRED_DATA" flag again. 

FINAL VERIFICATION (STRICT ENTITY ISOLATION): Look at the expanded data provided. Does it contain the EXACT required variables explicitly linked to the specific target requested by the user?
- EAGER MATCHING BAN: You are STRICTLY FORBIDDEN from taking data (dates, numbers, facts) attached to a generic, unnamed, or broadly described entity and applying it to a highly specific Proper Noun or named entity. Even if contextual clues strongly suggest they might be the same thing, you cannot merge them to force an answer.
- CATEGORY MISMATCH DISCLOSURE: For source/giver questions only, if no exact category match exists but exactly one date-matched received/got/acquired/was-given event exists with an explicit source person, answer transparently with the actual item and source. Do not claim the item belongs to the user's requested category.
- If YES (the required data is explicitly attached to the EXACT named target): Provide your final answer in the "agent_response" field.
- If NO (the exact target is STILL missing its required variables, or the variables are only attached to generic/unnamed entities): You MUST explicitly state that the information provided is not enough to answer the question. Do not guess or assume.
"""

        new_user_prompt = f"""
[RETRIEVED CONTEXT (EXPANDED SEARCH)]
[SEMANTIC - User Facts]: {sem_ctx_to_save}
[EPISODIC - Past Events]: {epi_ctx_to_save}
[PROCEDURAL - Strict Rules]: {state.get("procedural_context", "None")}

[USER QUESTION]
{state["user_prompt"]}

{override_command}
"""
        new_messages = [SystemMessage(content=system_instruction)] + list(state["chat_history"]) + [HumanMessage(content=new_user_prompt)]
        
        print("🧠 [Self-Correction] New context loaded. Re-generating JSON response...")
        fallback_response = await gen_llm.ainvoke(new_messages)
        
        # Update metrics
        m_fb = get_token_metrics(fallback_response)
        in_tokens += m_fb["input"]
        out_tokens += m_fb["output"]

        # 🐞 ADD THIS DEBUG BLOCK 🐞
        print("\n" + "="*50)
        print("🐞 [DEBUG] RAW PASS 2 (FALLBACK) OUTPUT:")
        print(fallback_response.content)
        print("="*50 + "\n")
        
        # Parse the Fallback JSON
        fb_json_str = extract_json_block(str(fallback_response.content))
        try:
            if fb_json_str:
                fb_data = json.loads(fb_json_str)
                final_output = fb_data.get("agent_response", "")
            else:
                final_output = extract_pure_text(fallback_response)
        except Exception:
            final_output = extract_pure_text(fallback_response)

    # Clean up any residual markdown or thinking blocks inside the agent_response string
    final_output = re.sub(r"<thinking>.*?</thinking>", "", final_output, flags=re.DOTALL).strip()

    if not final_output:
        final_output = "I have processed that request using my tools, but I don't have a specific summary to display. Please let me know if you need anything else."

    print(f"Agent Response :{final_output}")

    end_time = time.time()
    print(
        f"\n⏱️ [Metrics] Generation Time: {end_time - start_time:.2f}s | Tokens: In({in_tokens}) Out({out_tokens})"
    )

    current_metrics = state.get("metrics", {})
    current_metrics.update({"generation_in": in_tokens, "generation_out": out_tokens})

    # 🚀 NEW: FIRE BACKGROUND LEARNING TO REDUCE LATENCY
    if not state.get("skip_learning", False):

        
        # Define a safe wrapper that respects the 4-task limit
        async def bounded_learning():
            async with LEARNING_SEMAPHORE:
                await background_memory_update(
                    state["user_prompt"],
                    final_output,
                    sem_ctx_to_save,  # 🛠️ Pass updated Semantic context here
                    epi_ctx_to_save,  # 🛠️ Pass updated Episodic context here
                    state.get("procedural_context", ""),
                    state.get("current_date")
                )
        
        # Fire-and-forget, but track it in the global set
        task = asyncio.create_task(bounded_learning())
        PENDING_LEARNING_TASKS.add(task)
        task.add_done_callback(PENDING_LEARNING_TASKS.discard)

    # 🛠️ RETURN THE UPDATED CONTEXTS SO LANGGRAPH STATE UPDATES EVERYWHERE
    return {
            "final_response": final_output, 
            "metrics": current_metrics,
            "semantic_context": sem_ctx_to_save,
            "episodic_context": epi_ctx_to_save
        }


# ==========================================
# 7. LEARNING NODE (Robust Extraction)
# ==========================================
@persistent_network_retry(initial_delay=3.0, max_delay=60.0, timeout=None) # 🛠️ NEVER DROP
async def learn_vector_memory(
    db: VectorMemoryManager,
    memory_type: str,
    user_prompt: str,
    ai_response: str,
    current_context: str,
    current_date: str = None, 
    related_context: str = "",
) -> tuple[str, dict]:
    """Extracts facts/episodes with strict isolation and an importance threshold."""

    if memory_type == "Semantic":
        definition = (
            "Atomic, standalone facts that build the user's core identity profile. These are permanent or long-term "
            "attributes (e.g., name, age, demographics, ethnicity, origins, job title, health/dietary needs, specific tech/creative preferences, relationships, core routines and many more such things)."
            "Extract any enduring, long-term information that defines WHO the user is, WHAT they do, or HOW they live."
        )
        instruction_extension = (
            "ATOMICITY & ENTITY-PRESERVATION RULE (CRITICAL): Separate unrelated facts, BUT NEVER sever relational links. "
            "NEVER drop or omit any proper nouns, names of people, or specific places mentioned by the user. If the user mentions a couple or multiple people (e.g., 'Jen and Tom', 'Rachel and Mike'), you MUST include ALL NAMES in the extracted memory. " # 🛠️ ADDED THIS SENTENCE
            "IMPLIED BACKGROUND FACTS & ALTERNATIVES (CRITICAL): You MUST extract facts, locations, and affiliations that are implied or mentioned in passing. "
            "Humans often state facts relationally. If a user says 'I use my Nespresso machine when I don't have time to walk to Blue Bottle Coffee', you MUST extract TWO separate facts: 1. 'User uses a Nespresso machine at home.' 2. 'User goes to Blue Bottle Coffee.' "
            "NEVER ignore physical locations, studios, workplaces, or secondary entities mentioned in subordinate clauses or as fallback options. "
            "HEURISTIC: If the information reveals the user's background, origins, physical/mental traits, or strict lifestyle parameters (things that will likely still be true 5 years from now), you MUST extract it as a standalone fact. "
            "Additionally, extract CURRENT ACTIVE STATUSES or TRACKED INVENTORIES. "
            "Even if a status is not permanent, if it involves a specific count, a pending logistical requirement, or a multi-stage endeavor, it must be captured as a fact. "
            "Example: If the user mentions a specific number of items in a certain stage of a process, or a count of assets they are managing, record that specific quantity and status."
            "If the user says 'I am 22 and my friend Sarah likes destiny', issue TWO 'ADD' actions: 1. 'User is 22 years old.' 2. 'User has a friend named Sarah who is interested in destiny.' "
            "NEVER replace specific names with generic pronouns like 'a friend' or 'they'."
        )
        example = (
            '{ "action": "ADD", "content": "User\'s name is Sourav." }, '
            '{ "action": "ADD", "content": "User works at Quarq Labs." }'
            '{ "action": "ADD", "content": "User goes to Blue Bottle Coffee." }, '
            '{ "action": "ADD", "content": "User uses a Nespresso machine at home." }, '
            '{ "action": "ADD", "content": "User\'s high school friend is named Sarah." }, '
        )
        exclusion_rule = "DO NOT extract temporary states, emotions, or the narrative of the conversation. DO NOT extract greetings, pleasantries, or basic conversational filler."
    else:
        definition = (
            "The narrative occurrence of the conversation. Focus on WHAT happened during the "
            "interaction (e.g., User introduced themselves, Agent provided code, User corrected a mistake)."
        )
        instruction_extension = (
            "EVENT-ONLY & ENTITY-PRESERVATION RULE (CRITICAL): Focus on the milestone achieved in the conversation. "
            "HOWEVER, you MUST preserve specific names, identities, dates, locations, and entities within the event. "
            "If a date is available in the interaction and is tied to a specific event, or can be resolved from the provided current system time for a direct present-day phrase, attach that date directly to the exact event being stored. "
            "Never store a named event as undated if the interaction provides a date for it. "
            "Do not duplicate facts that belong in Semantic memory."
        )
        example = (
            '{ "action": "ADD", "content": "User introduced themselves for the first time and established their professional role." }'
            '{ "action": "ADD", "content": "User transitioned from exploring existential theory to actively applying meaning after a conversation with their friend Sourav." }'
            '{ "action": "ADD", "content": "User is doing xyz project about xxx" }'
        )
        exclusion_rule = "CRITICAL: DO NOT extract temporary states, greetings, or small talk. (Note: You ARE allowed and encouraged to include specific names, brands, or places if they are directly involved in the event)."

    # 🛠️ FIXED: Use the provided date
    if current_date:
        current_time = current_date
        # Extract just the year from the string if possible, else fallback to system year
        try:
            current_year = re.search(r"\d{4}", current_date).group()
        except Exception:
            current_year = datetime.now().year
    else:
        current_year = datetime.now().year 
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    artifact_context = re.sub(r"\s+", " ", user_prompt).strip()
    artifact_context = artifact_context[:220] if artifact_context else "the generated artifact"

    artifact_source_text = f"{user_prompt}\n\n{ai_response}"

    structured_artifact_memories = (
        extract_structured_artifact_memories(
            artifact_source_text,
            current_time,
            artifact_context,
        )
        if memory_type == "Episodic"
        else []
    )

    structured_artifact_section = (
        "\n".join(f"- {m}" for m in structured_artifact_memories)
        if structured_artifact_memories
        else "None"
    )

    learning_prompt = f"""
    You are a Cognitive Memory Editor managing a {memory_type} database.
    Your job is to consolidate information by issuing ADD, UPDATE, or DELETE commands.

    DEFINITION OF {memory_type.upper()} MEMORY:
    {definition}

    {instruction_extension}

    CURRENT ACTIVE MEMORIES (With Database IDs):
    {current_context if current_context else 'None'}

    CURRENT RELATED MEMORIES FROM OTHER MEMORY TYPES (READ-ONLY):
    {related_context if related_context else 'None'}
    
    Interaction:
    User: {user_prompt}
    AI: {ai_response}

    STRUCTURED ARTIFACT UNITS:
    {structured_artifact_section}

    If STRUCTURED ARTIFACT UNITS is not None, each listed unit is mandatory evidence.
    Preserve each unit as its own ADD action unless the same unit already exists and should be updated.
    Do not replace structured artifact units with only a broad summary.

    CRITICAL TIME RESOLUTION: The current system time is {current_time}. 
    If the user uses relative time words (e.g., "yesterday", "last month", "tomorrow"), you MUST convert them into absolute dates within the "content" string you generate.
    If the user mentions an event but doesn't specify the year (e.g., "in August", "for my birthday"), you MUST append the current year ({current_year}) to the date. Example: "in August" -> "in August {current_year}". 
    Example: If user says "I started a diet yesterday", store "User started a diet on [Calculated Date]."

    CURRENT TURN EVENT DATE ANCHORING:
    The current system time is the timestamp of the conversation being learned.
    If the original conversation text describes a specific user event as happening today, now, just now, earlier today, recently completed, or just completed, and no explicit event date is written, use the current system date as the narrative date for that exact event.
    For Episodic memory, attach that date directly to the exact named event/entity in the same sentence.
    Do not use the current system date merely because the wrapper instruction says to review, remember, ingest, or summarize the conversation.

    TRANSFER / ACQUISITION EVENT ANCHORING:
    When the user says they got, received, acquired, bought, found, inherited, borrowed, were given, picked up, started, joined, or completed something, treat that as a concrete event.

    If the acquisition/transfer/completion phrase uses "today", "now", "just", "recently", "this morning", "this week", or another relative time phrase, resolve it using the current system time and attach the resolved date directly to the event.

    Preserve all transfer anchors:
    - recipient/actor,
    - item/object,
    - source/giver/seller/place if present,
    - date/time,
    - provenance/background if present.

    Do not weaken a dated transfer event into only a timeless ownership fact.

    General example:
    User text: "I got a vintage camera from my cousin today; it belonged to my grandfather."
    Correct memory: "On [current date], user got a vintage camera from their cousin that belonged to their grandfather."
    Wrong memory: "User owns a vintage camera that belonged to their grandfather."

    STRICT EXTRACTION & CONSOLIDATION RULES:
    1. ATOMICITY: Each 'ADD' or 'UPDATE' action must contain exactly ONE independent piece of information.
    2. TYPE ISOLATION: {exclusion_rule}
    3. HIGH-FIDELITY: Preserve specific entities, brand names, and proper nouns (for Semantic). 
    4. NEW INFO: If new information is provided that doesn't exist, use "ADD".
    5. DUPLICATE & UPDATE DECISION:
    Before issuing an ADD, scan both CURRENT ACTIVE MEMORIES and CURRENT RELATED MEMORIES FROM OTHER MEMORY TYPES.

    Treat memories as the same real-world event/item when they share the same canonical anchors:
    - actor/entity,
    - action/property,
    - event/item/object,
    - date/time,
    - location,
    - named counterpart or organization.

    Ignore wording differences, memory type labels, and exact-vs-qualified numeric wording when deciding whether two memories refer to the same real-world event/item.

    If the same event/item already exists in the current memory type, use UPDATE instead of ADD.
    If the same event/item exists only in another memory type, do not create a conflicting numeric version. Preserve the most precise known numeric form if you create or update a memory in this memory type.

    
    6. CORRECTIONS, EVOLUTIONS & CONTEXTUAL MERGING (CRITICAL): 
       - If the user provides new details about an existing topic, use "UPDATE" with the ID of the old memory to combine the facts. 
       - If the user mentions an event, project, or item, and then provides specific details (e.g., duration, cost, names) about it in subsequent sentences within the chunk, you MUST merge all of those details into a SINGLE comprehensive sentence.
       - NOUN & REFERENT RECOVERY (CRITICAL): Users often use casual or incomplete grammar, leaving adjectives or quantifiers hanging without their noun (e.g., "the blue one", "the 400-page"). If you see a lone descriptor or measurement, you MUST look at the surrounding context to find the specific noun it describes and explicitly attach it when merging. Example: "I hated the 3-hour" -> "User hated the 3-hour movie." You MUST preserve all specific entities and names. Never delete a name or detail just because the user didn't repeat it in the next sentence!
    7. DATE PRESERVATION & EVENT ANCHORING (CRITICAL):
        If the interaction contains any explicit date, resolved relative date, weekday, month, year, or temporal phrase tied to an event, the extracted memory MUST preserve that date in the same sentence as the exact named event/entity.
        Do not store a named event without its date when the date is available anywhere in the interaction.
        Do not move a date onto a generic summary if a specific named event is present.
        For Episodic memory, event memories should generally begin with the narrative date when one is known.
        If multiple events happen on different dates, create separate actions or an UPDATE that preserves each event with its own date.
        If an item/object is received, acquired, bought, found, inherited, borrowed, or given on a known date, the extracted memory must preserve that date with the acquisition/transfer verb, not only with later discussion, research, maintenance, or ownership.

    8. QUANTITATIVE FIDELITY:
    Preserve every number with its owner, measured action/property, event/item, and qualifier. Do not drop numbers during ADD or UPDATE.
    If the user's text contains a number or measurement (e.g., "5-hour", "$40", "3 weeks", "12 items"), that exact number must appear in the extracted memory. Resolve dangling numbers to their noun using surrounding context.
    Values in subordinate or relative clauses belong to the grammatical subject of that clause, not automatically to the user or nearest person.

    9. STATE-TRANSITION FIDELITY:
    If the user says they started, joined, signed up for, subscribed to, tried, began using, installed, adopted, bought access to, or first used something, preserve that onset verb and resolved date. Do not compress start/use transitions into passive ownership or access facts.

    10. NUMERIC PRECISION CONSOLIDATION:
    For the same real-world event/item, prefer the most precise numeric form:
    exact unqualified value > qualified value ("about", "over", "at least") > range > vague quantity.

    Never replace an existing exact unqualified value with a less precise restatement.
    If new text repeats the same event/item with a less precise number and adds no new non-numeric details, output no action.
    If new text adds useful details, update the memory while preserving the most precise known number and the new details.

    General example:
    Old memory: "User bought a monitor for $300 on May 2, 2023."
    New text: "The monitor cost about $300 and has a USB-C hub."
    Correct update: "User bought a monitor for $300 on May 2, 2023; it was later described as costing about $300 and has a USB-C hub."
    Wrong update: "User bought a monitor for about $300."

    11. REVOCATION: If the user explicitly revokes information, use "DELETE" on the old ID.
    12. NO PREFIXES: Do not use "Semantic:" or "Episodic:" labels in the content.
    13. NO GREETINGS: Ignore "Hello", "How are you", etc.

    OUTPUT FORMAT:
    You must return a raw JSON object with an "actions" array. Return exactly `{{"actions": []}}` if no changes are needed.
    {{
        "actions": [
            {{ "action": "ADD", "content": "Fact or Event 1" }},
            {{ "action": "ADD", "content": "Fact or Event 2" }},
            {{ "action": "UPDATE", "id": "uuid", "content": "Updated Fact preserving old names" }},
            {{ "action": "DELETE", "id": "uuid" }}
        ]
    }}
    Example for this task:
    {example}
    """

    response = await learn_llm.ainvoke([HumanMessage(content=learning_prompt)])


    # 🛠️ FIXED: Use robust extractor
    content = extract_pure_text(response)

    # print("learning content:")
    # print(content)

    # print("data:",content, current_context)
    m = get_token_metrics(response)


    actions_executed = 0

    if (
        (content and content.upper() != "NONE" and content != '{"actions": []}')
        or (memory_type == "Episodic" and structured_artifact_memories)
    ):
        try:
            # Isolate the JSON object explicitly
            json_str = extract_json_block(content, is_array=False)
            data = json.loads(json_str) if json_str else {"actions": []}
            actions = data.get("actions", [])

            # print("actions:")
            # print(actions)

            if memory_type == "Episodic":
                existing_contents = {
                    str(act.get("content", "")).strip()
                    for act in actions
                    if isinstance(act, dict)
                }
                for memory in structured_artifact_memories:
                    if memory not in existing_contents:
                        actions.append({"action": "ADD", "content": memory})
                        existing_contents.add(memory)

            for act in actions:

                await db.execute_action(act)
                actions_executed += 1

        except json.JSONDecodeError:
            # 2. FALLBACK PARSER: If the LLM just dumped raw text instead of JSON
            print(
                f"⚠️ [Warning] {memory_type} Editor returned raw text, falling back to ADD action."
            )

            # Strip accidental prefixes
            clean_content = re.sub(
                r"^(Semantic|Episodic):\s*",
                "",
                content,
                flags=re.IGNORECASE | re.MULTILINE,
            )

            lines = clean_content.split("\n")

            for line in lines:
                if line.strip():
                    fallback_act = {
                        "action": "ADD",
                        "content": line.strip(),  # 🛠️ REMOVED THE TIMESTAMP INJECTION LOGIC HERE
                    }
                    await db.execute_action(fallback_act)
                    actions_executed += 1

        except Exception as e:
            print(f"❌ [Error] Memory Execution failed: {e}")
            raise e  # 🛠️ ADD THIS: Signal the decorator to retry the whole process

    return actions_executed, m


async def execute_procedural_action(act: dict) -> bool:
    act_type = act.get("action", "").upper()
    raw_id = act.get("id")

    record_id = None
    if raw_id:
        match = re.search(r"([0-9a-fA-F\-]{36})", raw_id)
        if match:
            record_id = match.group(1)

    def mutate_rules():
        rules = _load_rules_file()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if act_type == "DELETE" and record_id:
            new_rules = [r for r in rules if r.get("id") != record_id]
            _save_rules_file(new_rules)
            return len(new_rules) != len(rules)

        if act_type == "ADD" and act.get("rule"):
            rules.append(
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": AGENT_ID,
                    "rule": act.get("rule"),
                    "reasoning": act.get("reasoning", ""),
                    "target_entity": act.get("target_entity", "global"),
                    "tags": act.get("tags", ["general"]),
                    "created_at": now,
                    "updated_at": now,
                }
            )
            _save_rules_file(rules)
            return True

        if act_type == "UPDATE" and record_id and act.get("rule"):
            for rule in rules:
                if rule.get("id") == record_id:
                    rule["rule"] = act.get("rule")
                    rule["reasoning"] = act.get("reasoning", "")
                    rule["target_entity"] = act.get("target_entity", "global")
                    rule["tags"] = act.get("tags", ["general"])
                    rule["updated_at"] = now
                    _save_rules_file(rules)
                    return True

        return False

    return await asyncio.to_thread(mutate_rules)



@persistent_network_retry(initial_delay=3.0, max_delay=60.0, timeout=None) # 🛠️ NEVER DROP
async def learn_procedural_memory(
    user_prompt: str, ai_response: str, current_context: str
) -> tuple[int, dict]:
    """Extracts, updates, or deletes explicit behavioral rules with strict formatting constraints."""

    learning_prompt = f"""
    You are a Cognitive Procedural Database Editor.
    Your job is to identify, consolidate, and clean up behavioral rules, personas, and formatting constraints.

    CRITICAL BOUNDARIES (WHAT NOT TO EXTRACT):
    1. DO NOT extract factual information about the user (e.g., their name, job, or projects). That belongs in Semantic Memory.
    2. DO NOT extract specific details of the current task (e.g., "The user needed an email about being late"). That belongs in Episodic Memory.
    3. ONLY extract a rule if the user expressed a general preference for HOW you should behave, format, or generate outputs in the future (e.g., "Always use a formal tone", "Never use emojis", "Format code in snake_case").

    ENTITY SPECIFICITY (CRITICAL RULE):
    - If the user specifies that a rule applies to a specific person (e.g., "Elias"), project, or context, you MUST explicitly include that name/context at the very beginning of the rule text (e.g., "When generating content for Elias..."). Set the "target_entity" field to this name.
    - If no target is specified, leave "target_entity" as "global".

    CURRENT ACTIVE RULES (With Database IDs):
    {current_context if current_context else 'None'}
    
    Interaction:
    User: {user_prompt}
    AI: {ai_response}

    STRICT CONSOLIDATION RULES:
    1. If the user establishes a NEW rule, use "ADD".
    2. CONTRADICTION REMOVAL & LOSSY-UPDATE PREVENTION (CRITICAL): If the user corrects a past rule (e.g., "Stop calling me that" or fixing a typo like 'todasy' to 'today'), you MUST use "UPDATE" and provide the ID of the old rule. When updating, you MUST preserve all existing specific entities and examples from the old rule.  DO NOT "ADD" the correction, as leaving the old rule active will confuse the generation agent.
         - Example: If an old rule says "Avoid allergens (e.g., peanuts, dairy)" and the user adds "gluten", the updated rule MUST say "(e.g., peanuts, dairy, gluten)". Do not drop old examples unless explicitly revoked.
    3. If the user explicitly revokes a rule, use "DELETE" on the old ID.

    OUTPUT FORMAT:
    Return a raw JSON object with an "actions" array. Return exactly `{{"actions": []}}` if no changes are needed.
    {{
        "actions": [
            {{
                "action": "ADD", 
                "rule": "[Timestamp] New rule here.",
                "target_entity": "global",
                "tags": ["formatting"],
                "reasoning": "Why this rule was added."
            }},
            {{
                "action": "UPDATE", 
                "id": "uuid-from-context",
                "rule": "[Timestamp] Corrected rule here.",
                "target_entity": "global",
                "tags": ["formatting"],
                "reasoning": "Why it was updated."
            }},
            {{
                "action": "DELETE", 
                "id": "uuid-to-delete",
                "reasoning": "Why it was removed."
            }}
        ]
    }}
    """

    response = await learn_llm.ainvoke([HumanMessage(content=learning_prompt)])
     # 🛠️ FIXED: Use robust extractor
    content = extract_pure_text(response)
    m = get_token_metrics(response)

    

    actions_executed = 0

    if content and content.upper() != "NONE" and content != '{"actions": []}':
        try:
            # Isolate the JSON object explicitly
            json_str = extract_json_block(content, is_array=False)
            data = json.loads(json_str)
            actions = data.get("actions", [])

            for act in actions:
                act_type = act.get("action", "").upper()

                changed = await execute_procedural_action(act)
                if changed:
                    print(f"✅ [Rules] {act_type} completed locally.")
                    actions_executed += 1

        except json.JSONDecodeError:
            # 2. FALLBACK PARSER: If the LLM dumped raw text instead of JSON
            print(
                "⚠️ [Warning] Procedural Editor returned raw text, falling back to basic ADD action."
            )

            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            lines = content.split("\n")

            for line in lines:
                if line.strip() and len(line.strip()) > 10:
                    fallback_act = {
                        "action": "ADD",
                        "rule": f"{timestamp} {line.strip()}",
                        "reasoning": "Fallback extraction",
                        "target_entity": "global",
                        "tags": ["fallback"],
                    }
                    changed = await execute_procedural_action(fallback_act)
                    if changed:
                        actions_executed += 1

        except Exception as e:
            print(f"❌ [Error] Rule Execution failed: {e}")
            raise e  # 🛠️ ADD THIS: Signal the decorator to retry the whole process

    return actions_executed, m


async def update_memories_node(state: AgentState):
    start_time = time.time()

    u_prompt = state["user_prompt"]
    ai_resp = state["final_response"]

    results = await asyncio.gather(
        learn_vector_memory(
            semantic_db,
            "Semantic",
            u_prompt,
            ai_resp,
            state.get("semantic_context", ""),
            state.get("current_date"),
        ),
        learn_vector_memory(
            episodic_db,
            "Episodic",
            u_prompt,
            ai_resp,
            state.get("episodic_context", ""),
            state.get("current_date"),
        ),
        learn_procedural_memory(u_prompt, ai_resp, state.get("procedural_context", "")),
    )

    # Unpack the tuples
    sem_content, sem_tokens = results[0]
    epi_content, epi_tokens = results[1]
    pro_content, pro_tokens = results[2]

    # Unpack and Sum
    total_in = sum(r[1]["input"] for r in results)
    total_out = sum(r[1]["output"] for r in results)

    # Fetch existing metrics (from retrieval + generation) and add learning stats
    current_metrics = state.get("metrics", {})
    current_metrics.update({"learning_in": total_in, "learning_out": total_out})

    end_time = time.time()  # END TIMER

    print("\n--- Memory Learning Complete ---")
    if sem_content:
        print(f"💡 Learned Semantic: {sem_content}")
    if epi_content:
        print(f"💡 Learned Episodic: {epi_content}")
    if pro_content:
        print(f"💡 Learned Procedural: {pro_content}")
    if not any([sem_content, epi_content, pro_content]):
        print("No new memories learned this turn.")

    print(
        f"⏱️ [Metrics] Learning Time: {end_time - start_time:.2f}s | Total Tokens: In({total_in}) Out({total_out})"
    )
    print("--------------------------------\n")
    return {"metrics": current_metrics}


# ==========================================
# 7b. BACKGROUND LEARNING PROCESS
# ==========================================
async def background_memory_update(
    user_prompt: str,
    ai_resp: str,
    semantic_ctx: str,
    episodic_ctx: str,
    procedural_ctx: str,
    current_date: str = None, # 🛠️ ADDED
):
    """Runs memory extraction silently in the background so the user doesn't wait."""
    start_time = time.time()

    semantic_related_ctx = f"[Episodic]\n{episodic_ctx}" if episodic_ctx else ""
    episodic_related_ctx = f"[Semantic]\n{semantic_ctx}" if semantic_ctx else ""

    results = await asyncio.gather(
        learn_vector_memory(
            semantic_db, "Semantic", user_prompt, ai_resp, semantic_ctx, current_date,semantic_related_ctx
        ),
        learn_vector_memory(
            episodic_db, "Episodic", user_prompt, ai_resp, episodic_ctx, current_date,episodic_related_ctx
        ),
        learn_procedural_memory(user_prompt, ai_resp, procedural_ctx),
    )

    # print("result:" ,results)

    # Unpack the tuples
    sem_content, sem_tokens = results[0]
    epi_content, epi_tokens = results[1]
    pro_content, pro_tokens = results[2]

    total_in = sum(r[1]["input"] for r in results)
    total_out = sum(r[1]["output"] for r in results)

    end_time = time.time()

    print("\n--- Background Memory Learning Complete ---")
    if sem_content:
        print(f"💡 Learned Semantic: {sem_content}")
    if epi_content:
        print(f"💡 Learned Episodic: {epi_content}")
    if pro_content:
        print(f"💡 Learned Procedural: {pro_content}")
    if not any([sem_content, epi_content, pro_content]):
        print("No new memories learned this turn.")

    print(
        f"⏱️ [Metrics] Background Learning Time: {end_time - start_time:.2f}s | Total Tokens: In({total_in}) Out({total_out})"
    )
    print("--------------------------------\n")


# ==========================================
# 8. BUILD GRAPH & INTERFACES
# ==========================================
def route_after_generation(state: AgentState):
    return END if state.get("skip_learning", False) else "update_memories"


workflow = StateGraph(AgentState)
workflow.add_node("retrieve_memories", retrieve_memories_node)
workflow.add_node("route_tools", route_tools_node)  # NEW
workflow.add_node("generate_response", generate_response_node)
# workflow.add_node("update_memories", update_memories_node)

workflow.add_edge(START, "retrieve_memories")
workflow.add_edge("retrieve_memories", "route_tools")  # NEW PATH
workflow.add_edge("route_tools", "generate_response")  # NEW PATH
# workflow.add_conditional_edges("generate_response", route_after_generation)
# workflow.add_edge("update_memories", END)
workflow.add_edge("generate_response", END)  # 🚀 Returns to the user instantly!

app = workflow.compile()


async def main_chat_loop():
    print(
        "🤖 Quarq Agent  V3 - Cognitive Memory Editor, Temporal Truth Protocol, and Background Learning"
    )
    chat_history = []
    while True:
        u_input = input("\nYou: ")
        if u_input.lower() in ["exit", "quit"]:
            break

        initial_state: AgentState = {
            "user_prompt": u_input,
            "chat_history": chat_history,
            "semantic_context": "",
            "episodic_context": "",
            "procedural_context": "",
            "selected_skills": [],
            "skill_markdown": "",
            "final_response": "",
            "skip_learning": False,
            "user_id": "local_terminal_user",  # ADDED
            "channel_type": "terminal",  # ADDED
            "metrics": {},
        }
        final_state = await app.ainvoke(initial_state)
        print(f"\n🤖 Agent: {final_state['final_response']}")

        chat_history.extend(
            [
                HumanMessage(content=u_input),
                AIMessage(content=final_state["final_response"]),
            ]
        )


if __name__ == "__main__":
    asyncio.run(main_chat_loop())
