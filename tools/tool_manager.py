"""Tool manager — the router between the main agent and its skills.

Design:
  1. On import, scan `tools/` and build a skill registry.
  2. `select_skills()` makes ONE LLM call with a catalog of one-liners and
     returns the chosen skills name (or None). 
  3. `load_skill()` returns the full skill.md + tool list for the chosen skill,
     which the main graph then feeds to the generate node.

The whole point of this split is progressive disclosure: the router never
sees full skill docs, only one-liners. The main LLM only sees the full doc
of the skill actually chosen. Scales to many skills without context bloat.
"""
import os
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from tools import discover_skills


load_dotenv()

_raw_api_key = os.getenv("OPENAI_API_KEY")
if not _raw_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment.")

# Registry built once at import time. Adding a new skill requires a process
# restart — fine for a prototype; hot-reload can be bolted on later.
_SKILLS = discover_skills()
_LAST_ROUTER_METRICS = {"input": 0, "output": 0, "total": 0}

# Dedicated router model. Kept cheap/fast because it only ever emits a single word.
_router_llm = ChatOpenAI(
   api_key=SecretStr(_raw_api_key),
    temperature=0,
    model="gpt-4o-mini",
)


def _get_token_metrics(response) -> dict:
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
        if isinstance(usage, dict):
            metrics["input"] = usage.get("prompt_tokens", 0)
            metrics["output"] = usage.get("completion_tokens", 0)
            metrics["total"] = usage.get("total_tokens", 0)

    return metrics


def get_last_router_metrics() -> dict:
    return dict(_LAST_ROUTER_METRICS)


def _skills_matching_triggers(user_prompt: str) -> list[str]:
    prompt = re.sub(r"\s+", " ", str(user_prompt or "").lower()).strip()
    if not prompt:
        return []

    matches = []
    for name, info in _SKILLS.items():
        raw_triggers = str(info.get("triggers") or "")
        triggers = [trigger.strip().lower() for trigger in raw_triggers.split(",")]
        if any(trigger and trigger in prompt for trigger in triggers):
            matches.append(name)
    return matches



# ==========================================
# 🛠️ HELPER: ROBUST CONTENT EXTRACTION 
# ==========================================
def extract_pure_text(response) -> str:
    """Safely extracts plain text from an LLM response, stripping thinking blocks and formatting."""
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



async def select_skills(user_prompt: str, recent_history: str = "", memory_context: str = "") -> list[str]:
    """Single-shot multi-router. Returns a list of chosen skill names or None.

    Given the catalog of skills (just one-liner descriptions), ask the LLM
    which skills — if any — the user's prompt calls for. Returns None for
    conceptual questions, chit-chat, or anything that doesn't require an action.
    """
    global _LAST_ROUTER_METRICS
    _LAST_ROUTER_METRICS = {"input": 0, "output": 0, "total": 0}

    if not _SKILLS:
        return []

    trigger_choices = _skills_matching_triggers(user_prompt)

    catalog = "\n".join(
        f"- {name}: {info['description']}" for name, info in _SKILLS.items()
    )

    router_prompt = f"""You route user requests to the correct skill or multiple skills.

    Available skills:
    {catalog}

    Recent conversation:
    {recent_history}

    Retrieved Memory Context:
    {memory_context}

    Current user message for which you need to select skills ( most important ):
    {user_prompt}

    INSTRUCTIONS:
    1. Use the Recent conversation and Retrieved Memory to understand the implicit context of the user's message (e.g., if they say "send it", memory might reveal they mean an email).
    2. If the user's request requires actions from MULTIPLE skills (e.g., searching an email and then creating a calendar event), return a comma-separated list of those skill names (e.g., "email, calendar").
    3. If it only requires one skill, return just that skill name.
    4. Return "none" if no skill applies — conceptual questions, chit-chat, discussing a topic without asking for an action, or anything ambiguous. When in doubt, return "none".
    5. Return ONLY the list of skill names. No punctuation other than commas. No explanation."""

    response = await _router_llm.ainvoke([HumanMessage(content=router_prompt)])
    _LAST_ROUTER_METRICS = _get_token_metrics(response)

    # 🛠️ FIXED: Use the robust parser to strip formatting and thinking blocks
    content_str = extract_pure_text(response)


    raw_choices = str(content_str).strip().lower().replace(".", "").split(",")
    choices = [c.strip() for c in raw_choices]
    
    valid_choices = []
    for choice in trigger_choices + choices:
        if choice in _SKILLS and choice not in valid_choices:
            valid_choices.append(choice)
    return valid_choices


def load_skill(name: str) -> dict:
    """Return the full skill payload for the main graph to use.

    Returns {'markdown': <full skill.md text>, 'tools': [<tool_fn>, ...]}.
    The caller injects the markdown into the system prompt and binds the tools
    to the LLM via `llm.bind_tools(...)`.
    """
    skill = _SKILLS[name]
    with open(skill["skill_md_path"]) as f:
        markdown = f.read()
    return {"markdown": markdown, "tools": skill["tools"]}


def list_skills() -> dict:
    """Return the full registry. Handy for debugging and introspection."""
    return _SKILLS
