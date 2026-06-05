# =====================================================
# Quarq Agent — Single-Tenant Worker
# =====================================================
# This container serves exactly one user. Identity is injected at
# `docker run` time via the USER_ID environment variable; the Node
# dispatcher forwards prompts to POST /api/chat.

import os
import logging
import asyncio
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agent_connector import get_quarq_response
from agent import wipe_all_memories_for_api
from agent_tools_config import handle_tool_command, load_enabled_cloud_tools
from local_channel_store import (
    append_attachment_note,
    append_chat_pair,
    decode_base64_payload,
    get_recent_history_items,
    list_chat_history_channels,
    recent_attachment_ids,
    refresh_attachments_for_context,
    render_attachment_context,
    store_attachment_from_bytes,
)
from tools.composio.client import clear_composio_session_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("quarq_agent")
for noisy_logger in ("httpx", "httpcore", "openai", "openai._base_client"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


# =====================================================
# CONFIG
# =====================================================
load_dotenv()

AGENT_USER_ID = os.getenv("USER_ID")
if not AGENT_USER_ID:
    raise RuntimeError("USER_ID environment variable is required")

TELEGRAM_WEBHOOK_PATH = "/api/telegram/webhook"
TELEGRAM_API_BASE = "https://api.telegram.org"
TELEGRAM_MESSAGE_LIMIT = 3900
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")
TELEGRAM_TYPING_INTERVAL_SECONDS = 4
DEFAULT_CHANNEL_FILE_MAX_BYTES = 20_000_000
CHANNEL_FILE_MAX_BYTES = int(os.getenv("CHANNEL_FILE_MAX_BYTES", str(DEFAULT_CHANNEL_FILE_MAX_BYTES)))
EVENT_BUFFER_SIZE = 300
CHAT_HISTORY_WINDOW_MESSAGES = 8


app = FastAPI(title="Quarq Agent", version="0.4.4")
EVENTS = deque(maxlen=EVENT_BUFFER_SIZE)
EVENT_LOCK = asyncio.Lock()
EVENT_SEQ = 0
JOBS: dict[str, dict] = {}
JOB_DONE_EVENTS: dict[str, asyncio.Event] = {}
JOB_QUEUE: asyncio.Queue[str] = asyncio.Queue()
JOB_LOCK = asyncio.Lock()
JOB_WORKER_TASK: asyncio.Task | None = None
CHAT_HISTORY_LOCK = asyncio.Lock()


class ChatRequest(BaseModel):
    prompt: str
    channel_type: str = "web"
    skip_learning: bool = False
    current_date: Optional[str] = None
    conversation_id: Optional[str] = None
    attachment_ids: list[str] = Field(default_factory=list)


class FileIngestRequest(BaseModel):
    data_base64: str
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    channel_type: str = "api"
    conversation_id: Optional[str] = None
    source_kind: str = "file"
    source_metadata: dict = Field(default_factory=dict)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def help_text() -> str:
    return "\n".join(
        [
            "Available commands:",
            "/help - show commands",
            "/status - show agent/API status",
            "/tools - list enabled native and cloud tools",
            "/which-tool <task> - show which tool fits a task",
            "/cloud-tools - list cloud tools available to enable",
            "/add-tool <tool> - enable a cloud tool",
            "/remove-tool <tool> - disable a cloud tool",
            "/wipe - clear local memories",
            "/quit - stop the local CLI only; remote channels cannot stop the process",
        ]
    )


def status_payload() -> dict:
    return {
        "status": "ok",
        "user_id": AGENT_USER_ID,
        "telegram_webhook_path": TELEGRAM_WEBHOOK_PATH,
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN),
        "telegram_allowed_users_configured": TELEGRAM_ALLOWED_USERS is not None,
        "job_queue_size": JOB_QUEUE.qsize(),
        "enabled_cloud_tools": load_enabled_cloud_tools(),
        "chat_history_channels": list_chat_history_channels(),
    }


def status_text() -> str:
    return "\n".join(f"{key}: {value}" for key, value in status_payload().items())


async def handle_channel_command(prompt: str, channel_type: str) -> Optional[dict]:
    command = prompt.strip().split(maxsplit=1)[0].lower()
    if "@" in command:
        command = command.split("@", 1)[0]

    if command in {"/start", "/help"}:
        response = help_text()
    elif command == "/status":
        response = status_text()
    elif command == "/wipe":
        await record_event("system", "Memory wipe started", f"Requested from {channel_type}.")
        await wipe_all_memories_for_api()
        await record_event("system", "Memory wipe complete", "Local memories were cleared.")
        response = "Local memories were cleared."
    elif command in {"/quit", "/exit"}:
        response = "This command only works in the local CLI. Remote channels cannot stop the local process."
    else:
        response = handle_tool_command(prompt)
        if response is None:
            return None
        if command in {"/add-tool", "/enable-tool", "/remove-tool", "/disable-tool"}:
            clear_composio_session_cache()

    await record_event(
        "system",
        "Command handled",
        response,
        {"channel": channel_type, "command": command},
    )
    return {"response": response, "metrics": {}, "contexts": {}, "command": command}


async def handle_and_store_channel_command(req: ChatRequest) -> Optional[dict]:
    command_result = await handle_channel_command(req.prompt, req.channel_type)
    if command_result:
        await append_chat_history(
            req.channel_type,
            req.prompt,
            command_result["response"],
            conversation_id=req.conversation_id,
            attachment_ids=req.attachment_ids,
        )
    return command_result


async def record_event(
    kind: str,
    title: str,
    message: str = "",
    data: Optional[dict] = None,
):
    global EVENT_SEQ

    async with EVENT_LOCK:
        EVENT_SEQ += 1
        event = {
            "id": EVENT_SEQ,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "title": title,
            "message": message,
            "data": data or {},
        }
        EVENTS.append(event)
        return event


def serialize_job(job: dict) -> dict:
    return {
        key: value
        for key, value in job.items()
        if key not in {"last_event_key"}
    }


def job_result_payload(job: dict) -> dict:
    result = job.get("result") or {}
    return {
        "response": result.get("response", ""),
        "metrics": result.get("metrics", {}),
        "contexts": result.get("contexts", {}),
    }


def job_event_title(stage: str, data: Optional[dict] = None) -> str:
    data = data or {}
    tool_status = data.get("tool_status")
    if stage == "retrieval":
        return "Retrieving memory"
    if stage == "tool_routing":
        return "Routing tools"
    if stage == "generation":
        return "Generating response"
    if stage == "tool" and tool_status == "completed":
        return "Tool completed"
    if stage == "tool" and tool_status == "failed":
        return "Tool failed"
    if stage == "tool":
        return "Tool is being used"
    if stage == "finalizing":
        return "Finalizing response"
    return "Job status"


def public_tool_name(tool_name: str | None) -> str | None:
    if not tool_name:
        return None
    if str(tool_name).upper().startswith("COMPOSIO_"):
        return "cloud tools"
    if str(tool_name) == "configure_cloud_tools":
        return "cloud tools"
    return str(tool_name)


def public_skill_names(skills: list | None) -> list:
    public_names = []
    for skill in skills or []:
        if str(skill) == "composio":
            public_names.append("cloud tools")
        else:
            public_names.append(skill)
    return public_names


async def get_recent_chat_history(
    channel_type: str,
    conversation_id: str | None = None,
) -> list[BaseMessage]:
    async with CHAT_HISTORY_LOCK:
        items = get_recent_history_items(
            channel_type,
            conversation_id,
            limit=CHAT_HISTORY_WINDOW_MESSAGES,
        )

    messages: list[BaseMessage] = []
    for item in items:
        content = append_attachment_note(
            str(item.get("content") or ""),
            item.get("attachment_ids") or [],
        )
        if item.get("role") == "ai":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


async def append_chat_history(
    channel_type: str,
    user_prompt: str,
    agent_response: str,
    conversation_id: str | None = None,
    attachment_ids: list[str] | None = None,
) -> None:
    async with CHAT_HISTORY_LOCK:
        append_chat_pair(
            channel_type,
            user_prompt,
            agent_response,
            conversation_id=conversation_id,
            attachment_ids=attachment_ids or [],
        )


async def get_job_snapshot(job_id: str) -> dict:
    async with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return serialize_job(job.copy())


async def update_job_progress(
    job_id: str,
    stage: str,
    message: str = "",
    data: Optional[dict] = None,
) -> None:
    data = data or {}
    raw_tool_name = data.get("tool_name")
    display_tool_name = public_tool_name(raw_tool_name)
    title = job_event_title(stage, data)
    event_key = (
        stage,
        message,
        raw_tool_name,
        data.get("tool_status"),
        tuple(data.get("skills") or []),
    )
    display_message = (
        message.replace(str(raw_tool_name), display_tool_name or str(raw_tool_name))
        if raw_tool_name and display_tool_name
        else message
    )
    event_data = {**data}
    if raw_tool_name:
        event_data["tool_name"] = display_tool_name
    if data.get("skills"):
        event_data["skills"] = public_skill_names(data.get("skills"))

    should_record = True
    async with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None or job.get("status") in {"completed", "failed"}:
            return

        if job.get("last_event_key") == event_key:
            should_record = False

        job["status"] = "running"
        job["stage"] = stage
        job["message"] = display_message
        job["tool_name"] = display_tool_name
        job["updated_at"] = now_iso()
        job["last_event_key"] = event_key

    if should_record:
        await record_event(
            "job",
            title,
            display_message,
            {"job_id": job_id, "stage": stage, **event_data},
        )


async def create_completed_job(req: ChatRequest, result: dict) -> dict:
    job_id = str(uuid.uuid4())
    timestamp = now_iso()
    job = {
        "id": job_id,
        "type": "chat",
        "status": "completed",
        "stage": "completed",
        "message": "Command completed.",
        "tool_name": None,
        "request": req.model_dump(),
        "result": {
            "response": result.get("response", ""),
            "metrics": result.get("metrics", {}),
            "contexts": result.get("contexts", {}),
        },
        "error": None,
        "created_at": timestamp,
        "updated_at": timestamp,
        "started_at": timestamp,
        "completed_at": timestamp,
    }
    done_event = asyncio.Event()
    done_event.set()
    async with JOB_LOCK:
        JOBS[job_id] = job
        JOB_DONE_EVENTS[job_id] = done_event
    return serialize_job(job)


async def enqueue_chat_job(req: ChatRequest) -> dict:
    job_id = str(uuid.uuid4())
    timestamp = now_iso()
    job = {
        "id": job_id,
        "type": "chat",
        "status": "queued",
        "stage": "queued",
        "message": "Waiting for the agent worker.",
        "tool_name": None,
        "request": req.model_dump(),
        "result": None,
        "error": None,
        "created_at": timestamp,
        "updated_at": timestamp,
        "started_at": None,
        "completed_at": None,
    }
    async with JOB_LOCK:
        JOBS[job_id] = job
        JOB_DONE_EVENTS[job_id] = asyncio.Event()

    await JOB_QUEUE.put(job_id)
    await record_event(
        "request",
        "Chat request",
        req.prompt,
        {
            "job_id": job_id,
            "channel": req.channel_type,
            "conversation_id": req.conversation_id,
            "skip_learning": req.skip_learning,
            "attachment_count": len(req.attachment_ids),
        },
    )
    return serialize_job(job)


async def complete_job(job_id: str, result: dict) -> None:
    async with JOB_LOCK:
        job = JOBS[job_id]
        job["status"] = "completed"
        job["stage"] = "completed"
        job["message"] = "Response ready."
        job["tool_name"] = None
        job["result"] = result
        job["updated_at"] = now_iso()
        job["completed_at"] = job["updated_at"]
        done_event = JOB_DONE_EVENTS.get(job_id)
        if done_event:
            done_event.set()


async def fail_job(job_id: str, error: str) -> None:
    async with JOB_LOCK:
        job = JOBS[job_id]
        job["status"] = "failed"
        job["stage"] = "failed"
        job["message"] = "The job failed."
        job["tool_name"] = None
        job["error"] = error
        job["updated_at"] = now_iso()
        job["completed_at"] = job["updated_at"]
        done_event = JOB_DONE_EVENTS.get(job_id)
        if done_event:
            done_event.set()


async def wait_for_job(job_id: str, timeout: Optional[float] = None) -> dict:
    async with JOB_LOCK:
        done_event = JOB_DONE_EVENTS.get(job_id)
    if done_event is None:
        raise KeyError(job_id)

    await asyncio.wait_for(done_event.wait(), timeout=timeout)
    return await get_job_snapshot(job_id)


async def run_chat_job(job_id: str) -> None:
    async with JOB_LOCK:
        job = JOBS[job_id]
        job["status"] = "running"
        job["stage"] = "starting"
        job["message"] = "Starting agent request."
        job["started_at"] = now_iso()
        job["updated_at"] = job["started_at"]
        request_data = dict(job["request"])

    req = ChatRequest(**request_data)
    started = time.perf_counter()

    async def status_callback(
        stage: str,
        message: str = "",
        data: Optional[dict] = None,
    ) -> None:
        await update_job_progress(job_id, stage, message, data)

    try:
        await update_job_progress(job_id, "retrieval", "Searching memory.")
        chat_history = await get_recent_chat_history(req.channel_type, req.conversation_id)
        context_attachment_ids = []
        for attachment_id in recent_attachment_ids(req.channel_type, req.conversation_id):
            if attachment_id not in context_attachment_ids:
                context_attachment_ids.append(attachment_id)
        for attachment_id in req.attachment_ids:
            if attachment_id not in context_attachment_ids:
                context_attachment_ids.append(attachment_id)
        await refresh_attachments_for_context(context_attachment_ids)
        attachment_context = render_attachment_context(context_attachment_ids)
        response, metrics, contexts = await get_quarq_response(
            user_prompt=req.prompt,
            user_id=AGENT_USER_ID,
            channel_type=req.channel_type,
            chat_history=chat_history,
            skip_learning=req.skip_learning,
            current_date=req.current_date,
            status_callback=status_callback,
            attachments_context=attachment_context,
        )
        await append_chat_history(
            req.channel_type,
            req.prompt,
            response,
            conversation_id=req.conversation_id,
            attachment_ids=req.attachment_ids,
        )
        elapsed = time.perf_counter() - started
        result = {"response": response, "metrics": metrics, "contexts": contexts}
        await complete_job(job_id, result)
        await record_event(
            "response",
            "Telegram response" if req.channel_type == "telegram" else "Chat response",
            response or "",
            {
                "job_id": job_id,
                "channel": req.channel_type,
                "conversation_id": req.conversation_id,
                "elapsed": round(elapsed, 2),
                "metrics": metrics,
                "contexts": context_line_counts(contexts),
                "attachment_count": len(req.attachment_ids),
            },
        )
    except Exception as e:
        logger.error("Agent job error for %s: %s", AGENT_USER_ID, e, exc_info=True)
        await fail_job(job_id, str(e))
        await record_event("error", "Chat error", str(e), {"job_id": job_id, "channel": req.channel_type})


async def job_worker() -> None:
    while True:
        job_id = await JOB_QUEUE.get()
        try:
            await run_chat_job(job_id)
        finally:
            JOB_QUEUE.task_done()


def context_line_counts(contexts: dict) -> dict:
    return {
        key: len(str(value or "").splitlines())
        for key, value in (contexts or {}).items()
    }


def parse_allowed_telegram_users() -> set[int] | None:
    raw_value = os.getenv("TELEGRAM_ALLOWED_USERS")
    if not raw_value:
        return None

    allowed = set()
    for item in raw_value.split(","):
        clean_item = item.strip()
        if clean_item:
            try:
                allowed.add(int(clean_item))
            except ValueError:
                logger.warning("Ignoring invalid TELEGRAM_ALLOWED_USERS value: %s", clean_item)
    return allowed


TELEGRAM_ALLOWED_USERS = parse_allowed_telegram_users()


@app.on_event("startup")
async def start_background_job_worker():
    global JOB_WORKER_TASK
    if JOB_WORKER_TASK is None or JOB_WORKER_TASK.done():
        JOB_WORKER_TASK = asyncio.create_task(job_worker())


@app.on_event("shutdown")
async def stop_background_job_worker():
    if JOB_WORKER_TASK is not None:
        JOB_WORKER_TASK.cancel()
        try:
            await JOB_WORKER_TASK
        except asyncio.CancelledError:
            pass


def split_telegram_message(text: str) -> list[str]:
    if len(text) <= TELEGRAM_MESSAGE_LIMIT:
        return [text]

    chunks = []
    remaining = text
    while remaining:
        chunk = remaining[:TELEGRAM_MESSAGE_LIMIT]
        split_at = chunk.rfind("\n")
        if split_at < TELEGRAM_MESSAGE_LIMIT // 2:
            split_at = chunk.rfind(" ")
        if split_at < TELEGRAM_MESSAGE_LIMIT // 2:
            split_at = TELEGRAM_MESSAGE_LIMIT
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    return [chunk for chunk in chunks if chunk]


async def telegram_api_call(method: str, payload: dict):
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    url = f"{TELEGRAM_API_BASE}/bot{TELEGRAM_BOT_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error: {data}")
        return data


async def send_telegram_message(chat_id: int, text: str):
    for chunk in split_telegram_message(text or "No response was returned."):
        await telegram_api_call("sendMessage", {"chat_id": chat_id, "text": chunk})


async def send_telegram_chat_action(chat_id: int, action: str = "typing"):
    await telegram_api_call("sendChatAction", {"chat_id": chat_id, "action": action})


async def keep_telegram_typing(chat_id: int):
    while True:
        try:
            await send_telegram_chat_action(chat_id)
        except Exception as e:
            logger.debug("Telegram typing action failed: %s", e)
        await asyncio.sleep(TELEGRAM_TYPING_INTERVAL_SECONDS)


def telegram_message_from_update(update: dict) -> tuple[dict, str]:
    for key in ("message", "edited_message", "channel_post", "edited_channel_post"):
        if update.get(key):
            return update[key] or {}, key
    return {}, "unknown"


def format_file_size(size_bytes: int | None) -> str:
    if not size_bytes:
        return "unknown size"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    if size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.1f} KB"
    return f"{size_bytes} bytes"


def telegram_download_limit_message(size_bytes: int | None = None) -> str:
    limit = format_file_size(CHANNEL_FILE_MAX_BYTES)
    if size_bytes:
        return (
            f"That file is {format_file_size(size_bytes)}, which is above this agent's "
            f"Telegram download limit of {limit}. Please send a smaller file or share "
            "the important text directly."
        )
    return (
        f"That file is above this agent's Telegram download limit of {limit}. "
        "Please send a smaller file or share the important text directly."
    )


def is_telegram_size_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "file is too big",
            "file is too large",
            "request entity too large",
            "above channel_file_max_bytes",
            "telegram download limit",
        )
    )


def public_attachment_error(error: Exception | str) -> str:
    text = str(error)
    if "telegram download limit" in text.lower():
        return text
    if isinstance(error, Exception) and is_telegram_size_limit_error(error):
        return telegram_download_limit_message()
    return "I could not download or read it locally. Please try again with a smaller/common file."


def format_attachment_failure_message(failures: list[dict]) -> str:
    if not failures:
        return ""

    lines = ["I could not process these attachment(s):"]
    for failure in failures[:5]:
        label = failure.get("filename") or failure.get("kind") or "attachment"
        lines.append(f"- {label}: {failure.get('message')}")
    if len(failures) > 5:
        lines.append(f"- {len(failures) - 5} more attachment(s) also failed.")
    return "\n".join(lines)


def telegram_file_references(message: dict) -> list[dict]:
    refs = []

    if message.get("photo"):
        photos = message.get("photo") or []
        best_photo = max(
            photos,
            key=lambda item: item.get("file_size") or (item.get("width", 0) * item.get("height", 0)),
        )
        refs.append(
            {
                "kind": "photo",
                "file_id": best_photo.get("file_id"),
                "filename": f"telegram_photo_{best_photo.get('file_unique_id') or best_photo.get('file_id')}.jpg",
                "mime_type": "image/jpeg",
                "metadata": {
                    "width": best_photo.get("width"),
                    "height": best_photo.get("height"),
                    "file_size": best_photo.get("file_size"),
                },
            }
        )

    field_map = {
        "document": "document",
        "audio": "audio",
        "voice": "voice",
        "video": "video",
        "video_note": "video_note",
        "animation": "animation",
        "sticker": "sticker",
    }
    for field, kind in field_map.items():
        value = message.get(field)
        if not value:
            continue
        refs.append(
            {
                "kind": kind,
                "file_id": value.get("file_id"),
                "filename": value.get("file_name") or f"telegram_{kind}_{value.get('file_unique_id') or value.get('file_id')}",
                "mime_type": value.get("mime_type"),
                "metadata": {
                    key: value.get(key)
                    for key in (
                        "file_size",
                        "duration",
                        "width",
                        "height",
                        "emoji",
                        "set_name",
                    )
                    if value.get(key) is not None
                },
            }
        )

    return [ref for ref in refs if ref.get("file_id")]


async def download_telegram_file(file_id: str) -> tuple[bytes, dict]:
    try:
        file_info = await telegram_api_call("getFile", {"file_id": file_id})
    except Exception as exc:
        if is_telegram_size_limit_error(exc):
            raise RuntimeError(telegram_download_limit_message()) from exc
        raise

    result = file_info.get("result") or {}
    file_size = int(result.get("file_size") or 0)
    if file_size and file_size > CHANNEL_FILE_MAX_BYTES:
        raise RuntimeError(telegram_download_limit_message(file_size))

    file_path = result.get("file_path")
    if not file_path:
        raise RuntimeError("Telegram did not return a downloadable file path.")

    url = f"{TELEGRAM_API_BASE}/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.content

    if len(content) > CHANNEL_FILE_MAX_BYTES:
        raise RuntimeError(telegram_download_limit_message(len(content)))
    return content, result


async def store_telegram_attachments(
    refs: list[dict],
    chat_id: int,
    telegram_user_id: int | None,
    username: str,
    update_id: int | None,
) -> tuple[list[dict], list[dict]]:
    records = []
    failures = []
    for ref in refs:
        try:
            content, file_info = await download_telegram_file(ref["file_id"])
            metadata = {
                "telegram_user_id": telegram_user_id,
                "username": username,
                "chat_id": chat_id,
                "update_id": update_id,
                "telegram_file": file_info,
                **(ref.get("metadata") or {}),
            }
            record = await store_attachment_from_bytes(
                content,
                filename=ref.get("filename"),
                mime_type=ref.get("mime_type"),
                channel_type="telegram",
                conversation_id=str(chat_id),
                source_kind=ref.get("kind") or "telegram_file",
                source_metadata=metadata,
            )
            records.append(record)
            await record_event(
                "attachment",
                "Attachment stored",
                f"{record.get('original_filename') or record.get('stored_filename')} saved.",
                {
                    "channel": "telegram",
                    "conversation_id": str(chat_id),
                    "attachment_id": record.get("id"),
                    "mime_type": record.get("mime_type"),
                    "size_bytes": record.get("size_bytes"),
                },
            )
        except Exception as exc:
            await record_event(
                "error",
                "Attachment storage failed",
                str(exc),
                {"channel": "telegram", "file_id": ref.get("file_id"), "kind": ref.get("kind")},
            )
            failures.append(
                {
                    "kind": ref.get("kind"),
                    "filename": ref.get("filename"),
                    "message": public_attachment_error(exc),
                }
            )
    return records, failures


def prompt_for_telegram_message(text: str, attachment_records: list[dict]) -> str:
    if text:
        return text
    labels = [
        str(record.get("original_filename") or record.get("stored_filename") or record.get("id"))
        for record in attachment_records
    ]
    if labels:
        return "I sent these attachment(s). Please read them and respond: " + ", ".join(labels)
    return ""


async def process_telegram_update(update: dict):
    message, update_kind = telegram_message_from_update(update)
    text = str(message.get("text") or message.get("caption") or "").strip()
    file_refs = telegram_file_references(message)
    chat = message.get("chat") or {}
    sender = message.get("from") or {}
    chat_id = chat.get("id")
    telegram_user_id = sender.get("id")
    username = sender.get("username") or sender.get("first_name") or "unknown"

    if (not text and not file_refs) or chat_id is None:
        await record_event(
            "telegram",
            "Telegram update ignored",
            "No supported message content was present in the update.",
            {"update_id": update.get("update_id")},
        )
        return

    await record_event(
        "telegram",
        "Telegram edit inbound" if update_kind.startswith("edited_") else "Telegram inbound",
        text or f"{len(file_refs)} attachment(s)",
        {
            "update_kind": update_kind,
            "chat_id": chat_id,
            "telegram_user_id": telegram_user_id,
            "username": username,
            "attachment_count": len(file_refs),
        },
    )

    if (
        TELEGRAM_ALLOWED_USERS is not None
        and int(telegram_user_id or 0) not in TELEGRAM_ALLOWED_USERS
    ):
        await record_event(
            "warning",
            "Telegram blocked",
            f"User {username} is not in TELEGRAM_ALLOWED_USERS.",
            {"telegram_user_id": telegram_user_id, "username": username},
        )
        try:
            await send_telegram_message(
                chat_id,
                "This local agent is not configured for your Telegram account.",
            )
        except Exception as e:
            logger.error("Telegram blocked-user reply failed: %s", e, exc_info=True)
        return

    try:
        attachment_records, attachment_failures = await store_telegram_attachments(
            file_refs,
            chat_id=chat_id,
            telegram_user_id=telegram_user_id,
            username=username,
            update_id=update.get("update_id"),
        )
        if attachment_failures and not attachment_records:
            failure_message = format_attachment_failure_message(attachment_failures)
            await send_telegram_message(
                chat_id,
                failure_message,
            )
            await append_chat_history(
                "telegram",
                text or "I sent attachment(s), but they could not be processed.",
                failure_message,
                conversation_id=str(chat_id),
            )
            return
        if attachment_failures:
            await send_telegram_message(chat_id, format_attachment_failure_message(attachment_failures))

        command_result = None
        if text and not attachment_records:
            command_req = ChatRequest(
                prompt=text,
                channel_type="telegram",
                conversation_id=str(chat_id),
                skip_learning=False,
            )
            command_result = await handle_and_store_channel_command(command_req)
        if command_result:
            await send_telegram_message(chat_id, command_result["response"])
            return

        prompt = prompt_for_telegram_message(text, attachment_records)
        attachment_ids = [record["id"] for record in attachment_records]
        job = await enqueue_chat_job(
            ChatRequest(
                prompt=prompt,
                channel_type="telegram",
                conversation_id=str(chat_id),
                attachment_ids=attachment_ids,
                skip_learning=False,
            )
        )
        typing_task = asyncio.create_task(keep_telegram_typing(chat_id))
        try:
            completed_job = await wait_for_job(job["id"])
        finally:
            typing_task.cancel()
            await asyncio.gather(typing_task, return_exceptions=True)
        if completed_job["status"] == "failed":
            raise RuntimeError(completed_job.get("error") or "Telegram job failed")
        response = job_result_payload(completed_job)["response"]
        await send_telegram_message(chat_id, response)
    except Exception as e:
        logger.error("Telegram processing error: %s", e, exc_info=True)
        await record_event("error", "Telegram processing error", str(e))
        try:
            await send_telegram_message(
                chat_id,
                "The local agent hit an error while processing that message.",
            )
        except Exception as send_error:
            logger.error("Telegram error reply failed: %s", send_error, exc_info=True)


@app.get("/")
async def health():
    return status_payload()


@app.get("/api/events")
async def get_events(after: int = 0):
    async with EVENT_LOCK:
        events = [event for event in EVENTS if event["id"] > after]
    return {"events": events}


@app.post("/api/jobs", status_code=202)
async def create_job(req: ChatRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    command_result = await handle_and_store_channel_command(req)
    if command_result:
        job = await create_completed_job(req, command_result)
        return {"job": job}

    job = await enqueue_chat_job(req)
    return {"job": job}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        job = await get_job_snapshot(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found") from None
    return {"job": job}


@app.post("/api/files", status_code=201)
async def ingest_file(req: FileIngestRequest):
    try:
        content = decode_base64_payload(req.data_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="data_base64 must be valid base64") from None

    if len(content) > CHANNEL_FILE_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"file is {format_file_size(len(content))}, above the configured "
                f"channel limit of {format_file_size(CHANNEL_FILE_MAX_BYTES)}"
            ),
        )

    try:
        record = await store_attachment_from_bytes(
            content,
            filename=req.filename,
            mime_type=req.mime_type,
            channel_type=req.channel_type,
            conversation_id=req.conversation_id,
            source_kind=req.source_kind,
            source_metadata=req.source_metadata,
        )
        await record_event(
            "attachment",
            "Attachment stored",
            f"{record.get('original_filename') or record.get('stored_filename')} saved.",
            {
                "channel": req.channel_type,
                "conversation_id": req.conversation_id,
                "attachment_id": record.get("id"),
                "mime_type": record.get("mime_type"),
                "size_bytes": record.get("size_bytes"),
            },
        )
        return {"attachment": record}
    except Exception as e:
        logger.error("File ingest error: %s", e, exc_info=True)
        await record_event("error", "Attachment storage failed", str(e), {"channel": req.channel_type})
        raise HTTPException(status_code=500, detail="file ingest failed") from e


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        command_result = await handle_and_store_channel_command(req)
        if command_result:
            return command_result

        job = await enqueue_chat_job(req)
        completed_job = await wait_for_job(job["id"])
        if completed_job["status"] == "failed":
            raise RuntimeError(completed_job.get("error") or "agent job failed")
        return job_result_payload(completed_job)
    except Exception as e:
        logger.error(f"Agent error for {AGENT_USER_ID}: {e}", exc_info=True)
        await record_event("error", "Chat error", str(e), {"channel": req.channel_type})
        raise HTTPException(status_code=500, detail="agent processing failed")


@app.post("/api/memories/wipe")
async def wipe_memories():
    try:
        await record_event("system", "Memory wipe started", "Requested from API.")
        await wipe_all_memories_for_api()
        await record_event("system", "Memory wipe complete", "Local memories were cleared.")
        return {"status": "ok", "user_id": AGENT_USER_ID}
    except Exception as e:
        logger.error(f"Memory wipe error for {AGENT_USER_ID}: {e}", exc_info=True)
        await record_event("error", "Memory wipe error", str(e))
        raise HTTPException(status_code=500, detail="memory wipe failed")


@app.post(TELEGRAM_WEBHOOK_PATH)
async def telegram_webhook(
    update: dict,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
):
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=503, detail="telegram is not configured")

    if (
        TELEGRAM_WEBHOOK_SECRET
        and x_telegram_bot_api_secret_token != TELEGRAM_WEBHOOK_SECRET
    ):
        raise HTTPException(status_code=403, detail="invalid telegram webhook secret")

    await record_event(
        "telegram",
        "Telegram webhook accepted",
        f"update_id={update.get('update_id')}",
    )
    background_tasks.add_task(process_telegram_update, update)
    return {"status": "accepted"}
