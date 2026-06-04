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
from pydantic import BaseModel
from dotenv import load_dotenv
from agent_connector import get_quarq_response
from agent import wipe_all_memories_for_api

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
EVENT_BUFFER_SIZE = 300


app = FastAPI(title="Quarq Agent", version="0.4.1")
EVENTS = deque(maxlen=EVENT_BUFFER_SIZE)
EVENT_LOCK = asyncio.Lock()
EVENT_SEQ = 0
JOBS: dict[str, dict] = {}
JOB_DONE_EVENTS: dict[str, asyncio.Event] = {}
JOB_QUEUE: asyncio.Queue[str] = asyncio.Queue()
JOB_LOCK = asyncio.Lock()
JOB_WORKER_TASK: asyncio.Task | None = None


class ChatRequest(BaseModel):
    prompt: str
    channel_type: str = "web"
    skip_learning: bool = False
    current_date: Optional[str] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def help_text() -> str:
    return "\n".join(
        [
            "Available commands:",
            "/help - show commands",
            "/status - show agent/API status",
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
        return None

    await record_event(
        "system",
        "Command handled",
        response,
        {"channel": channel_type, "command": command},
    )
    return {"response": response, "metrics": {}, "contexts": {}, "command": command}


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
    title = job_event_title(stage, data)
    event_key = (
        stage,
        message,
        data.get("tool_name"),
        data.get("tool_status"),
        tuple(data.get("skills") or []),
    )

    should_record = True
    async with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None or job.get("status") in {"completed", "failed"}:
            return

        if job.get("last_event_key") == event_key:
            should_record = False

        job["status"] = "running"
        job["stage"] = stage
        job["message"] = message
        job["tool_name"] = data.get("tool_name")
        job["updated_at"] = now_iso()
        job["last_event_key"] = event_key

    if should_record:
        await record_event(
            "job",
            title,
            message,
            {"job_id": job_id, "stage": stage, **data},
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
            "skip_learning": req.skip_learning,
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
        response, metrics, contexts = await get_quarq_response(
            user_prompt=req.prompt,
            user_id=AGENT_USER_ID,
            channel_type=req.channel_type,
            chat_history=[],
            skip_learning=req.skip_learning,
            current_date=req.current_date,
            status_callback=status_callback,
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
                "elapsed": round(elapsed, 2),
                "metrics": metrics,
                "contexts": context_line_counts(contexts),
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


async def process_telegram_update(update: dict):
    message = update.get("message") or {}
    text = str(message.get("text") or "").strip()
    chat = message.get("chat") or {}
    sender = message.get("from") or {}
    chat_id = chat.get("id")
    telegram_user_id = sender.get("id")
    username = sender.get("username") or sender.get("first_name") or "unknown"

    if not text or chat_id is None:
        await record_event(
            "telegram",
            "Telegram update ignored",
            "No text message was present in the update.",
            {"update_id": update.get("update_id")},
        )
        return

    await record_event(
        "telegram",
        "Telegram inbound",
        text,
        {
            "chat_id": chat_id,
            "telegram_user_id": telegram_user_id,
            "username": username,
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
        command_result = await handle_channel_command(text, "telegram")
        if command_result:
            await send_telegram_message(chat_id, command_result["response"])
            return

        job = await enqueue_chat_job(
            ChatRequest(prompt=text, channel_type="telegram", skip_learning=False)
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

    command_result = await handle_channel_command(req.prompt, req.channel_type)
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


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        command_result = await handle_channel_command(req.prompt, req.channel_type)
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
