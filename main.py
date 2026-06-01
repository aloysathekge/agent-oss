# =====================================================
# Quarq Agent — Single-Tenant Worker
# =====================================================
# This container serves exactly one user. Identity is injected at
# `docker run` time via the USER_ID environment variable; the Node
# dispatcher forwards prompts to POST /api/chat.

import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
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


# =====================================================
# CONFIG
# =====================================================
load_dotenv()

AGENT_USER_ID = os.getenv("USER_ID")
if not AGENT_USER_ID:
    raise RuntimeError("USER_ID environment variable is required")


app = FastAPI(title="Quarq Agent", version="0.4.0")


class ChatRequest(BaseModel):
    prompt: str
    channel_type: str = "web"
    skip_learning: bool = False
    current_date: Optional[str] = None


@app.get("/")
async def health():
    return {"status": "ok", "user_id": AGENT_USER_ID}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        response, metrics, contexts = await get_quarq_response(
            user_prompt=req.prompt,
            user_id=AGENT_USER_ID,
            channel_type=req.channel_type,
            chat_history=[],
            skip_learning=req.skip_learning,
            current_date=req.current_date,
        )
        return {"response": response, "metrics": metrics, "contexts": contexts}
    except Exception as e:
        logger.error(f"Agent error for {AGENT_USER_ID}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="agent processing failed")


@app.post("/api/memories/wipe")
async def wipe_memories():
    try:
        await wipe_all_memories_for_api()
        return {"status": "ok", "user_id": AGENT_USER_ID}
    except Exception as e:
        logger.error(f"Memory wipe error for {AGENT_USER_ID}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="memory wipe failed")
