"""Session-backed cloud tools for external app actions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool

from agent_tools_config import load_enabled_cloud_tools


DEFAULT_COMPOSIO_TOOLKITS = (
    "github",
    "gmail",
    "googlecalendar",
    "slack",
    "notion",
    "linear",
)

BASE_DIR = Path(__file__).resolve().parents[2]
_SESSION_CACHE: dict[tuple[str, tuple[str, ...]], Any] = {}


@tool
def configure_cloud_tools() -> str:
    """Explain how to configure cloud tools when external app actions are unavailable."""

    return (
        "Cloud tools are not configured for this local agent. Set CLOUD_TOOLS_API_KEY "
        "in .env, optionally set CLOUD_TOOLKITS, install the cloud tool "
        "dependencies, and restart the agent."
    )


def parse_toolkits(value: str | None = None) -> list[str]:
    """Parse configured cloud-tool slugs with stable defaults."""

    if value is None:
        enabled_tools = load_enabled_cloud_tools()
        if enabled_tools:
            return enabled_tools

    raw_value = value if value is not None else os.getenv("CLOUD_TOOLKITS") or os.getenv("COMPOSIO_TOOLKITS", "")
    items = [item.strip().lower() for item in str(raw_value or "").split(",")]
    toolkits = []
    seen = set()

    for item in items:
        if item and item not in seen:
            toolkits.append(item)
            seen.add(item)

    return toolkits or list(DEFAULT_COMPOSIO_TOOLKITS)


def clear_composio_session_cache() -> None:
    """Clear cached sessions. Used by tests."""

    _SESSION_CACHE.clear()


def _runtime_user_id(runtime_config: dict | None) -> str:
    runtime_config = runtime_config or {}
    return (
        str(runtime_config.get("user_id") or "").strip()
        or os.getenv("USER_ID", "").strip()
        or "local_user"
    )


def _ensure_composio_cache_dir() -> None:
    cache_override = os.getenv("CLOUD_TOOLS_CACHE_DIR") or os.getenv("COMPOSIO_CACHE_DIR")
    if cache_override and cache_override.strip():
        os.environ.setdefault("COMPOSIO_CACHE_DIR", cache_override.strip())
        return

    memory_root = Path(os.getenv("LOCAL_MEMORY_ROOT", "local_memory")).expanduser()
    if not memory_root.is_absolute():
        memory_root = BASE_DIR / memory_root

    cache_dir = memory_root / "cloud_tools_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["COMPOSIO_CACHE_DIR"] = str(cache_dir)


def _create_session(composio: Any, user_id: str, toolkits: list[str]) -> Any:
    """Create a cloud-tool session across supported SDK shapes."""

    sessions = getattr(composio, "sessions", None)
    if sessions is not None and hasattr(sessions, "create"):
        return sessions.create(user_id=user_id, toolkits=toolkits)

    if hasattr(composio, "create"):
        return composio.create(user_id=user_id, toolkits=toolkits)

    if hasattr(composio, "session"):
        return composio.session(user_id=user_id, toolkits=toolkits)

    raise RuntimeError("The installed cloud-tool SDK does not expose a session creator.")


def _fallback_tool_with_reason(reason: str):
    _ = reason
    return configure_cloud_tools


def get_composio_tools(runtime_config: dict | None = None) -> list[Any]:
    """Return LangChain tools from a user-scoped cloud-tool session."""

    load_dotenv()
    api_key = (os.getenv("CLOUD_TOOLS_API_KEY") or os.getenv("COMPOSIO_API_KEY", "")).strip()
    if not api_key:
        return [_fallback_tool_with_reason("CLOUD_TOOLS_API_KEY is missing.")]

    _ensure_composio_cache_dir()

    try:
        from composio import Composio
        from composio_langchain import LangchainProvider
    except Exception as exc:
        return [_fallback_tool_with_reason(f"Cloud tool SDK could not be loaded: {exc}.")]

    user_id = _runtime_user_id(runtime_config)
    toolkits = parse_toolkits()
    cache_key = (user_id, tuple(toolkits))

    try:
        if cache_key not in _SESSION_CACHE:
            composio = Composio(api_key=api_key, provider=LangchainProvider())
            _SESSION_CACHE[cache_key] = _create_session(composio, user_id, toolkits)

        session = _SESSION_CACHE[cache_key]
        if not hasattr(session, "tools"):
            return [_fallback_tool_with_reason("Cloud tool session did not expose session.tools().")]

        tools = session.tools()
    except Exception as exc:
        return [_fallback_tool_with_reason(f"Cloud tool session setup failed: {exc}.")]

    return list(tools or []) or [_fallback_tool_with_reason("Cloud tool session returned no tools.")]
