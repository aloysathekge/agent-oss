"""LangChain @tool functions for the Agent Identity Manager skill.

The active agent identity is stored in a local JSON config file. Environment
values are only startup defaults when the local file or a field is missing.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agent_config import get_agent_identity_config_path, save_agent_config


def _clean_text(value: str | None) -> str:
    return str(value or "").strip()


def _parse_use_cases(agent_use_cases_json: str) -> list[str] | str:
    try:
        parsed: Any = json.loads(agent_use_cases_json)
    except json.JSONDecodeError:
        return "Error: agent_use_cases_json is not valid JSON."

    if not isinstance(parsed, list) or not all(
        isinstance(item, str) and item.strip() for item in parsed
    ):
        return "Error: agent_use_cases_json must be a valid JSON array of non-empty strings."

    seen = set()
    use_cases = []
    for item in parsed:
        cleaned = " ".join(item.split())
        if cleaned not in seen:
            seen.add(cleaned)
            use_cases.append(cleaned)

    return use_cases


@tool
def update_agent_identity(
    agent_name: str = None,
    agent_personality: str = None,
    agent_use_cases_json: str = None,
    agent_custom_prompt: str = None,
    config: RunnableConfig = None,
) -> str:
    """Updates the core local identity of the agent."""

    updates = {}

    name = _clean_text(agent_name)
    if name:
        updates["agent_name"] = name

    personality = _clean_text(agent_personality)
    if personality:
        updates["agent_personality"] = personality

    if agent_custom_prompt is not None:
        updates["agent_custom_prompt"] = _clean_text(agent_custom_prompt)

    use_cases_text = _clean_text(agent_use_cases_json)
    if use_cases_text:
        use_cases = _parse_use_cases(use_cases_text)
        if isinstance(use_cases, str):
            return use_cases
        updates["agent_use_cases"] = use_cases

    if not updates:
        return "No valid updates provided. Nothing was changed in the local identity config."

    try:
        saved_config = save_agent_config(updates)
    except OSError as exc:
        return f"Local Config Error: Failed to update agent identity. {exc}"

    success_msg = "Successfully updated local identity config:\n"
    for key in updates:
        success_msg += f"- {key}: {saved_config.get(key)}\n"
    success_msg += f"\nConfig file: {get_agent_identity_config_path()}"
    return success_msg
