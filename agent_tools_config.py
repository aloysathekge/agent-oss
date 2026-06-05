"""Local cloud-tool configuration helpers.

The public UI calls these "cloud tools". The implementation can be backed by
any provider, but commands and user-facing text should stay provider-neutral.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CLOUD_TOOL_SLUGS = (
    "github",
    "gmail",
    "googlecalendar",
    "slack",
    "notion",
    "linear",
)

CLOUD_TOOL_CATALOG: dict[str, dict[str, Any]] = {
    "github": {
        "label": "GitHub",
        "description": "Read repositories, issues, pull requests, and create or update GitHub records.",
        "aliases": ("repo", "repository", "pull request", "pr", "issue", "github"),
    },
    "gmail": {
        "label": "Gmail",
        "description": "Read, search, draft, and send email after account connection.",
        "aliases": ("email", "gmail", "inbox", "unread", "mail"),
    },
    "googlecalendar": {
        "label": "Google Calendar",
        "description": "Read calendars, check availability, and create or update events.",
        "aliases": ("calendar", "meeting", "schedule", "event", "google calendar"),
    },
    "slack": {
        "label": "Slack",
        "description": "Search channels, read messages, and post Slack messages.",
        "aliases": ("slack", "channel", "workspace"),
    },
    "notion": {
        "label": "Notion",
        "description": "Search, read, create, and update Notion pages and databases.",
        "aliases": ("notion", "page", "database", "doc"),
    },
    "linear": {
        "label": "Linear",
        "description": "Search, create, and update Linear issues and projects.",
        "aliases": ("linear", "ticket", "bug", "issue", "project"),
    },
    "googledrive": {
        "label": "Google Drive",
        "description": "Search, read, and manage files in Google Drive.",
        "aliases": ("drive", "google drive", "file", "folder"),
    },
    "googlesheets": {
        "label": "Google Sheets",
        "description": "Read and update spreadsheets.",
        "aliases": ("sheet", "spreadsheet", "google sheets"),
    },
    "googledocs": {
        "label": "Google Docs",
        "description": "Read and update documents.",
        "aliases": ("docs", "document", "google docs"),
    },
    "jira": {
        "label": "Jira",
        "description": "Search, create, and update Jira issues.",
        "aliases": ("jira", "ticket", "sprint", "issue"),
    },
    "trello": {
        "label": "Trello",
        "description": "Read and update Trello boards, lists, and cards.",
        "aliases": ("trello", "board", "card", "list"),
    },
    "asana": {
        "label": "Asana",
        "description": "Read and update Asana tasks and projects.",
        "aliases": ("asana", "task", "project"),
    },
    "hubspot": {
        "label": "HubSpot",
        "description": "Read and update CRM contacts, companies, and deals.",
        "aliases": ("hubspot", "crm", "contact", "deal"),
    },
    "salesforce": {
        "label": "Salesforce",
        "description": "Read and update CRM records.",
        "aliases": ("salesforce", "crm", "lead", "opportunity"),
    },
    "discord": {
        "label": "Discord",
        "description": "Read and send Discord messages.",
        "aliases": ("discord", "server", "channel"),
    },
    "microsoftteams": {
        "label": "Microsoft Teams",
        "description": "Read and send Teams messages.",
        "aliases": ("teams", "microsoft teams", "chat"),
    },
}


def _safe_agent_id(value: str | None) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value or "local_agent")


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def get_cloud_tools_config_path() -> Path:
    explicit_path = os.getenv("CLOUD_TOOLS_CONFIG_PATH")
    if explicit_path:
        return _resolve_path(explicit_path)

    memory_root = _resolve_path(os.getenv("LOCAL_MEMORY_ROOT", "local_memory"))
    return memory_root / _safe_agent_id(os.getenv("AGENT_ID")) / "agent_tools.json"


def normalize_tool_slug(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if not text:
        return ""

    compact = re.sub(r"[^a-z0-9]", "", text)
    for slug, info in CLOUD_TOOL_CATALOG.items():
        names = [slug, str(info["label"]).lower(), *info.get("aliases", ())]
        if text in names or compact == re.sub(r"[^a-z0-9]", "", slug):
            return slug
        if compact == re.sub(r"[^a-z0-9]", "", str(info["label"]).lower()):
            return slug
        if any(compact == re.sub(r"[^a-z0-9]", "", str(alias).lower()) for alias in info.get("aliases", ())):
            return slug

    return compact


def parse_cloud_tool_slugs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = str(value).split(",")

    slugs = []
    seen = set()
    for item in items:
        slug = normalize_tool_slug(str(item))
        if slug and slug not in seen:
            slugs.append(slug)
            seen.add(slug)
    return slugs


def _env_cloud_tool_slugs() -> list[str]:
    raw = os.getenv("CLOUD_TOOLKITS") or os.getenv("COMPOSIO_TOOLKITS")
    return parse_cloud_tool_slugs(raw) or list(DEFAULT_CLOUD_TOOL_SLUGS)


def load_cloud_tools_config() -> dict[str, Any]:
    config = {"enabled_cloud_tools": _env_cloud_tool_slugs()}
    path = get_cloud_tools_config_path()
    if not path.exists():
        return config

    try:
        file_config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return config

    if isinstance(file_config, dict) and "enabled_cloud_tools" in file_config:
        config["enabled_cloud_tools"] = parse_cloud_tool_slugs(file_config.get("enabled_cloud_tools"))

    if not config["enabled_cloud_tools"]:
        config["enabled_cloud_tools"] = list(DEFAULT_CLOUD_TOOL_SLUGS)
    return config


def save_cloud_tools_config(enabled_cloud_tools: list[str]) -> dict[str, Any]:
    config = {"enabled_cloud_tools": parse_cloud_tool_slugs(enabled_cloud_tools)}
    path = get_cloud_tools_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)
    return config


def load_enabled_cloud_tools() -> list[str]:
    return list(load_cloud_tools_config()["enabled_cloud_tools"])


def add_cloud_tools(names: list[str]) -> tuple[list[str], list[str]]:
    current = load_enabled_cloud_tools()
    current_set = set(current)
    added = []
    for name in names:
        slug = normalize_tool_slug(name)
        if slug and slug not in current_set:
            current.append(slug)
            current_set.add(slug)
            added.append(slug)
    save_cloud_tools_config(current)
    return current, added


def remove_cloud_tools(names: list[str]) -> tuple[list[str], list[str]]:
    requested = {normalize_tool_slug(name) for name in names}
    requested.discard("")
    current = load_enabled_cloud_tools()
    kept = [slug for slug in current if slug not in requested]
    removed = [slug for slug in current if slug in requested]
    save_cloud_tools_config(kept)
    return kept, removed


def tool_label(slug: str) -> str:
    info = CLOUD_TOOL_CATALOG.get(slug)
    return str(info.get("label") if info else slug)


def format_slug_list(slugs: list[str]) -> str:
    return ", ".join(f"{tool_label(slug)} (`{slug}`)" for slug in slugs) or "none"


def format_enabled_tools() -> str:
    enabled = load_enabled_cloud_tools()
    rows = [
        "Native tools:",
        "- Agent identity manager - update the local agent name, personality, use cases, and custom prompt.",
        "",
        "Enabled cloud tools:",
    ]
    rows.extend(
        f"- {tool_label(slug)} (`{slug}`) - {CLOUD_TOOL_CATALOG.get(slug, {}).get('description', 'Cloud app actions.')}"
        for slug in enabled
    )
    rows.extend(
        [
            "",
            "Expand with `/cloud-tools`, then enable one with `/add-tool <tool>`.",
            "Ask `/which-tool <task>` when you want to know which tool fits a request.",
        ]
    )
    return "\n".join(rows)


def format_cloud_tool_catalog() -> str:
    enabled = set(load_enabled_cloud_tools())
    rows = ["Available cloud tools:"]
    for slug, info in sorted(CLOUD_TOOL_CATALOG.items(), key=lambda item: str(item[1]["label"]).lower()):
        marker = "enabled" if slug in enabled else "available"
        rows.append(f"- {info['label']} (`{slug}`) [{marker}] - {info['description']}")
    rows.append("")
    rows.append("Enable one with `/add-tool <tool>`, for example `/add-tool gmail`.")
    return "\n".join(rows)


def recommend_tool_for_task(task: str) -> str:
    text = str(task or "").strip()
    if not text:
        return "Usage: `/which-tool <task>`\nExample: `/which-tool check my unread emails`"

    lower_text = text.lower()
    if any(word in lower_text for word in ("agent name", "personality", "use case", "custom prompt", "identity")):
        return "Use the native Agent identity manager for that request."

    matches = []
    for slug, info in CLOUD_TOOL_CATALOG.items():
        terms = [slug, str(info["label"]).lower(), *info.get("aliases", ())]
        if any(str(term).lower() in lower_text for term in terms):
            matches.append(slug)

    if matches:
        enabled = set(load_enabled_cloud_tools())
        rows = ["Recommended cloud tool:" if len(matches) == 1 else "Recommended cloud tools:"]
        for slug in matches[:5]:
            status = "enabled" if slug in enabled else f"not enabled; run `/add-tool {slug}`"
            rows.append(f"- {tool_label(slug)} (`{slug}`) - {status}")
        return "\n".join(rows)

    return (
        "I do not see a specific tool requirement from that wording. "
        "If the task needs an external app, run `/cloud-tools` to browse options."
    )


def handle_tool_command(prompt: str) -> str | None:
    stripped = str(prompt or "").strip()
    if not stripped:
        return None

    command, _, rest = stripped.partition(" ")
    command = command.lower()
    if "@" in command:
        command = command.split("@", 1)[0]

    if command == "/tools":
        return format_enabled_tools()
    if command == "/cloud-tools":
        return format_cloud_tool_catalog()
    if command == "/which-tool":
        return recommend_tool_for_task(rest)
    if command in {"/add-tool", "/enable-tool"}:
        names = rest.split()
        if not names:
            return "Usage: `/add-tool <tool>`\nExample: `/add-tool gmail`"
        enabled, added = add_cloud_tools(names)
        if added:
            return f"Enabled: {format_slug_list(added)}\n\nCurrent cloud tools: {format_slug_list(enabled)}"
        return f"Already enabled.\n\nCurrent cloud tools: {format_slug_list(enabled)}"
    if command in {"/remove-tool", "/disable-tool"}:
        names = rest.split()
        if not names:
            return "Usage: `/remove-tool <tool>`\nExample: `/remove-tool slack`"
        enabled, removed = remove_cloud_tools(names)
        if removed:
            return f"Disabled: {format_slug_list(removed)}\n\nCurrent cloud tools: {format_slug_list(enabled)}"
        return f"No matching enabled tools were found.\n\nCurrent cloud tools: {format_slug_list(enabled)}"
    return None
