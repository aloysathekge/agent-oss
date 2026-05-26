"""LangChain `@tool` functions for the Email skill.

Each function is a thin wrapper over the Gmail API. No routing, no prompt
engineering, no retries. The LLM decides *when* to call these; the code here
just does what it's asked and returns a stringified result.

Docstrings are deliberately concise — the LLM reads them directly as tool
descriptions when choosing which tool to invoke.
"""
import base64
from email.mime.text import MIMEText

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from .gmail_client import get_service


def _encode(to: str, subject: str, body: str) -> dict:
    """Build a Gmail-API-compatible raw message payload."""
    msg = MIMEText(body)
    msg["to"] = to
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    return {"raw": raw}


@tool
def send_email(to: str, subject: str, body: str, config: RunnableConfig) -> str:
    """Send an email immediately. IRREVERSIBLE. Returns the sent message id."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id: return "Error: Could not authenticate user identity."
    # service = get_service(user_id)
    service = get_service()
    sent = service.users().messages().send(userId="me", body=_encode(to, subject, body)).execute()
    return f"Sent. id={sent['id']}"


@tool
def draft_email(to: str, subject: str, body: str, config: RunnableConfig) -> str:
    """Create a Gmail draft WITHOUT sending. Use when the user wants to review before sending."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id: return "Error: Could not authenticate user identity."
    # service = get_service(user_id)
    service = get_service()
    draft = service.users().drafts().create(
        userId="me",
        body={"message": _encode(to, subject, body)},
    ).execute()
    return f"Draft created. id={draft['id']}"


@tool
def search_inbox(query: str, config: RunnableConfig, max_results: int = 10) -> str:
    """Search the inbox with Gmail query syntax (e.g. 'from:alice@x.com is:unread newer_than:7d').

    Returns one line per match: '<message_id> | <from> | <subject> | <snippet>'.
    Pass a message_id to `read_email` for the full body.
    """
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id: return "Error: Could not authenticate user identity."
    # service = get_service(user_id)
    service = get_service()
    resp = service.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()
    ids = [m["id"] for m in resp.get("messages", [])]

    lines = []
    for mid in ids:
        m = service.users().messages().get(
            userId="me",
            id=mid,
            format="metadata",
            metadataHeaders=["From", "Subject"],
        ).execute()
        headers = {h["name"]: h["value"] for h in m["payload"].get("headers", [])}
        snippet = m.get("snippet", "")[:120]
        lines.append(
            f"{mid} | {headers.get('From', '')} | {headers.get('Subject', '')} | {snippet}"
        )

    return "\n".join(lines) if lines else "No messages matched."


@tool
def read_email(message_id: str, config: RunnableConfig) -> str:
    """Fetch the full plain-text body of a single email. Use message_id from `search_inbox`."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id: return "Error: Could not authenticate user identity."
    # service = get_service(user_id)
    service = get_service()
    m = service.users().messages().get(userId="me", id=message_id, format="full").execute()

    # Walk MIME parts looking for text/plain; fall back to snippet if none found.
    def _extract(part) -> str:
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(part["body"]["data"]).decode(errors="ignore")
        for sub in part.get("parts", []):
            found = _extract(sub)
            if found:
                return found
        return ""

    body = _extract(m["payload"]) or m.get("snippet", "")
    headers = {h["name"]: h["value"] for h in m["payload"].get("headers", [])}
    return (
        f"From: {headers.get('From', '')}\n"
        f"Subject: {headers.get('Subject', '')}\n\n"
        f"{body}"
    )
