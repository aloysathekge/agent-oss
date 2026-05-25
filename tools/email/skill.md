---
name: email
description: Send, search, read, and draft emails from the user's Gmail account.
triggers: send an email, email, inbox, draft, reply, forward, check my mail, gmail
---

# Email Skill

## When to use
- The user asks to send, reply to, or forward a message.
- The user asks what's in their inbox, or to search for emails by sender / subject / keyword / date.
- The user asks you to draft an email they'll review before sending.
- The user wants to read the contents of a specific email.

## When NOT to use
- The user is talking about emails *conceptually* (how email works, writing docs about email, explaining DMARC / SPF, etc.).
- The user mentions an email address as contact info without asking for an action on it.
- The user says "send a message" without specifying email — it could mean Slack, Telegram, SMS. Ask which channel first.

## Tools available
- `send_email(to, subject, body)` — sends immediately. No undo. Use sparingly.
- `draft_email(to, subject, body)` — saves to Gmail Drafts, does NOT send. Default choice.
- `search_inbox(query, max_results=10)` — Gmail query syntax. Returns one line per match: `id | from | subject | snippet`.
- `read_email(message_id)` — full plain-text body of a single email. `message_id` comes from `search_inbox`.

## Operating rules
- **Default to `draft_email`, not `send_email`**, unless the user explicitly says "send". Sending is irreversible.
- Never invent recipients. If the "to" address isn't in the prompt or recent context, ask the user for it.
- For reply flows: `search_inbox` first to locate the thread, `read_email` to pull full context, then `draft_email`.
- Respect procedural memory when drafting bodies (tone, signature, formatting rules, commit-message style, etc.).
- Keep search queries tight — prefer `from:`, `subject:`, `newer_than:Nd`, `is:unread` over free-text.

## Examples

**User:** "Email Sarah that the security review is done."
→ Ambiguous recipient. Reply: *"Which Sarah — do you have her address, or should I search the inbox for a likely match?"*

**User:** "Draft an email to alice@example.com thanking her for the intro."
→ `draft_email(to="alice@example.com", subject="Thanks for the intro", body=<short thank-you respecting procedural tone rules>)` → confirm the draft is in Gmail Drafts for review.

**User:** "What did legal send me this week?"
→ `search_inbox(query="from:legal@ newer_than:7d", max_results=10)` → summarize matches; offer to open any specific one with `read_email`.

**User:** "Reply to the latest email from alice."
→ `search_inbox(query="from:alice", max_results=1)` → `read_email(message_id=<result>)` → `draft_email(to=<alice's address>, subject="Re: <original subject>", body=<reply grounded in the original body>)`.
