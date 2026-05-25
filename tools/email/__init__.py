"""Email skill package. Exports the list of tools the router binds to the main LLM."""
from .tools import draft_email, read_email, search_inbox, send_email

# Convention: every skill package exports `<NAME>_TOOLS`.
# `tools/__init__.py` picks this up during skill discovery.
EMAIL_TOOLS = [send_email, draft_email, search_inbox, read_email]
