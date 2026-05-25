"""Calendar skill package. Exports the list of tools the router binds to the main LLM."""
from .tools import check_availability, create_event, list_events, schedule_meeting

CALENDAR_TOOLS = [check_availability, schedule_meeting, create_event, list_events]
