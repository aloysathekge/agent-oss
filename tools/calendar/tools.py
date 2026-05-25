"""LangChain @tool functions for the Calendar skill.

Each function wraps a Google Calendar v3 API call. Datetimes are accepted as
ISO 8601 strings (e.g. '2026-04-17T14:00:00-07:00') — the LLM constructs
these from natural-language time references using the user's timezone from
semantic memory.
"""
from datetime import datetime, timedelta

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig # NEW IMPORT

from .gcal_client import get_service


@tool
def check_availability(date: str, start_time: str, end_time: str, config: RunnableConfig) -> str:
    """Check if the user is free during a given time window.

    Args:
        date: Date in YYYY-MM-DD format (e.g. '2026-04-17').
        start_time: Start time in HH:MM format, 24-hour (e.g. '14:00').
        end_time: End time in HH:MM format, 24-hour (e.g. '15:00').

    Returns 'Available — no conflicts' or a list of conflicting events.
    """

    # 1. EXTRACT USER ID
    user_id = config.get("configurable", {}).get("user_id")
    channel = config.get("configurable", {}).get("channel_type")



    if not user_id: return "Error: Could not authenticate user identity."

    # 2. PASS TO SERVICE to fetch user all calender token details( access_token , refresh_token , expiry etc ) from DB 
    # service = get_service(user_id)

    
    service = get_service()

    tz = "America/Los_Angeles"
    time_min = f"{date}T{start_time}:00-07:00"
    time_max = f"{date}T{end_time}:00-07:00"

    body = {
        "timeMin": time_min,
        "timeMax": time_max,
        "timeZone": tz,
        "items": [{"id": "primary"}],
    }
    result = service.freebusy().query(body=body).execute()
    busy_slots = result["calendars"]["primary"]["busy"]

    if not busy_slots:
        return f"Available — no conflicts between {start_time} and {end_time} on {date}."

    lines = [f"Busy — {len(busy_slots)} conflict(s):"]
    for slot in busy_slots:
        lines.append(f"  {slot['start']} to {slot['end']}")
    return "\n".join(lines)


@tool
def schedule_meeting(
    summary: str, attendees: str, start_datetime: str, end_datetime: str, config: RunnableConfig
) -> str:
    """Create a calendar event WITH attendee invites (sends email invitations).

    Args:
        summary: Meeting title (e.g. 'Sprint Planning').
        attendees: Comma-separated email addresses (e.g. 'alice@x.com,bob@x.com').
        start_datetime: ISO 8601 start (e.g. '2026-04-17T14:00:00-07:00').
        end_datetime: ISO 8601 end (e.g. '2026-04-17T15:00:00-07:00').
    """
    user_id = config.get("configurable", {}).get("user_id")
    channel = config.get("configurable", {}).get("channel_type")
    if not user_id: return "Error: Could not authenticate user identity."
    # service = get_service(user_id)
    service = get_service()

    attendee_list = [
        {"email": addr.strip()} for addr in attendees.split(",") if addr.strip()
    ]

    event = {
        "summary": summary,
        "start": {"dateTime": start_datetime},
        "end": {"dateTime": end_datetime},
        "attendees": attendee_list,
    }
    created = service.events().insert(
        calendarId="primary", body=event, sendUpdates="all"
    ).execute()

    return (
        f"Meeting scheduled. id={created['id']} "
        f"link={created.get('htmlLink', 'N/A')}"
    )


@tool
def create_event(
    summary: str, start_datetime: str, end_datetime: str, config: RunnableConfig, description: str = ""
) -> str:
    """Create a personal calendar event (no attendees, no invites sent).

    Args:
        summary: Event title (e.g. 'Deep Work Block').
        start_datetime: ISO 8601 start (e.g. '2026-04-17T14:00:00-07:00').
        end_datetime: ISO 8601 end (e.g. '2026-04-17T15:00:00-07:00').
        description: Optional longer description or notes for the event.
    """
    user_id = config.get("configurable", {}).get("user_id")
    channel = config.get("configurable", {}).get("channel_type")
    if not user_id: return "Error: Could not authenticate user identity."
    # service = get_service(user_id)
    service = get_service()

    event = {
        "summary": summary,
        "start": {"dateTime": start_datetime},
        "end": {"dateTime": end_datetime},
    }
    if description:
        event["description"] = description

    created = service.events().insert(calendarId="primary", body=event).execute()
    return (
        f"Event created. id={created['id']} "
        f"link={created.get('htmlLink', 'N/A')}"
    )


@tool
def list_events(time_min: str, time_max: str, config: RunnableConfig, max_results: int = 10) -> str:
    """List upcoming calendar events in a given time range.

    Args:
        time_min: ISO 8601 start of range (e.g. '2026-04-17T00:00:00-07:00').
        time_max: ISO 8601 end of range (e.g. '2026-04-18T00:00:00-07:00').
        max_results: Maximum number of events to return (default 10).

    Returns one line per event: 'id | summary | start | end'.
    """
    user_id = config.get("configurable", {}).get("user_id")
    channel = config.get("configurable", {}).get("channel_type")
    if not user_id: return "Error: Could not authenticate user identity."
    # service = get_service(user_id)
    service = get_service()

    result = service.events().list(
        calendarId="primary",
        timeMin=time_min,
        timeMax=time_max,
        maxResults=max_results,
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    events = result.get("items", [])
    if not events:
        return "No events found in the given time range."

    lines = []
    for ev in events:
        start = ev["start"].get("dateTime", ev["start"].get("date", ""))
        end = ev["end"].get("dateTime", ev["end"].get("date", ""))
        lines.append(f"{ev['id']} | {ev.get('summary', '(no title)')} | {start} | {end}")

    return "\n".join(lines)
