---
name: calendar
description: Check availability, schedule meetings, create events, and list upcoming events on the user's Google Calendar.
triggers: meeting, schedule, calendar, free, busy, availability, event, standup, sync, appointment
---

# Calendar Skill

## When to use
- The user asks about their schedule, availability, or what meetings they have.
- The user asks to create, schedule, or book a meeting or event.
- The user asks to check if a time slot is free.
- The user asks to block time on their calendar.

## When NOT to use
- The user is discussing dates or times conceptually (planning a project timeline, talking about deadlines without asking for a calendar action).
- The user mentions a meeting in passing without asking to create or check one.
- The user asks about someone else's calendar without involving their own.

## Tools available
- `check_availability(date, start_time, end_time)` — queries freebusy API for conflicts. Use YYYY-MM-DD for date, HH:MM for times.
- `schedule_meeting(summary, attendees, start_datetime, end_datetime)` — creates an event WITH attendee invites (sends email invitations). Attendees is a comma-separated string of email addresses.
- `create_event(summary, start_datetime, end_datetime, description="")` — creates a personal event (no attendees, no invites). Use for focus blocks, reminders, personal events.
- `list_events(time_min, time_max, max_results=10)` — lists events in a given time range. Returns id, summary, start, end per event.

## Operating rules
- **Default to `create_event` for solo events, `schedule_meeting` only when attendees are explicitly mentioned.**
- **Check availability before scheduling** if the user hasn't confirmed the time is free. Call `check_availability` first, then `schedule_meeting` or `create_event`.
- **Never invent attendee emails.** If the user says "schedule with Alice" but doesn't give her address, ask for it or search their email contacts.
- **Use the user's timezone** (PST / America/Los_Angeles, from semantic memory) when constructing ISO 8601 datetimes. Always include the timezone offset (e.g. -07:00).
- **Datetime format:** all datetime parameters use ISO 8601 (e.g. `2026-04-17T14:00:00-07:00`). Construct these from the user's natural language. `check_availability` uses simpler YYYY-MM-DD + HH:MM format.

## Examples

**User:** "Am I free tomorrow at 2pm?"
→ `check_availability(date="2026-04-18", start_time="14:00", end_time="15:00")` → report result.

**User:** "Schedule a meeting with alice@example.com tomorrow at 3pm for an hour."
→ `schedule_meeting(summary="Meeting with Alice", attendees="alice@example.com", start_datetime="2026-04-18T15:00:00-07:00", end_datetime="2026-04-18T16:00:00-07:00")` → confirm with event link.

**User:** "Block 9-11am tomorrow for deep work."
→ `create_event(summary="Deep Work Block", start_datetime="2026-04-18T09:00:00-07:00", end_datetime="2026-04-18T11:00:00-07:00")` → confirm.

**User:** "What meetings do I have this week?"
→ `list_events(time_min="2026-04-14T00:00:00-07:00", time_max="2026-04-20T23:59:59-07:00")` → summarize.

**User:** "Set up a standup with the team."
→ Ambiguous — no attendee emails, no time. Ask: "Who should be on the invite, and what time works?"
