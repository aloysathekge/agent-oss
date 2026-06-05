---
name: composio
description: Use external SaaS and productivity apps through cloud tools, including GitHub, Gmail, Google Calendar, Slack, Notion, and Linear.
triggers: github, gmail, google calendar, calendar, slack, notion, linear, email, inbox, meeting, schedule, issue, pull request, pr, repo, repository, send message, external app, connect app, cloud tool, google drive, drive, google sheets, spreadsheet, google docs, document, jira, trello, asana, hubspot, salesforce, discord, microsoft teams, teams, crm
---

# Cloud Tools Skill

## When to use
- The user asks to take an action in an external app or SaaS system.
- The user asks to search, read, create, update, send, post, schedule, or manage something in GitHub, Gmail, Google Calendar, Slack, Notion, Linear, or another configured cloud tool.
- The user needs to connect or authorize an external account before an action can run.

## When NOT to use
- The user is only asking a conceptual question about an app or service.
- The user asks to update this agent's own name, personality, use cases, or custom prompt; use `agent_identity_manager` instead.
- The user asks a benchmark memory question; benchmark mode skips tool routing.

## Operating rules
- Use the cloud-tool session flow to search for relevant tools, manage connections, wait for auth, and execute the chosen tool.
- If a cloud tool returns an auth or connect link, show it directly to the user and explain that they need to complete the connection before the action can run.
- For irreversible writes such as sending email, deleting content, posting messages, merging pull requests, closing issues, creating public records, or updating external systems, only execute when the user explicitly requested that final action.
- If the user asks for a draft, preview, search, or plan, do not perform the final irreversible action.
- Never invent recipients, channels, repository names, account names, calendar attendees, or destination IDs. Ask for the missing value or search with cloud tools when appropriate.
- Keep external-app results concise and report the action outcome, important IDs/links, and any next step required from the user.
