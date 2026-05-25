---
name: agent_identity_manager
description: Update the agent's core name, overarching personality, use cases, or custom system directives.
triggers: change your name, your new name is, from now on act like, change your personality, update your core instructions, your primary job is
---

# Agent Identity Manager Skill

## When to use
- The user explicitly tells you to change your name (e.g., "From now on your name is Robin").
- The user gives you a fundamental, overarching personality shift (e.g., "Be professional").
- The user redefines your primary purpose (e.g., "Your main job is now to help me code").
- The user provides a strict, global system directive (e.g., "Always speak in Spanish").

## When NOT to use
- The user is giving a temporary instruction for the current conversation.
- The user is asking you to remember a fact about *them* or the world (that goes into Semantic Memory).
- The user is giving a specific formatting rule for a specific task.

## Tools available
- `update_agent_identity(agent_name, agent_personality, agent_use_cases_json, agent_custom_prompt)` — Updates the core database record for this agent.

## Operating rules
- **OMIT UNCHANGED FIELDS:** You must ONLY pass arguments for the fields you are specifically instructed to change. Do NOT pass empty strings `""` for other fields. Completely omit them from the tool call.
- **`agent_personality` MUST BE ONE WORD:** You are strictly forbidden from passing phrases. If the user says "Be highly technical, precise, and Jarvis-like", you must compress that to a single word like `"technical"` or `"precise"`.
- **`agent_use_cases_json`** must be a strictly formatted JSON array of strings, e.g., `'["Code Review", "Scheduling"]'`.
- After successfully updating your identity, you must immediately adopt the new persona in your final response to the user.

## Examples

**User:** "From now on, I want your name to be Kakarot."
→ `update_agent_identity(agent_name="Kakarot")`

**User:** "Stop being so professional. I want you to be highly sarcastic and witty."
→ `update_agent_identity(agent_personality="sarcastic")`

**User:** "Your new primary jobs are reviewing my Python code and organizing my calendar."
→ `update_agent_identity(agent_use_cases_json='["Code Review", "Calendar Management"]')`

**User:** "I want you to always append 'Sir' to the end of your sentences from now on."
→ `update_agent_identity(agent_custom_prompt="Always append 'Sir' to the end of every sentence.")`