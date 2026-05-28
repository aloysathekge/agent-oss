# Quarq Agent

**Local memory. Hybrid retrieval. Self-correcting reasoning. Benchmark-grade recall.**

Quarq Agent is a memory-first AI agent built by QuarqLabs for long-context personal intelligence, grounded recall, temporal reasoning, quantitative reasoning, and tool use.

It is designed as an open, inspectable alternative to memory agents such as Hermes or OpenClaw, with a stronger emphasis on durable local memory, strict attribution, self-correcting retrieval, and benchmark-grade long-term recall.

The current local benchmark run is in progress on LongMemEval-S and is tracking in the 99.6% to 99.7% range. The latest checked report in `reports/longmemeval_results.json` shows `256` yes, `1` no, or `99.61%`, with the full 500-question run still continuing.

## Why Quarq Exists

Most agents can chat. Fewer can remember. Almost none can remember carefully.

Quarq Agent is built around a simple idea: memory is not just vector search. A serious memory agent needs to know what a memory means, when it happened, what numbers belong to, which entity a fact is attached to, when evidence is incomplete, and when it must search again instead of guessing.

Quarq combines:

- local FAISS vector memory
- semantic, episodic, and procedural memory separation
- hybrid vector plus keyword retrieval
- HyDE-style query expansion
- dynamic recall depth
- strict temporal grounding
- numeric attribution and exact aggregation rules
- self-correcting fallback retrieval
- background memory consolidation
- LangGraph orchestration
- progressive tool routing

The result is an agent that behaves less like a stateless chatbot and more like a disciplined cognitive system.

## What Makes It Different

Quarq is not a wrapper around a vector database. It is a full memory reasoning loop.

Standard RAG systems usually fail long-memory tasks for one of four reasons:

1. They retrieve the wrong memory.
2. They retrieve the right memory but attach it to the wrong entity.
3. They confuse storage time with event time.
4. They calculate with nearby numbers that do not belong to the question.

Quarq directly attacks those failure modes with retrieval decomposition, evidence attribution, temporal guardrails, numeric scope checks, and a second-pass recovery path when the first context is incomplete.

## Highlights

- Local-first memory: no Supabase pgvector dependency. Memories and rules are saved under `local_memory/<AGENT_ID>/`.
- Three memory types: semantic facts, episodic events, and procedural behavioral rules.
- FAISS-backed retrieval: normalized OpenAI embeddings with `IndexFlatIP` cosine-style similarity.
- Hybrid search: every retrieval pass combines vector search and direct keyword matching.
- HyDE query optimizer: rewrites the user prompt into multiple retrieval probes before search.
- Dynamic thresholds: wide-net `deep` mode for aggregation, timelines, broad categories, and recommendations; stricter `standard` mode for point facts.
- Required-data fallback: the model can request a targeted second retrieval pass when evidence is missing.
- Temporal truth protocol: separates database storage time from narrative event time.
- Quantitative fidelity: numbers are stored and used with owner, property, item, and exactness.
- Background learning: user responses return immediately while memory extraction runs asynchronously.
- Progressive tool loading: tool docs are only injected when a skill is selected.
- Benchmark mode: disables tool routing and waits for pending background learning before final evaluation.

## Architecture

```text
User / API
    |
    v
LangGraph StateGraph
    |
    +-- retrieve_memories
    |     |
    |     +-- HyDE query generation
    |     +-- semantic FAISS search
    |     +-- episodic FAISS search
    |     +-- keyword search
    |     +-- procedural rule routing
    |
    +-- route_tools
    |     |
    |     +-- skill catalog router
    |     +-- progressive skill markdown loading
    |
    +-- generate_response
          |
          +-- grounded answer synthesis
          +-- optional ReAct tool loop
          +-- REQUIRED_DATA fallback retrieval
          +-- background memory learning
```

The graph is intentionally compact:

```python
START -> retrieve_memories -> route_tools -> generate_response -> END
```

Learning is launched in the background from `generate_response`, which keeps the interactive path fast while still preserving durable memory.

## Memory System

Quarq uses three memory layers.

### Semantic Memory

Semantic memory stores durable user facts:

- identity
- preferences
- relationships
- routines
- long-term projects
- possessions
- stable traits
- active statuses and inventories

Example:

```text
User owns a crystal chandelier that originally belonged to their great-grandmother and was given to them by their aunt.
```

### Episodic Memory

Episodic memory stores events and interaction history:

- what happened
- when it happened
- who was involved
- what was decided
- what the user asked for
- what changed

Example:

```text
On March 4, 2023, user received a crystal chandelier from their aunt that originally belonged to their great-grandmother.
```

### Procedural Memory

Procedural memory stores behavioral rules:

- tone preferences
- formatting preferences
- project-specific instructions
- forbidden wording
- content generation constraints

Procedural rules are tagged and routed, so the model sees only the relevant rules for the current prompt instead of carrying every rule forever.

## Local Storage Layout

The current runtime stores memory locally:

```text
local_memory/
  <AGENT_ID>/
    semantic_memory/
      index.faiss
      memories.json
    episodic_memory/
      index.faiss
      memories.json
    procedural_memory/
      rules.json
```

`AGENT_ID` determines which memory folder is used. Reusing the same `AGENT_ID` reuses the same memory. Changing `AGENT_ID` gives you a clean isolated agent profile.

Each semantic and episodic memory record includes:

- UUID
- agent ID
- memory type
- content
- embedding
- created timestamp
- updated timestamp

The formatted retrieval output intentionally preserves this shape:

```text
[STORED_AT: 2026-05-25 14:00:00] [ID: <uuid>] <memory content>
```

Downstream deduplication, recency sorting, contradiction handling, and memory update logic all depend on that stable format.

## Retrieval Pipeline

Quarq does not simply embed the latest user prompt and hope for the best.

The retrieval node first asks a lightweight model to produce a structured search plan:

```json
{
  "vector_queries": [
    "User total driving duration and travel history",
    "User vehicle, road trip, transit records",
    "User travel milestones, driving time calculation",
    "hours, road trip, destinations"
  ],
  "keywords": "driving, hours, total",
  "search_mode": "deep"
}
```

Then it performs:

1. Semantic vector search
2. Episodic vector search
3. Semantic keyword search
4. Episodic keyword search
5. ID-based deduplication
6. recency sorting
7. procedural rule routing

Search modes:

- `standard`: strict retrieval for point facts, threshold `0.38`
- `deep`: wide recall for totals, timelines, histories, recommendations, and broad categories, threshold `0.28`

This is why Quarq can answer questions that require multiple memories rather than only nearest-neighbor recall.

## Temporal Reasoning

Quarq treats time as evidence, not decoration.

The agent distinguishes:

- storage timestamp: when the memory was saved
- narrative date: when the event actually happened
- benchmark current date: the simulated date for an evaluation question
- relative dates: phrases like "yesterday", "today", "last month"

The Temporal Truth Protocol prevents common long-memory errors:

- using database timestamps as event dates
- borrowing dates from nearby but unrelated memories
- assuming a discussion date is the same as an event date
- calculating date gaps from guessed anchors

For "how long ago" questions, Quarq searches for the named event first and only uses the current date as the calculation anchor after retrieval.

## Quantitative Reasoning

Long-memory benchmarks often punish sloppy number handling. Quarq's numeric protocol is built to avoid that.

For totals, counts, durations, prices, quantities, or money questions, the model must identify:

- actor or entity
- measured action or property
- event or item
- exactness

It excludes numbers that are merely nearby or topically related.

Example:

```text
User helped organize a concert, which raised over $5,000.
```

The amount belongs to the concert, not automatically to the user. It is also a lower-bound value, not an exact addend.

For exact totals, Quarq sums only exact unqualified values unless the user explicitly asks for a minimum, estimate, or range.

## Self-Correcting Retrieval

If the first retrieval pass does not contain enough evidence, Quarq can emit:

```json
{
  "agent_response": "",
  "flags": ["REQUIRED_DATA"],
  "hyde_queries": ["aunt meetup", "received chandelier", "chandelier handoff"]
}
```

The runtime then performs a targeted fallback search and regenerates the answer with expanded context.

The fallback pass has a strict final verification rule: if the exact target is still missing, the agent must say the information is not available instead of guessing.

This makes the agent aggressive about recall but conservative about truth.

## Learning Pipeline

After every non-benchmark response, Quarq starts background learning.

The learning model extracts:

- semantic memories
- episodic memories
- procedural rules

It can issue:

```json
{
  "actions": [
    {"action": "ADD", "content": "New memory"},
    {"action": "UPDATE", "id": "uuid", "content": "Updated memory"},
    {"action": "DELETE", "id": "uuid"}
  ]
}
```

Important learning behaviors:

- preserves specific names and proper nouns
- resolves relative dates using the current date
- anchors transfer and acquisition events
- preserves every number and qualifier
- avoids duplicate memories across semantic and episodic layers
- prefers exact values over approximate or bounded restatements
- updates existing records instead of creating conflicting duplicates

Background learning is protected by:

- persistent retry loop
- exponential backoff
- concurrency limit of 4 learning tasks
- benchmark synchronization before final questions

## Tool System

Quarq includes a progressive skill router.

Instead of injecting all tool instructions into every prompt, the router sees a compact catalog and selects only the relevant skills. The generation model then receives the full markdown and bound tools for those selected skills.

Current included skills:

- email
- calendar
- PDF generation
- agent identity management

Tool execution uses a ReAct loop with a maximum of 5 iterations. If the loop reaches the limit, the model is forced to stop calling tools and produce a final text response.

### Adding A New Tool

Tool expansion is deliberately simple. To add a new capability, drop a new folder inside `tools/` and follow the existing skill convention.

```text
tools/
  your_tool/
    skill.md
    __init__.py
    tools.py
```

`skill.md` defines the router-facing metadata:

```markdown
---
name: your_tool
description: One-line description of what this skill can do.
triggers: keyword one, keyword two, natural language trigger
---

# Your Tool Skill

Describe when to use it, when not to use it, available tools, and operating rules.
```

`tools.py` defines LangChain tool callables, and `__init__.py` exports them using the folder-name convention:

```python
from .tools import your_first_tool, your_second_tool

YOUR_TOOL_TOOLS = [your_first_tool, your_second_tool]
```

That is it. `tools/__init__.py` automatically scans every subdirectory with a `skill.md`, imports the package, reads the frontmatter, and registers the exported `<FOLDER_NAME>_TOOLS` list. No central registry edit is required.

This gives Quarq a clean capability expansion path: add a folder, describe the skill, export the tools, restart the process, and the agent can route to the new capability.

## Benchmarks

Quarq includes a LongMemEval runner:

```bash
python run_dataset_evals.py
```

The benchmark pipeline:

1. Loads `eval_datasets/longmemeval_s_cleaned.json`
2. Splits haystack sessions into chunks
3. Feeds each chunk through the agent with learning enabled
4. Waits for background memories before final questions
5. Asks the benchmark question with learning disabled
6. Judges with a binary evaluator
7. Writes results to `reports/longmemeval_results.json`

### Parallel Evaluation

For faster local benchmark runs, Quarq also includes a process-based parallel evaluator:

```bash
EVAL_WORKERS=2 python run_dataset_evals_parallel.py
```

The parallel runner is designed for long benchmark runs where you do not want to lose completed progress. It first loads all completed question IDs from:

```text
reports/longmemeval_results.json
reports/longmemeval_results.worker*.json
```

Then it calculates the remaining questions, splits only those questions across the requested number of workers, and launches one isolated process per worker.

Each worker receives its own `AGENT_ID`:

```text
<AGENT_ID>_eval_worker_0
<AGENT_ID>_eval_worker_1
<AGENT_ID>_eval_worker_2
```

That means each worker gets an isolated FAISS memory folder under `local_memory/`, so multiple questions can be learned and evaluated at the same time without memory collision.

Worker outputs are written independently:

```text
reports/longmemeval_results.worker0.json
reports/longmemeval_results.worker1.json
reports/longmemeval_results.worker2.json
```

When all workers finish, the runner merges worker outputs back into:

```text
reports/longmemeval_results.json
```

Choose `EVAL_WORKERS` based on the machine and API limits. A good starting point is the number of performance cores you are comfortable dedicating to the run, then increase until OpenAI rate limits or local CPU pressure become the bottleneck. On an 8-core machine, `2` to `4` workers is usually a practical starting range; on larger machines, scale upward gradually.

To monitor live results across both the main and worker result files:

```bash
python monitor_results.py
```

Current checked local report:

```text
LongMemEval-S checkpoint: 240 yes / 1 no
Accuracy: 99.59%
Status: 500-question run in progress
```

These results are local and in progress until independently reproduced from a clean checkout.

## Requirements

- Python 3.11 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Quick Start

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```bash
OPENAI_API_KEY=your_api_key
USER_ID=local_user
AGENT_ID=local_agent
LOCAL_MEMORY_ROOT=local_memory
```

Run the terminal agent:

```bash
python agent.py
```

Run the API server:

```bash
uvicorn main:app --reload
```

Call the API:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What do you remember about me?", "channel_type": "web"}'
```

## Environment Variables

| Variable | Required | Description |
|---|---:|---|
| `OPENAI_API_KEY` | yes | Used for generation, retrieval planning, learning, and embeddings. |
| `AGENT_ID` | no | Selects the local memory namespace. Defaults to `local_agent`. |
| `USER_ID` | API only | Required by `main.py` for the FastAPI worker. |
| `LOCAL_MEMORY_ROOT` | no | Root folder for local memory. Defaults to `local_memory`. |
| `AGENT_NAME` | no | Runtime persona name. |
| `AGENT_PERSONALITY` | no | Runtime tone/personality. |
| `AGENT_USE_CASES` | no | Comma-separated use-case description. |
| `AGENT_CUSTOM_PROMPT` | no | Extra custom behavior instructions. |

## Repository Map

```text
agent.py                  Core LangGraph agent, memory, retrieval, generation, learning
agent_connector.py        Public async integration gateway
main.py                   FastAPI single-tenant worker
run_dataset_evals.py      LongMemEval evaluation runner
run_dataset_evals_parallel.py
                          Parallel LongMemEval evaluation runner
monitor_results.py        Benchmark monitoring helper
tools/                    Skill registry and tool implementations
eval_datasets/            Cleaned LongMemEval dataset
reports/                  Evaluation outputs and checkpoints
local_memory/             Local FAISS and JSON memory stores
```

## Design Principles

Quarq is built around a few hard rules:

- Retrieve broadly, reason narrowly.
- Store memories with ownership, dates, and qualifiers intact.
- Prefer saying "missing data" over inventing an answer.
- Treat temporal and numeric claims as evidence-bound operations.
- Keep user-facing latency low by learning in the background.
- Keep the context window clean with routing and progressive disclosure.
- Make the memory system portable, local, and easy to inspect.

## Status

Quarq Agent v0.4.0 is an active OSS release candidate.

The current version is optimized for long-memory evaluation and single-user local memory. The next natural steps are:

- package cleanup
- dependency trimming
- unit tests for memory storage and retrieval
- reproducible benchmark scripts
- Docker packaging
- memory compaction and archival policies
- multi-user serving with isolated local stores

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
