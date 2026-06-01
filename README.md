# Quarq Agent

**Local memory. Hybrid retrieval. Self-correcting reasoning. Benchmark-grade recall.**

Quarq Agent is a memory-first AI agent built by QuarqLabs for long-context personal intelligence, grounded recall, temporal reasoning, quantitative reasoning, and tool use.

It is designed as an open, inspectable alternative to memory agents such as Hermes or OpenClaw, with a stronger emphasis on durable local memory, strict attribution, self-correcting retrieval, and benchmark-grade long-term recall.

The current local implementation includes the structured artifact learning pipeline in `agent.py`. Normal prose learning still runs as before, while tables, lists, artifact blocks, quotes, budgets, timelines, metrics, ratios, and other compact evidence formats are extracted into deterministic memory units.

Local LongMemEval-S reports are checkpoints while extractor behavior is being validated. Treat checked-in report files as local progress snapshots, not final published benchmark numbers.

## Contents

- [Why Quarq Exists](#why-quarq-exists)
- [What Makes It Different](#what-makes-it-different)
- [Highlights](#highlights)
- [Architecture](#architecture)
- [Memory System](#memory-system)
- [Structured Artifact Learning](#structured-artifact-learning)
- [Local Storage Layout](#local-storage-layout)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Temporal Reasoning](#temporal-reasoning)
- [Quantitative Reasoning](#quantitative-reasoning)
- [Self-Correcting Retrieval](#self-correcting-retrieval)
- [Learning Pipeline](#learning-pipeline)
- [Tool System](#tool-system)
- [Benchmarks](#benchmarks)
- [Current Local Metrics](#current-local-metrics)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Repository Map](#repository-map)
- [Design Principles](#design-principles)
- [Status](#status)
- [License](#license)

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
- structured artifact extraction for table rows, lists, blocks, quotes, budgets, timelines, metrics, ratios, and other evidence-shaped outputs
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
- Structured artifact learning: high-signal rows and items are learned as separate deterministic memories without removing the normal prose learning path.
- Duplicate protection: batch writes skip exact duplicate content before embedding, then use the normal vector duplicate check so structured extraction does not flood memory with repeated rows.
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

## Structured Artifact Learning

The local agent keeps the original normal-text learning path intact. Full user and assistant turns are still passed to the learning model, so ordinary narrative details, decisions, preferences, and summaries can become semantic, episodic, or procedural memories.

Structured extractors run beside that path for artifact-shaped content that summarization can otherwise compress too aggressively. They create deterministic episodic units for high-signal data such as:

- markdown table rows
- explicit artifact blocks such as `::title:: == description`
- numbered sections for objectives, parameters, methods, options, steps, recommendations, and similar headings
- recommendation, remedy, dish, shop, restaurant, and product list items
- ingredient and material items
- budget, cost, allocation, and campaign plan rows
- timeline and dated event clauses
- implementation or "uses algorithm/tool" relationships
- attributed quotations and exact source claims
- metric, percentage, improvement, and score relationships
- ratios, dilutions, and mixture instructions
- music sections, chord/note style rows, and chess move notation
- counted entity headings such as encounter counts, party sizes, item totals, or named grouped entities

The extractor layer is intentionally capped and high-signal. It is not meant to memorize every sentence. Its job is to preserve compact data-bearing rows and items that future recall questions often target verbatim.

Extracted units are injected into the learning prompt as `STRUCTURED ARTIFACT UNITS` and appended as episodic `ADD` actions when the learning model omits them. They still pass through the normal local `execute_actions` path, including exact duplicate blocking, batch embedding, and vector duplicate checks.

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
- structured artifact units from high-signal tables, lists, blocks, and row-like content

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
- keeps normal prose learning active even when structured extractors also find tables, lists, blocks, or other artifact units
- uses exact duplicate blocking and capped deterministic extraction to keep structured memories controlled

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
7. Passes `question_type` into benchmark mode, stores it with each result row, and writes results to `reports/longmemeval_results.json`

### Parallel Evaluation

For faster local benchmark runs, Quarq also includes a process-based parallel evaluator:

```bash
EVAL_WORKERS=5 python run_dataset_evals_parallel.py
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
...
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

The default is `5` workers. Choose `EVAL_WORKERS` based on the machine and API limits, then increase until OpenAI rate limits or local CPU pressure become the bottleneck.

### Prompt Regression Samples

For faster iteration on prompt and retrieval changes, use the sample runner:

```bash
python run_dataset_evals_sample.py
```

It reuses the parallel evaluator, selects a deterministic sample, writes a manifest of sampled questions, keeps per-worker checkpoints, and merges results into a sample-specific report. Useful environment variables:

```text
EVAL_SAMPLE_NAME=prompt_regression
EVAL_SAMPLE_SIZE=60
EVAL_SAMPLE_SOURCE_LIMIT=500
EVAL_SAMPLE_SEED=test6
EVAL_SAMPLE_QUESTION_IDS=<comma-separated ids>
EVAL_SAMPLE_FRESH=1
EVAL_WORKERS=20
```

To monitor live results across both the main and worker result files:

```bash
python monitor_results.py
```

Current local report files:

```text
reports/longmemeval_results.json
reports/longmemeval_results.worker*.json
```

### Current Local Metrics

Current local LongMemEval-S metrics, computed from `reports/longmemeval_results.json` and joined with `eval_datasets/longmemeval_s_cleaned.json` by `question_id`:

| Question type | Correct | Incorrect | Total | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Overall | 491 | 9 | 500 | 98.20% |
| knowledge-update | 77 | 1 | 78 | 98.72% |
| multi-session | 129 | 4 | 133 | 96.99% |
| single-session-assistant | 56 | 0 | 56 | 100.00% |
| single-session-preference | 30 | 0 | 30 | 100.00% |
| single-session-user | 70 | 0 | 70 | 100.00% |
| temporal-reasoning | 129 | 4 | 133 | 96.99% |

These metrics represent the current LongMemEval-S progress while Quarq Agent is actively being improved. Some answers may change as failing or uncertain questions are rerun and fixes are added.

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
