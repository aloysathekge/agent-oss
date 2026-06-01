import asyncio
import glob
import json
import os
import random
import re
import subprocess
import sys

import run_dataset_evals_parallel as base


# =========================================================
# PROMPT REGRESSION SAMPLE CONFIG
# =========================================================

SAMPLE_NAME = os.getenv("EVAL_SAMPLE_NAME", "prompt_regression")
SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "60"))
SAMPLE_SOURCE_LIMIT = int(os.getenv("EVAL_SAMPLE_SOURCE_LIMIT", "500"))
SAMPLE_SEED = os.getenv("EVAL_SAMPLE_SEED", "test6")
SAMPLE_QUESTION_IDS = [
    qid.strip()
    for qid in os.getenv(
        "EVAL_SAMPLE_QUESTION_IDS",
        os.getenv("EVAL_QUESTION_IDS", ""),
    ).split(",")
    if qid.strip()
]
FRESH_RUN = (
    os.getenv("EVAL_SAMPLE_FRESH", "0") == "1"
    or os.getenv("EVAL_SAMPLE_RESUME", "1") == "0"
)

RESULT_PREFIX = f"longmemeval_sample_{SAMPLE_NAME}"
MASTER_RESULTS_PATH = os.path.join(
    base.REPORTS_DIR, f"{RESULT_PREFIX}_results.json"
)
MANIFEST_PATH = os.path.join(
    base.REPORTS_DIR, f"{RESULT_PREFIX}_questions.json"
)


def worker_results_glob() -> str:
    return os.path.join(base.REPORTS_DIR, f"{RESULT_PREFIX}_results.worker*.json")


def checkpoint_glob() -> str:
    return os.path.join(base.REPORTS_DIR, f"eval_checkpoint.{SAMPLE_NAME}*.json")


def shard_glob() -> str:
    return os.path.join(base.REPORTS_DIR, f"eval_shard.{SAMPLE_NAME}.worker*.json")


def get_sample_shard_paths() -> list[str]:
    return sorted(glob.glob(shard_glob()))


def get_worker_result_paths() -> list[str]:
    return sorted(glob.glob(worker_results_glob()))


def configure_base_paths():
    worker_id = os.getenv("EVAL_WORKER_ID")

    base.NUM_WORKERS = int(os.getenv("EVAL_WORKERS", "20"))
    base.WORKER_ID = worker_id
    base.SHARD_PATH = os.getenv("EVAL_SHARD_PATH")
    base.MASTER_RESULTS_PATH = MASTER_RESULTS_PATH
    base.QUESTION_IDS = SAMPLE_QUESTION_IDS
    base.get_worker_result_paths = get_worker_result_paths

    if worker_id is not None:
        base.CHECKPOINT_PATH = os.path.join(
            base.REPORTS_DIR, f"eval_checkpoint.{SAMPLE_NAME}.worker{worker_id}.json"
        )
        base.RESULTS_PATH = os.path.join(
            base.REPORTS_DIR, f"{RESULT_PREFIX}_results.worker{worker_id}.json"
        )
    else:
        base.CHECKPOINT_PATH = os.path.join(
            base.REPORTS_DIR, f"eval_checkpoint.{SAMPLE_NAME}.parallel.json"
        )
        base.RESULTS_PATH = MASTER_RESULTS_PATH


def clear_previous_sample_outputs():
    for path in [
        MASTER_RESULTS_PATH,
        MANIFEST_PATH,
        *glob.glob(worker_results_glob()),
        *glob.glob(checkpoint_glob()),
        *glob.glob(shard_glob()),
    ]:
        if os.path.exists(path):
            os.remove(path)


def sample_state_exists() -> bool:
    return any(
        [
            os.path.exists(MASTER_RESULTS_PATH),
            os.path.exists(MANIFEST_PATH),
            bool(glob.glob(worker_results_glob())),
            bool(glob.glob(checkpoint_glob())),
            bool(glob.glob(shard_glob())),
        ]
    )


def load_sample_dataset(resume_existing: bool = False) -> list:
    dataset = base.load_dataset_local()

    if SAMPLE_QUESTION_IDS:
        by_question_id = {
            item.get("question_id"): (source_index, item)
            for source_index, item in enumerate(dataset, start=1)
            if item.get("question_id")
        }
        missing_ids = [qid for qid in SAMPLE_QUESTION_IDS if qid not in by_question_id]
        if missing_ids:
            raise RuntimeError(f"Question id(s) not found in dataset: {missing_ids}")

        selected_pairs = [by_question_id[qid] for qid in SAMPLE_QUESTION_IDS]
        manifest = [
            {
                "source_index": source_index,
                "question_id": item["question_id"],
                "question_type": item.get("question_type", "") or "",
                "question": item["question"],
                "expected_answer": item["answer"],
            }
            for source_index, item in selected_pairs
        ]
        base.save_json(MANIFEST_PATH, manifest)
        return [item for _, item in selected_pairs]

    if resume_existing and os.path.exists(MANIFEST_PATH):
        manifest = base.load_json_list(MANIFEST_PATH)
        by_question_id = {
            item.get("question_id"): item
            for item in dataset
            if item.get("question_id")
        }
        selected = [
            by_question_id[row["question_id"]]
            for row in manifest
            if isinstance(row, dict) and row.get("question_id") in by_question_id
        ]
        if selected:
            return selected

    source_limit = min(SAMPLE_SOURCE_LIMIT, len(dataset))
    source = list(enumerate(dataset[:source_limit], start=1))
    sample_size = min(SAMPLE_SIZE, len(source))

    rng = random.Random(SAMPLE_SEED)
    selected = rng.sample(source, sample_size)
    selected.sort(key=lambda pair: pair[0])

    manifest = [
        {
            "source_index": source_index,
            "question_id": item["question_id"],
            "question_type": item.get("question_type", "") or "",
            "question": item["question"],
            "expected_answer": item["answer"],
        }
        for source_index, item in selected
    ]
    base.save_json(MANIFEST_PATH, manifest)

    return [item for _, item in selected]


def load_sample_dataset_from_shards(paths: list[str]) -> list:
    dataset = []
    seen = set()

    for path in paths:
        for item in base.load_json_list(path):
            if not isinstance(item, dict):
                continue
            question_id = item.get("question_id")
            if not question_id or question_id in seen:
                continue
            dataset.append(item)
            seen.add(question_id)

    return dataset


def shard_worker_id(path: str) -> str:
    match = re.search(r"\.worker(\d+)\.json$", os.path.basename(path))
    if not match:
        raise RuntimeError(f"Could not infer worker id from shard path: {path}")
    return match.group(1)


def merge_sample_worker_results():
    merged = {}

    for row in base.load_json_list(MASTER_RESULTS_PATH):
        if isinstance(row, dict) and row.get("question_id"):
            merged[row["question_id"]] = row

    for path in get_worker_result_paths():
        for row in base.load_json_list(path):
            if isinstance(row, dict) and row.get("question_id"):
                merged[row["question_id"]] = row

    base.save_json(MASTER_RESULTS_PATH, list(merged.values()))
    print(f"✅ Merged {len(merged)} sample results into {MASTER_RESULTS_PATH}")


def get_sample_completed_question_ids() -> set[str]:
    completed = set()

    for path in [MASTER_RESULTS_PATH, *get_worker_result_paths()]:
        for row in base.load_json_list(path):
            if isinstance(row, dict) and row.get("question_id"):
                completed.add(row["question_id"])

    return completed


def print_sample_summary():
    rows = base.load_json_list(MASTER_RESULTS_PATH)
    yes_rows = [row for row in rows if row.get("result") == "YES"]
    no_rows = [row for row in rows if row.get("result") == "NO"]

    print("\n🏁 PROMPT REGRESSION SAMPLE COMPLETE")
    print(f"📄 Results: {MASTER_RESULTS_PATH}")
    print(f"📋 Sample manifest: {MANIFEST_PATH}")
    print(f"✅ YES: {len(yes_rows)}/{len(rows)}")
    print(f"❌ NO: {len(no_rows)}/{len(rows)}")

    if no_rows:
        print("\nQuestions that need review:")
        for row in no_rows:
            print(f"- {row.get('question_id')}: {row.get('question')}")
    else:
        print("🎉 All sampled historical questions passed.")


def run_parallel_sample():
    configure_base_paths()

    resume_existing = sample_state_exists() and not FRESH_RUN
    existing_shards = get_sample_shard_paths() if resume_existing else []

    if FRESH_RUN and base.WORKER_ID is None:
        clear_previous_sample_outputs()
        resume_existing = False
        existing_shards = []

    dataset = load_sample_dataset_from_shards(existing_shards) if existing_shards else []
    if not dataset:
        dataset = load_sample_dataset(resume_existing=resume_existing)
    completed_ids = get_sample_completed_question_ids()
    remaining = [item for item in dataset if item["question_id"] not in completed_ids]

    print("🚀 RUNNING LONGMEMEVAL PROMPT REGRESSION SAMPLE")
    print(f"🎲 Seed: {SAMPLE_SEED}")
    print(f"📚 Source window: first {SAMPLE_SOURCE_LIMIT} questions")
    print(f"🧪 Sample size: {len(dataset)}")
    print(f"✅ Completed already: {len(completed_ids)}")
    print(f"🧩 Remaining: {len(remaining)}")
    print(f"⚙️ Workers: {base.NUM_WORKERS}")
    print(f"📄 Results: {MASTER_RESULTS_PATH}")
    print(f"🔁 Resume existing sample: {resume_existing}")

    if not remaining:
        merge_sample_worker_results()
        print_sample_summary()
        return

    if base.NUM_WORKERS < 2:
        asyncio.run(base.run_worker_dataset(remaining))
        merge_sample_worker_results()
        print_sample_summary()
        return

    procs = []
    base_agent_id = os.getenv("AGENT_ID") or "local_agent"

    if existing_shards:
        shard_specs = [
            (shard_worker_id(shard_path), shard_path, base.load_json_list(shard_path))
            for shard_path in existing_shards
        ]
        print(f"🔁 Reusing {len(shard_specs)} existing sample shard(s) for checkpoint resume.")
    else:
        shards = base.split_round_robin(remaining, base.NUM_WORKERS)
        shard_specs = []
        for worker_id, shard in enumerate(shards):
            if not shard:
                continue
            shard_path = os.path.join(
                base.REPORTS_DIR, f"eval_shard.{SAMPLE_NAME}.worker{worker_id}.json"
            )
            base.save_json(shard_path, shard)
            shard_specs.append((str(worker_id), shard_path, shard))

    for worker_id, shard_path, shard in shard_specs:
        if not shard:
            continue
        env = os.environ.copy()
        env["EVAL_WORKER_ID"] = worker_id
        env["EVAL_SHARD_PATH"] = shard_path
        env["AGENT_ID"] = f"{base_agent_id}_{SAMPLE_NAME}_worker_{worker_id}"

        print(
            f"🚚 Starting sample worker {worker_id}: "
            f"{len(shard)} questions | AGENT_ID={env['AGENT_ID']}"
        )

        procs.append(subprocess.Popen([sys.executable, __file__], env=env))

    exit_codes = [proc.wait() for proc in procs]

    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"One or more sample eval workers failed: {exit_codes}")

    merge_sample_worker_results()
    print_sample_summary()


if __name__ == "__main__":
    configure_base_paths()

    if base.WORKER_ID is not None:
        if not base.SHARD_PATH:
            raise RuntimeError("Worker mode requires EVAL_SHARD_PATH.")

        with open(base.SHARD_PATH, "r", encoding="utf-8") as f:
            shard_dataset = json.load(f)

        asyncio.run(base.run_worker_dataset(shard_dataset))
    else:
        run_parallel_sample()
