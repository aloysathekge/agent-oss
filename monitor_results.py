import json
import time
import hashlib
import platform
import subprocess
from pathlib import Path

REPORTS_DIR = Path("reports")
RESULTS_PATTERN = "longmemeval_results*.json"
CHECK_INTERVAL = 2  # seconds


def speak(text):
    system = platform.system()

    try:
        if system == "Darwin":  # macOS
            subprocess.run(["say", text])

        elif system == "Linux":
            # Requires: sudo apt install espeak
            subprocess.run(["espeak", text])

        elif system == "Windows":
            command = f'''
PowerShell -Command "Add-Type -AssemblyName System.Speech;
(new-object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
'''
            subprocess.run(command, shell=True)

    except Exception as e:
        print("Speech error:", e)


def get_file_hash(path: Path):
    if not path.exists():
        return None

    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_result_files():
    return sorted(REPORTS_DIR.glob(RESULTS_PATTERN))


def get_files_hashes():
    return {path: get_file_hash(path) for path in get_result_files()}


def load_all_results():
    """Load and dedupe all result files, including worker shards."""
    merged = {}

    for path in get_result_files():
        try:
            data = load_json(path)
        except Exception as e:
            print(f"Could not load {path}: {e}")
            continue

        if not isinstance(data, list):
            continue

        for item in data:
            if not isinstance(item, dict):
                continue

            question_id = item.get("question_id")
            if question_id:
                merged[question_id] = item

    return list(merged.values())


def load_changed_results(changed_files):
    rows = []

    for path in changed_files:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"Could not load changed file {path}: {e}")
            continue

        if isinstance(data, list):
            rows.extend(item for item in data if isinstance(item, dict))

    return rows


def calculate_eval_stats(data):
    yes_count = 0
    no_count = 0

    for item in data:
        result = str(item.get("result", "")).strip().lower()

        if result == "yes":
            yes_count += 1
        elif result == "no":
            no_count += 1

    total = yes_count + no_count

    if total == 0:
        percentage = 0
    else:
        percentage = (yes_count / total) * 100

    return yes_count, no_count, total, percentage


def announce_result(item, data):
    question = item.get("question", "Unknown")
    expected = item.get("expected_answer", "Unknown")
    agent_answer = item.get("agent_answer", "Unknown")
    verdict = item.get("result", "Unknown")

    # Calculate stats
    yes_count, no_count, total, percentage = calculate_eval_stats(data)

    full_text = f"""
    New Result came.
    Question was: {question}.
    Expected answer is: {expected}.
    Agent answer was: {agent_answer}.
    Final verdict is: {verdict}.

    Total yes: {yes_count}
    Total no: {no_count}
    Evaluation percentage: {percentage:.2f} percent
    """

    print("\n==============================")
    print(full_text)
    print("==============================\n")

    # Alert
    for _ in range(3):
        speak("New Result came")

    # Speak details
    speak(f"Question was {question}")
    speak(f"Expected answer is {expected}")
    speak(f"Agent answer was {agent_answer}")
    speak(f"Final verdict is {verdict}")

    # Speak evaluation stats
    speak(
        f"Agent evaluation percentage is "
        f"{yes_count} out of {total}. "
        f"Which is {percentage:.2f} percent"
    )


def main():
    print(f"Watching files: {REPORTS_DIR / RESULTS_PATTERN}")

    last_hashes = get_files_hashes()
    announced_ids = {
        item.get("question_id")
        for item in load_all_results()
        if isinstance(item, dict) and item.get("question_id")
    }

    while True:
        try:
            current_hashes = get_files_hashes()

            changed_files = [
                path
                for path, current_hash in current_hashes.items()
                if last_hashes.get(path) != current_hash
            ]

            if changed_files:
                print("Files changed:", ", ".join(str(path) for path in changed_files))

                all_data = load_all_results()
                changed_rows = load_changed_results(changed_files)
                new_rows = [
                    item
                    for item in changed_rows
                    if item.get("question_id")
                    and item.get("question_id") not in announced_ids
                ]

                for item in new_rows:
                    announce_result(item, all_data)
                    announced_ids.add(item["question_id"])

                last_hashes = current_hashes

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("Stopped.")
            break

        except Exception as e:
            print("Error:", e)
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
