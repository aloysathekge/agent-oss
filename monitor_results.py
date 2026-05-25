import json
import time
import hashlib
import platform
import subprocess
from pathlib import Path

FILE_PATH = "reports/longmemeval_results.json"
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


def get_file_hash(path):
    if not Path(path).exists():
        return None

    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_latest_result(data):
    if not data:
        return None

    return data[-1]


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
    print(f"Watching file: {FILE_PATH}")

    last_hash = get_file_hash(FILE_PATH)

    while True:
        try:
            current_hash = get_file_hash(FILE_PATH)

            if current_hash != last_hash:
                print("File changed!")

                data = load_json(FILE_PATH)
                latest = get_latest_result(data)

                if latest:
                    announce_result(latest, data)

                last_hash = current_hash

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("Stopped.")
            break

        except Exception as e:
            print("Error:", e)
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()