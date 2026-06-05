import json
import os

DATASET_PATH = os.path.join("eval_datasets", "longmemeval_s_cleaned.json")
TARGET_QUESTION_ID = "370a8ff4"
OUTPUT_FILE = "debug_raw_data.txt"

def check_raw_data():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Find the specific question
    target_item = next((item for item in dataset if item.get("question_id") == TARGET_QUESTION_ID), None)

    if not target_item:
        print(f"❌ Could not find question ID {TARGET_QUESTION_ID} in dataset.")
        return

    
    keywords = [
    "370a8ff4",
    "How many weeks had passed since I recovered from the flu",
    "10th jog outdoors",
    "went on my 10th jog outdoors",
    "back in shape after a harsh winter",
    "recovered from the flu",
    "recently recovered from the flu",
    "recovering from the flu",
    "easing back into jogging",
    "two-week break",
    "break from jogging",
    "flu",
    "jogging",
    "outdoors",
    "April 10, 2023",
    "2023/04/10",
    "January 19, 2023",
    "2023/01/19",
    "December 26, 2022",
    "2022/12/26",
    "15 weeks"
]

    match_count = 0

    # Open file to write outputs
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        out_f.write("=" * 80 + "\n")
        out_f.write(f"🔍 RAW DATASET CHECK FOR QUESTION: {TARGET_QUESTION_ID}\n")
        out_f.write(f"❓ QUESTION: {target_item['question']}\n")
        out_f.write(f"🎯 EXPECTED ANSWER: {target_item['answer']}\n")
        out_f.write("=" * 80 + "\n\n")

        sessions = target_item.get("haystack_sessions", [])
        
        window = 2

        for session_idx, session in enumerate(sessions):
            matched_indices = []

            for msg_idx, msg in enumerate(session):
                content = msg.get("content", "")
                if any(kw.lower() in content.lower() for kw in keywords):
                    matched_indices.append(msg_idx)

            for msg_idx in matched_indices:
                match_count += 1
                start = max(0, msg_idx - window)
                end = min(len(session), msg_idx + window + 1)

                out_f.write(f"--- Session {session_idx + 1} | Match Message {msg_idx + 1} | Window {start + 1}-{end} ---\n")

                for nearby_idx in range(start, end):
                    nearby_msg = session[nearby_idx]
                    role = nearby_msg.get("role", "unknown").upper()
                    content = nearby_msg.get("content", "")
                    marker = ">>> " if nearby_idx == msg_idx else "    "
                    out_f.write(f"{marker}Message {nearby_idx + 1} ({role}):\n{content}\n\n")

                out_f.write("\n")

        out_f.write("=" * 80 + "\n")
        out_f.write(f"✅ Found {match_count} raw messages mentioning our strict keywords.\n")

    print(f"✅ Search complete! Open the file '{OUTPUT_FILE}' to see the clean, filtered results.")

if __name__ == "__main__":
    check_raw_data()