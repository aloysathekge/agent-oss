from supabase import create_client
import os
import time  # 🛠️ ADDED FOR RETRIES
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

AGENT_ID = os.getenv("AGENT_ID", "local_agent")

# =========================================================
# HELPER: SAFE DATABASE EXECUTION WITH RETRIES
# =========================================================
def safe_execute(query_builder, max_retries=4, delay=2.0):
    """Safely executes a Supabase query and retries if a network timeout occurs."""
    for attempt in range(max_retries):
        try:
            return query_builder.execute()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ [Network] Supabase timeout. Retrying in {delay}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"❌ [Error] Supabase query completely failed: {e}")
                raise e

# =========================================================
# ASK USER HOW TO OUTPUT LOGS
# =========================================================

save_to_file = input("Do you want to save logs to a txt file? (y/n): ").strip().lower() == "y"

log_file = None

if save_to_file:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"search_logs_{timestamp}.txt"
    print(f"📁 Logs will be saved to: {log_file}")


def log(message=""):
    """
    Print to console and optionally save to file
    """
    print(message)

    if save_to_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")

search_texts = [
    "MoMA",
    "Museum of Modern Art"
]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

for s in search_texts:

    search_text = s

    log("\n" + "#" * 120)
    log(f"🔍 SEARCHING FOR: {search_text}")
    log("#" * 120)

    # =========================================================
    # SEARCH MEMORIES
    # =========================================================

    # 🛠️ USE SAFE_EXECUTE
    memory_response = safe_execute(
        supabase.table("agent_memories")
        .select("*")
        .eq("agent_id", AGENT_ID)
        .ilike("content", f"%{search_text}%")
    )

    memory_rows = memory_response.data

    log("\n" + "=" * 100)
    log(f"🧠 FOUND {len(memory_rows)} MATCHING MEMORIES")
    log("=" * 100)

    for row in memory_rows:
        log("=" * 80)
        log(f"ID: {row.get('id')}")
        log(f"Memory Type: {row.get('memory_type')}")
        log(f"Created: {row.get('created_at')}")

        log("\nCONTENT:\n")
        log(row.get("content"))

        log("=" * 80)

    # =========================================================
    # SEARCH RULES
    # =========================================================

    # 🛠️ USE SAFE_EXECUTE
    rule_response = safe_execute(
        supabase.table("agent_rules")
        .select("*")
        .eq("agent_id", AGENT_ID)
        .ilike("rule", f"%{search_text}%")
    )

    rule_rows = rule_response.data

    log("\n" + "=" * 100)
    log(f"📜 FOUND {len(rule_rows)} MATCHING RULES")
    log("=" * 100)

    for row in rule_rows:
        log("=" * 80)
        log(f"ID: {row.get('id')}")
        log(f"Tags: {row.get('tags')}")
        log(f"Created: {row.get('created_at')}")

        log("\nRULE:\n")
        log(row.get("rule"))

        log("=" * 80)

print("\n✅ Search completed.")