"""LangChain @tool functions for the Agent Identity Manager skill.

This allows the agent to update its own core persona parameters (name, tone, 
use cases, and custom prompts) in the Supabase database.
"""
import os
import json
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from supabase import create_client, Client

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None


@tool
def update_agent_identity(
    agent_name: str = None, 
    agent_personality: str = None, 
    agent_use_cases_json: str = None, 
    agent_custom_prompt: str = None, 
    config: RunnableConfig = None
) -> str:
    """Updates the core identity of the agent in the database."""
    
    if not supabase:
        return "Error: Database connection not configured."

    user_id = config.get("configurable", {}).get("user_id")
    agent_id = os.getenv("AGENT_ID") 

    if not user_id or not agent_id:
        return "Error: Could not authenticate user or agent identity."

    updates = {}
    
    # 🛠️ FIX: Explicitly check for truthy values (ignores None and "")
    if agent_name and str(agent_name).strip():
        updates["agent_name"] = str(agent_name).strip()
        
    if agent_personality and str(agent_personality).strip():
        # 🛠️ FIX: Enforce the "Single Word" constraint
        words = str(agent_personality).strip().split()
        if len(words) > 1:
            updates["agent_personality"] = words[0] # Take only the first word
        else:
            updates["agent_personality"] = words[0]
            
    if agent_custom_prompt and str(agent_custom_prompt).strip():
        updates["agent_custom_prompt"] = str(agent_custom_prompt).strip()
    
    if agent_use_cases_json and str(agent_use_cases_json).strip():
        try:
            parsed_cases = json.loads(agent_use_cases_json)
            if isinstance(parsed_cases, list):
                updates["agent_use_cases"] = parsed_cases
            else:
                return "Error: agent_use_cases_json must be a valid JSON array of strings."
        except json.JSONDecodeError:
            return "Error: agent_use_cases_json is not valid JSON."

    if not updates:
        return "No valid updates provided. Nothing was changed in the database."

    try:
        res = supabase.table("agent_containers").update(updates).eq("id", agent_id).execute()
        
        if len(res.data) == 0:
            return "Error: Agent record not found in the database."
            
        success_msg = "Successfully updated core identity parameters:\n"
        for k, v in updates.items():
            success_msg += f"- {k}: {v}\n"
            
        return success_msg

    except Exception as e:
        return f"Database Error: Failed to update agent identity. {str(e)}"

