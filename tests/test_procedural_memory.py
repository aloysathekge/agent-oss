# Unit tests for procedural (behavioral rules) memory.
#
# Covers: loading, saving, adding rules, updating rules,
# deleting rules, and tag-based filtering logic.

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


@pytest.fixture
def agent_module(monkeypatch, tmp_path):
    """Import agent module with mocked externals, isolated to tmp_path."""
    mem_root = str(tmp_path / "local_memory")
    agent_mem_dir = os.path.join(mem_root, "test_agent")
    proc_dir = os.path.join(agent_mem_dir, "procedural_memory")
    proc_file = os.path.join(proc_dir, "rules.json")
    os.makedirs(proc_dir, exist_ok=True)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("AGENT_ID", "test_agent")
    monkeypatch.setenv("USER_ID", "test_user")
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", mem_root)

    with patch("langchain_openai.OpenAIEmbeddings", return_value=MagicMock()), \
         patch("langchain_openai.ChatOpenAI", return_value=MagicMock()), \
         patch("tools.tool_manager.discover_skills", return_value={}):
        if "agent" in sys.modules:
            del sys.modules["agent"]
        if "tools.tool_manager" in sys.modules:
            del sys.modules["tools.tool_manager"]
        if "tools" in sys.modules:
            del sys.modules["tools"]

        import agent

        # Override procedural paths to use tmp - must set BEFORE any calls
        agent.LOCAL_MEMORY_ROOT = mem_root
        agent.LOCAL_AGENT_MEMORY_DIR = agent_mem_dir
        agent.PROCEDURAL_DIR = proc_dir
        agent.PROCEDURAL_FILE = proc_file

        yield agent


class TestProceduralRulesStorage:
    """Tests for procedural rules file operations."""

    def test_load_empty_returns_list(self, agent_module):
        rules = agent_module._load_rules_file()
        assert rules == []

    def test_save_and_reload(self, agent_module):
        test_rules = [
            {
                "id": "rule-1",
                "agent_id": "test_agent",
                "rule": "Always be formal.",
                "reasoning": "User preference.",
                "target_entity": "global",
                "tags": ["tone"],
                "created_at": "2025-01-01 10:00:00",
                "updated_at": "2025-01-01 10:00:00",
            }
        ]
        agent_module._save_rules_file(test_rules)
        loaded = agent_module._load_rules_file()

        assert len(loaded) == 1
        assert loaded[0]["rule"] == "Always be formal."
        assert loaded[0]["tags"] == ["tone"]

    def test_save_procedural_rules_appends(self, agent_module):
        agent_module.save_procedural_rules([
            {"rule": "Use markdown formatting.", "tags": ["formatting"]},
            {"rule": "Never use slang.", "tags": ["tone"]},
        ])

        rules = agent_module._load_rules_file()
        assert len(rules) == 2
        assert rules[0]["rule"] == "Use markdown formatting."
        assert rules[1]["rule"] == "Never use slang."
        # Check that IDs were auto-generated
        assert "id" in rules[0]
        assert "id" in rules[1]
        assert rules[0]["id"] != rules[1]["id"]


class TestProceduralExecuteAction:
    """Tests for execute_procedural_action."""

    @pytest.mark.asyncio
    async def test_add_rule(self, agent_module):
        action = {
            "action": "ADD",
            "rule": "Respond in bullet points.",
            "reasoning": "User asked for it.",
            "target_entity": "global",
            "tags": ["formatting"],
        }
        result = await agent_module.execute_procedural_action(action)
        assert result is True

        rules = agent_module._load_rules_file()
        assert len(rules) == 1
        assert rules[0]["rule"] == "Respond in bullet points."

    @pytest.mark.asyncio
    async def test_delete_rule(self, agent_module):
        # First add a rule
        rule_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        agent_module._save_rules_file([
            {
                "id": rule_id,
                "agent_id": "test_agent",
                "rule": "To be deleted.",
                "reasoning": "",
                "target_entity": "global",
                "tags": ["global"],
                "created_at": "2025-01-01 10:00:00",
                "updated_at": "2025-01-01 10:00:00",
            }
        ])

        action = {"action": "DELETE", "id": rule_id}
        result = await agent_module.execute_procedural_action(action)
        assert result is True

        rules = agent_module._load_rules_file()
        assert len(rules) == 0

    @pytest.mark.asyncio
    async def test_update_rule(self, agent_module):
        rule_id = "b2c3d4e5-f6a7-8901-bcde-f12345678901"
        agent_module._save_rules_file([
            {
                "id": rule_id,
                "agent_id": "test_agent",
                "rule": "Old rule text.",
                "reasoning": "original",
                "target_entity": "global",
                "tags": ["tone"],
                "created_at": "2025-01-01 10:00:00",
                "updated_at": "2025-01-01 10:00:00",
            }
        ])

        action = {
            "action": "UPDATE",
            "id": rule_id,
            "rule": "New rule text.",
            "reasoning": "corrected",
            "target_entity": "global",
            "tags": ["tone", "corrected"],
        }
        result = await agent_module.execute_procedural_action(action)
        assert result is True

        rules = agent_module._load_rules_file()
        assert len(rules) == 1
        assert rules[0]["rule"] == "New rule text."
        assert rules[0]["tags"] == ["tone", "corrected"]

    @pytest.mark.asyncio
    async def test_update_nonexistent_rule(self, agent_module):
        agent_module._save_rules_file([])

        action = {
            "action": "UPDATE",
            "id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
            "rule": "Won't work.",
            "tags": ["ghost"],
        }
        result = await agent_module.execute_procedural_action(action)
        assert result is False

    @pytest.mark.asyncio
    async def test_add_without_rule_field(self, agent_module):
        action = {"action": "ADD", "reasoning": "No rule field."}
        result = await agent_module.execute_procedural_action(action)
        assert result is False


class TestLoadProceduralRules:
    """Tests for load_procedural_rules async function."""

    @pytest.mark.asyncio
    async def test_returns_sorted_by_recency(self, agent_module):
        agent_module._save_rules_file([
            {
                "id": "old",
                "agent_id": "test_agent",
                "rule": "Old rule.",
                "reasoning": "",
                "target_entity": "global",
                "tags": ["global"],
                "created_at": "2024-01-01 10:00:00",
                "updated_at": "2024-01-01 10:00:00",
            },
            {
                "id": "new",
                "agent_id": "test_agent",
                "rule": "New rule.",
                "reasoning": "",
                "target_entity": "global",
                "tags": ["global"],
                "created_at": "2025-06-01 10:00:00",
                "updated_at": "2025-06-01 10:00:00",
            },
        ])

        rules = await agent_module.load_procedural_rules()
        assert rules[0]["id"] == "new"
        assert rules[1]["id"] == "old"
