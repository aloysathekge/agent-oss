# Unit tests for pure helper functions in agent.py.
#
# Covers: extract_pure_text, extract_json_block,
# sort_memories_by_recency, get_formatted_rules_with_ids,
# get_token_metrics.

import os
import sys
import json
from unittest.mock import patch, MagicMock

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ==========================================
# IMPORT FIXTURE
# ==========================================
@pytest.fixture(autouse=True)
def _import_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("AGENT_ID", "test_agent")
    monkeypatch.setenv("USER_ID", "test_user")
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "mem"))

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
        yield agent


# ==========================================
# extract_pure_text
# ==========================================
class TestExtractPureText:
    """Tests for extract_pure_text helper."""

    def test_plain_string(self, _import_agent):
        agent = _import_agent

        class FakeResponse:
            content = "Hello world"

        assert agent.extract_pure_text(FakeResponse()) == "Hello world"

    def test_strips_thinking_tags(self, _import_agent):
        agent = _import_agent

        class FakeResponse:
            content = "<thinking>internal reasoning</thinking>Final answer."

        assert agent.extract_pure_text(FakeResponse()) == "Final answer."

    def test_nested_thinking_tags(self, _import_agent):
        agent = _import_agent

        class FakeResponse:
            content = "Start <thinking>some thought</thinking> middle <thinking>another</thinking> end"

        result = agent.extract_pure_text(FakeResponse())
        assert "thinking" not in result
        assert "Start" in result
        assert "end" in result

    def test_list_content(self, _import_agent):
        agent = _import_agent

        class FakeResponse:
            content = [{"text": "Part 1. "}, {"text": "Part 2."}]

        assert agent.extract_pure_text(FakeResponse()) == "Part 1. Part 2."

    def test_list_with_non_dict(self, _import_agent):
        agent = _import_agent

        class FakeResponse:
            content = ["simple string", "another"]

        result = agent.extract_pure_text(FakeResponse())
        assert "simple string" in result


# ==========================================
# extract_json_block
# ==========================================
class TestExtractJsonBlock:
    """Tests for extract_json_block helper."""

    def test_clean_json_object(self, _import_agent):
        agent = _import_agent
        text = '{"agent_response": "Hello!", "flags": [], "hyde_queries": []}'
        result = agent.extract_json_block(text)
        parsed = json.loads(result)
        assert parsed["agent_response"] == "Hello!"

    def test_json_in_markdown_fences(self, _import_agent):
        agent = _import_agent
        text = '```json\n{"key": "value"}\n```'
        result = agent.extract_json_block(text)
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_json_with_surrounding_text(self, _import_agent):
        agent = _import_agent
        text = 'Here is my response:\n{"answer": 42}\nThat is all.'
        result = agent.extract_json_block(text)
        parsed = json.loads(result)
        assert parsed["answer"] == 42

    def test_json_with_thinking_prefix(self, _import_agent):
        agent = _import_agent
        text = '<thinking>Let me analyze...</thinking>\n{"result": "done"}'
        result = agent.extract_json_block(text)
        parsed = json.loads(result)
        assert parsed["result"] == "done"

    def test_array_mode(self, _import_agent):
        agent = _import_agent
        text = 'Output: [{"action": "ADD", "content": "Fact"}]'
        result = agent.extract_json_block(text, is_array=True)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["action"] == "ADD"

    def test_nested_json_returns_outermost(self, _import_agent):
        agent = _import_agent
        text = '{"outer": {"inner": "value"}, "key": "top"}'
        result = agent.extract_json_block(text)
        parsed = json.loads(result)
        assert "outer" in parsed
        assert parsed["key"] == "top"

    def test_empty_input(self, _import_agent):
        agent = _import_agent
        assert agent.extract_json_block("") == ""
        assert agent.extract_json_block(None) == ""

    def test_no_json_present(self, _import_agent):
        agent = _import_agent
        assert agent.extract_json_block("Just plain text with no braces.") == ""

    def test_multiple_json_objects_returns_last_outermost(self, _import_agent):
        agent = _import_agent
        text = '{"first": 1}\nSome text\n{"second": 2}'
        result = agent.extract_json_block(text)
        parsed = json.loads(result)
        # Should return the last outermost
        assert "second" in parsed


# ==========================================
# sort_memories_by_recency
# ==========================================
class TestSortMemoriesByRecency:
    """Tests for sort_memories_by_recency helper."""

    def test_sorts_newest_first(self, _import_agent):
        agent = _import_agent
        block = (
            "[STORED_AT: 2024-01-01 10:00:00] [ID: aaa] Old memory\n"
            "[STORED_AT: 2024-06-15 10:00:00] [ID: bbb] Middle memory\n"
            "[STORED_AT: 2025-01-01 10:00:00] [ID: ccc] Newest memory"
        )
        result = agent.sort_memories_by_recency(block)
        lines = result.strip().split("\n")
        assert "Newest" in lines[0]
        assert "Old" in lines[-1]

    def test_respects_max_lines(self, _import_agent):
        agent = _import_agent
        lines = [
            f"[STORED_AT: 2024-01-{i:02d} 10:00:00] [ID: id{i}] Memory {i}"
            for i in range(1, 21)
        ]
        block = "\n".join(lines)
        result = agent.sort_memories_by_recency(block, max_lines=5)
        output_lines = [line for line in result.strip().split("\n") if line.strip()]
        assert len(output_lines) == 5

    def test_empty_input(self, _import_agent):
        agent = _import_agent
        assert agent.sort_memories_by_recency("") == "None"
        assert agent.sort_memories_by_recency("None") == "None"

    def test_lines_without_timestamps_go_last(self, _import_agent):
        agent = _import_agent
        block = (
            "No timestamp here\n"
            "[STORED_AT: 2025-03-01 10:00:00] [ID: xxx] Has timestamp"
        )
        result = agent.sort_memories_by_recency(block)
        lines = result.strip().split("\n")
        assert "Has timestamp" in lines[0]


# ==========================================
# get_formatted_rules_with_ids
# ==========================================
class TestGetFormattedRules:
    """Tests for get_formatted_rules_with_ids helper."""

    def test_formats_rules_correctly(self, _import_agent):
        agent = _import_agent
        rules = [
            {
                "id": "uuid-1",
                "rule": "Always use formal tone.",
                "created_at": "2025-03-15 09:30:00",
            },
            {
                "id": "uuid-2",
                "rule": "Never use emojis.",
                "created_at": "2025-04-01 12:00:00",
            },
        ]
        result = agent.get_formatted_rules_with_ids(rules)
        assert "[ID: uuid-1]" in result
        assert "[ID: uuid-2]" in result
        assert "formal tone" in result
        assert "emojis" in result

    def test_respects_limit(self, _import_agent):
        agent = _import_agent
        rules = [
            {"id": f"id-{i}", "rule": f"Rule {i}.", "created_at": "2025-01-01 00:00:00"}
            for i in range(20)
        ]
        result = agent.get_formatted_rules_with_ids(rules, limit=3)
        lines = [line for line in result.strip().split("\n") if line.strip()]
        assert len(lines) == 3

    def test_empty_rules(self, _import_agent):
        agent = _import_agent
        assert agent.get_formatted_rules_with_ids([]) == ""
        assert agent.get_formatted_rules_with_ids(None) == ""


# ==========================================
# get_token_metrics
# ==========================================
class TestGetTokenMetrics:
    """Tests for get_token_metrics helper."""

    def test_extracts_from_usage_metadata(self, _import_agent):
        agent = _import_agent

        class FakeResponse:
            usage_metadata = {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }

        metrics = agent.get_token_metrics(FakeResponse())
        assert metrics["input"] == 100
        assert metrics["output"] == 50
        assert metrics["total"] == 150

    def test_extracts_from_response_metadata(self, _import_agent):
        agent = _import_agent

        class FakeResponse:
            usage_metadata = None
            response_metadata = {
                "token_usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 80,
                    "total_tokens": 280,
                }
            }

        metrics = agent.get_token_metrics(FakeResponse())
        assert metrics["input"] == 200
        assert metrics["output"] == 80
        assert metrics["total"] == 280

    def test_missing_metadata_returns_zeros(self, _import_agent):
        agent = _import_agent

        class FakeResponse:
            usage_metadata = None
            response_metadata = {}

        metrics = agent.get_token_metrics(FakeResponse())
        assert metrics == {"input": 0, "output": 0, "total": 0}
