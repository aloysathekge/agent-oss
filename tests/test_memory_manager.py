# Unit tests for VectorMemoryManager.
#
# Covers: initialization, add, update, delete, search, keyword search,
# deduplication, persistence (save/reload), and clear.
# All embedding calls are mocked with deterministic fake vectors.

import os
import sys
import json
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import numpy as np
import faiss
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ==========================================
# DETERMINISTIC EMBEDDING HELPER
# ==========================================
def make_embedding(text: str, dim: int = 1536) -> list[float]:
    """Produce a deterministic unit vector from text."""
    rng = np.random.default_rng(seed=hash(text) % (2**32))
    vec = rng.standard_normal(dim).astype("float32")
    vec /= np.linalg.norm(vec)
    return vec.tolist()


# ==========================================
# FIXTURE
# ==========================================
@pytest.fixture
def memory_manager(tmp_path):
    """
    Create a VectorMemoryManager with mocked embeddings, isolated to tmp_path.
    """
    mem_root = str(tmp_path / "local_memory")
    os.makedirs(mem_root, exist_ok=True)

    # Mock environment
    env_patches = {
        "OPENAI_API_KEY": "sk-test-fake",
        "AGENT_ID": "test_agent",
        "USER_ID": "test_user",
        "LOCAL_MEMORY_ROOT": mem_root,
    }

    with patch.dict(os.environ, env_patches):
        # Mock the OpenAI embeddings client
        mock_embed = AsyncMock()
        mock_embed.aembed_query = AsyncMock(side_effect=lambda text: make_embedding(text))

        # Mock LLMs (not used in memory manager directly, but required at import)
        mock_llm = MagicMock()

        with patch("langchain_openai.OpenAIEmbeddings", return_value=mock_embed), \
             patch("langchain_openai.ChatOpenAI", return_value=mock_llm), \
             patch("tools.tool_manager.discover_skills", return_value={}):

            # Force reimport with our patches
            if "agent" in sys.modules:
                del sys.modules["agent"]
            if "tools.tool_manager" in sys.modules:
                del sys.modules["tools.tool_manager"]
            if "tools" in sys.modules:
                del sys.modules["tools"]

            import agent

            # Override the embed_client in the module with our mock
            agent.embed_client = mock_embed
            agent.LOCAL_MEMORY_ROOT = mem_root
            agent.LOCAL_AGENT_MEMORY_DIR = os.path.join(mem_root, "test_agent")

            # Create a fresh manager
            manager = agent.VectorMemoryManager("Semantic")
            # Override folder paths to use tmp
            manager.folder = os.path.join(mem_root, "test_agent", "semantic_memory")
            manager.index_file = os.path.join(manager.folder, "index.faiss")
            manager.store_file = os.path.join(manager.folder, "memories.json")
            os.makedirs(manager.folder, exist_ok=True)
            manager.memories = []
            manager.index = manager._new_index()
            manager._save_memories()

            yield manager


# ==========================================
# Tests
# ==========================================
class TestVectorMemoryManagerInit:
    """Initialization and storage layout tests."""

    def test_creates_storage_directory(self, memory_manager):
        assert os.path.isdir(memory_manager.folder)

    def test_empty_state_on_init(self, memory_manager):
        assert memory_manager.memories == []
        assert memory_manager.index.ntotal == 0

    def test_memories_json_exists(self, memory_manager):
        assert os.path.isfile(memory_manager.store_file)


class TestMemoryAdd:
    """Tests for adding memories."""

    @pytest.mark.asyncio
    async def test_add_single_memory(self, memory_manager):
        action = {"action": "ADD", "content": "User's name is Alice."}
        await memory_manager.execute_action(action)

        assert len(memory_manager.memories) == 1
        assert memory_manager.index.ntotal == 1
        assert memory_manager.memories[0]["content"] == "User's name is Alice."

    @pytest.mark.asyncio
    async def test_add_preserves_metadata(self, memory_manager):
        action = {"action": "ADD", "content": "User works at ACME Corp."}
        await memory_manager.execute_action(action)

        mem = memory_manager.memories[0]
        assert "id" in mem
        assert "created_at" in mem
        assert "updated_at" in mem
        assert "embedding" in mem
        assert mem["memory_type"] == "Semantic"
        assert len(mem["embedding"]) == 1536

    @pytest.mark.asyncio
    async def test_add_multiple_memories(self, memory_manager):
        contents = [
            "User's favorite color is blue.",
            "User has a dog named Rex.",
            "User lives in Berlin.",
        ]
        for c in contents:
            await memory_manager.execute_action({"action": "ADD", "content": c})

        assert len(memory_manager.memories) == 3
        assert memory_manager.index.ntotal == 3

    @pytest.mark.asyncio
    async def test_add_skips_empty_content(self, memory_manager):
        await memory_manager.execute_action({"action": "ADD", "content": ""})
        await memory_manager.execute_action({"action": "ADD", "content": "   "})

        assert len(memory_manager.memories) == 0

    @pytest.mark.asyncio
    async def test_add_deduplicates_near_identical(self, memory_manager):
        """Adding the exact same content twice should be deduplicated (cosine >= 0.95)."""
        content = "User's name is Alice."
        await memory_manager.execute_action({"action": "ADD", "content": content})
        await memory_manager.execute_action({"action": "ADD", "content": content})

        # Same text → same embedding → cosine 1.0 → should be skipped
        assert len(memory_manager.memories) == 1


class TestMemoryUpdate:
    """Tests for updating memories."""

    @pytest.mark.asyncio
    async def test_update_changes_content(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User's favorite color is blue."}
        )
        mem_id = memory_manager.memories[0]["id"]

        await memory_manager.execute_action(
            {"action": "UPDATE", "id": mem_id, "content": "User's favorite color is green."}
        )

        assert len(memory_manager.memories) == 1
        assert memory_manager.memories[0]["content"] == "User's favorite color is green."

    @pytest.mark.asyncio
    async def test_update_preserves_id(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User likes Python."}
        )
        mem_id = memory_manager.memories[0]["id"]

        await memory_manager.execute_action(
            {"action": "UPDATE", "id": mem_id, "content": "User loves Python and Rust."}
        )

        assert memory_manager.memories[0]["id"] == mem_id

    @pytest.mark.asyncio
    async def test_update_refreshes_embedding(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User likes cats."}
        )
        old_embedding = memory_manager.memories[0]["embedding"]

        mem_id = memory_manager.memories[0]["id"]
        await memory_manager.execute_action(
            {"action": "UPDATE", "id": mem_id, "content": "User likes dogs."}
        )

        new_embedding = memory_manager.memories[0]["embedding"]
        assert old_embedding != new_embedding

    @pytest.mark.asyncio
    async def test_update_nonexistent_id_is_noop(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User is tall."}
        )

        await memory_manager.execute_action(
            {"action": "UPDATE", "id": "00000000-0000-0000-0000-000000000000", "content": "Ghost."}
        )

        assert len(memory_manager.memories) == 1
        assert memory_manager.memories[0]["content"] == "User is tall."


class TestMemoryDelete:
    """Tests for deleting memories."""

    @pytest.mark.asyncio
    async def test_delete_removes_memory(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User has a cat."}
        )
        mem_id = memory_manager.memories[0]["id"]

        await memory_manager.execute_action({"action": "DELETE", "id": mem_id})

        assert len(memory_manager.memories) == 0
        assert memory_manager.index.ntotal == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_noop(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User has a cat."}
        )

        await memory_manager.execute_action(
            {"action": "DELETE", "id": "00000000-0000-0000-0000-000000000000"}
        )

        assert len(memory_manager.memories) == 1

    @pytest.mark.asyncio
    async def test_delete_one_of_many(self, memory_manager):
        for c in ["Fact A.", "Fact B.", "Fact C."]:
            await memory_manager.execute_action({"action": "ADD", "content": c})

        target_id = memory_manager.memories[1]["id"]  # Delete "Fact B"
        await memory_manager.execute_action({"action": "DELETE", "id": target_id})

        assert len(memory_manager.memories) == 2
        remaining = [m["content"] for m in memory_manager.memories]
        assert "Fact B." not in remaining
        assert "Fact A." in remaining
        assert "Fact C." in remaining


class TestMemorySearch:
    """Tests for vector similarity search."""

    @pytest.mark.asyncio
    async def test_search_finds_relevant_memory(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User's name is Alice."}
        )
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User works at a bakery."}
        )

        # Search for the exact content should return it (cosine with itself = 1.0)
        result = await memory_manager.search("User's name is Alice.", top_k=5, threshold=0.5)
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_search_empty_index_returns_empty(self, memory_manager):
        result = await memory_manager.search("anything", top_k=5, threshold=0.3)
        assert result == ""

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_empty(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User likes coffee."}
        )
        result = await memory_manager.search("", top_k=5, threshold=0.3)
        assert result == ""

    @pytest.mark.asyncio
    async def test_search_respects_threshold(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User's favorite programming language is Haskell."}
        )

        # With threshold=0.99, only near-exact matches should pass
        result = await memory_manager.search(
            "something completely unrelated like pizza recipes",
            top_k=5,
            threshold=0.99,
        )
        # Very unlikely to match at 0.99 threshold with random unrelated text
        # (deterministic embeddings from hash will be dissimilar)
        assert "Haskell" not in result

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, memory_manager):
        for i in range(10):
            await memory_manager.execute_action(
                {"action": "ADD", "content": f"Memory number {i} about topic alpha."}
            )

        result = await memory_manager.search("topic alpha", top_k=3, threshold=0.0)
        lines = [line for line in result.split("\n") if line.strip()]
        assert len(lines) <= 3


class TestKeywordSearch:
    """Tests for direct keyword matching."""

    @pytest.mark.asyncio
    async def test_keyword_finds_exact_match(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User bought a Tesla Model 3 in 2024."}
        )
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User likes hiking in the Alps."}
        )

        result = await memory_manager.keyword_search(["Tesla"], top_k=5)
        assert "Tesla" in result
        assert "Alps" not in result

    @pytest.mark.asyncio
    async def test_keyword_case_insensitive(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User visited TOKYO last summer."}
        )

        result = await memory_manager.keyword_search(["tokyo"], top_k=5)
        assert "TOKYO" in result

    @pytest.mark.asyncio
    async def test_keyword_skips_short_terms(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User is a software engineer."}
        )

        # Keywords shorter than 3 chars are ignored
        result = await memory_manager.keyword_search(["is", "a"], top_k=5)
        assert result == ""

    @pytest.mark.asyncio
    async def test_keyword_multiple_terms(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User has a German Shepherd named Bruno."}
        )
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User drives a BMW."}
        )

        result = await memory_manager.keyword_search(["Bruno", "BMW"], top_k=5)
        assert "Bruno" in result
        assert "BMW" in result

    @pytest.mark.asyncio
    async def test_keyword_empty_list(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "Something here."}
        )
        result = await memory_manager.keyword_search([], top_k=5)
        assert result == ""


class TestMemoryPersistence:
    """Tests for save/load round-tripping."""

    @pytest.mark.asyncio
    async def test_memories_persist_to_disk(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User's birthday is January 15."}
        )

        # Verify JSON file was written
        assert os.path.isfile(memory_manager.store_file)
        with open(memory_manager.store_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["content"] == "User's birthday is January 15."

    @pytest.mark.asyncio
    async def test_faiss_index_persists_to_disk(self, memory_manager):
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User speaks French."}
        )

        assert os.path.isfile(memory_manager.index_file)
        loaded_index = faiss.read_index(memory_manager.index_file)
        assert loaded_index.ntotal == 1

    @pytest.mark.asyncio
    async def test_reload_from_disk(self, memory_manager, tmp_path):
        """Simulate a restart by creating a new manager pointing to same folder."""
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User has two siblings."}
        )
        await memory_manager.execute_action(
            {"action": "ADD", "content": "User graduated from MIT."}
        )

        # Import agent module (should still be available from fixture)
        import agent

        # Create a new manager reading from the same folder
        new_manager = agent.VectorMemoryManager.__new__(agent.VectorMemoryManager)
        new_manager.memory_type = "Semantic"
        new_manager.dim = 1536
        new_manager.folder = memory_manager.folder
        new_manager.index_file = memory_manager.index_file
        new_manager.store_file = memory_manager.store_file
        new_manager.lock = asyncio.Lock()
        new_manager.memories = new_manager._load_memories()
        new_manager.index = new_manager._load_or_rebuild_index()

        assert len(new_manager.memories) == 2
        assert new_manager.index.ntotal == 2


class TestMemoryClear:
    """Tests for wiping all memories."""

    @pytest.mark.asyncio
    async def test_clear_removes_all(self, memory_manager):
        for c in ["Fact 1.", "Fact 2.", "Fact 3."]:
            await memory_manager.execute_action({"action": "ADD", "content": c})

        assert len(memory_manager.memories) == 3

        await memory_manager.clear()

        assert len(memory_manager.memories) == 0
        assert memory_manager.index.ntotal == 0
