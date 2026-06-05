import sys
import types


def test_missing_api_key_returns_configure_tool(monkeypatch, tmp_path):
    from tools.composio.client import clear_composio_session_cache, get_composio_tools

    monkeypatch.setenv("CLOUD_TOOLS_CONFIG_PATH", str(tmp_path / "agent_tools.json"))
    monkeypatch.delenv("CLOUD_TOOLS_API_KEY", raising=False)
    monkeypatch.setenv("COMPOSIO_API_KEY", "")
    clear_composio_session_cache()

    tools = get_composio_tools({"user_id": "test_user"})

    assert len(tools) == 1
    assert tools[0].name == "configure_cloud_tools"


def test_parse_toolkits_defaults_and_dedupes(monkeypatch, tmp_path):
    from tools.composio.client import DEFAULT_COMPOSIO_TOOLKITS, parse_toolkits

    monkeypatch.setenv("CLOUD_TOOLS_CONFIG_PATH", str(tmp_path / "agent_tools.json"))
    monkeypatch.delenv("CLOUD_TOOLKITS", raising=False)
    monkeypatch.setenv("COMPOSIO_TOOLKITS", "")
    assert parse_toolkits() == list(DEFAULT_COMPOSIO_TOOLKITS)

    assert parse_toolkits("github, gmail,github, linear ") == [
        "github",
        "gmail",
        "linear",
    ]


def test_factory_creates_user_scoped_session(monkeypatch, tmp_path):
    from tools.composio.client import clear_composio_session_cache, get_composio_tools

    captured = {}

    class FakeSession:
        def tools(self):
            return ["fake_tool"]

    class FakeComposio:
        def __init__(self, api_key, provider):
            captured["api_key"] = api_key
            captured["provider"] = provider

        def create(self, user_id, toolkits):
            captured["user_id"] = user_id
            captured["toolkits"] = toolkits
            return FakeSession()

    class FakeLangchainProvider:
        pass

    composio_module = types.ModuleType("composio")
    composio_module.Composio = FakeComposio
    composio_langchain_module = types.ModuleType("composio_langchain")
    composio_langchain_module.LangchainProvider = FakeLangchainProvider

    monkeypatch.setitem(sys.modules, "composio", composio_module)
    monkeypatch.setitem(sys.modules, "composio_langchain", composio_langchain_module)
    monkeypatch.setenv("CLOUD_TOOLS_CONFIG_PATH", str(tmp_path / "agent_tools.json"))
    monkeypatch.setenv("CLOUD_TOOLS_API_KEY", "test-key")
    monkeypatch.setenv("CLOUD_TOOLKITS", "github,gmail")
    clear_composio_session_cache()

    tools = get_composio_tools({"user_id": "user-123", "channel_type": "cli"})

    assert tools == ["fake_tool"]
    assert captured["api_key"] == "test-key"
    assert isinstance(captured["provider"], FakeLangchainProvider)
    assert captured["user_id"] == "user-123"
    assert captured["toolkits"] == ["github", "gmail"]


def test_factory_reuses_cached_session(monkeypatch, tmp_path):
    from tools.composio.client import clear_composio_session_cache, get_composio_tools

    calls = {"create": 0}

    class FakeSession:
        def tools(self):
            return ["cached_tool"]

    class FakeComposio:
        def __init__(self, api_key, provider):
            pass

        def create(self, user_id, toolkits):
            calls["create"] += 1
            return FakeSession()

    class FakeLangchainProvider:
        pass

    composio_module = types.ModuleType("composio")
    composio_module.Composio = FakeComposio
    composio_langchain_module = types.ModuleType("composio_langchain")
    composio_langchain_module.LangchainProvider = FakeLangchainProvider

    monkeypatch.setitem(sys.modules, "composio", composio_module)
    monkeypatch.setitem(sys.modules, "composio_langchain", composio_langchain_module)
    monkeypatch.setenv("CLOUD_TOOLS_CONFIG_PATH", str(tmp_path / "agent_tools.json"))
    monkeypatch.setenv("CLOUD_TOOLS_API_KEY", "test-key")
    monkeypatch.setenv("CLOUD_TOOLKITS", "github")
    clear_composio_session_cache()

    assert get_composio_tools({"user_id": "same-user"}) == ["cached_tool"]
    assert get_composio_tools({"user_id": "same-user"}) == ["cached_tool"]
    assert calls["create"] == 1
