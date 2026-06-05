import sys
import types

from agent_tools_config import (
    clear_cloud_tool_catalog_cache,
    handle_tool_command,
    load_cloud_tool_catalog,
    load_enabled_cloud_tools,
    normalize_tool_slug,
)


def test_normalize_tool_slug_accepts_aliases():
    assert normalize_tool_slug("email") == "gmail"
    assert normalize_tool_slug("Google Calendar") == "googlecalendar"
    assert normalize_tool_slug("pull request") == "github"


def test_add_and_list_cloud_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUD_TOOLS_CONFIG_PATH", str(tmp_path / "agent_tools.json"))
    monkeypatch.setenv("CLOUD_TOOLKITS", "github")
    monkeypatch.delenv("CLOUD_TOOLS_API_KEY", raising=False)
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
    clear_cloud_tool_catalog_cache()

    response = handle_tool_command("/add-tool gmail")

    assert "Gmail (`gmail`)" in response
    assert load_enabled_cloud_tools() == ["github", "gmail"]

    tools_response = handle_tool_command("/tools")
    assert "Native tools:" in tools_response
    assert "GitHub (`github`)" in tools_response
    assert "Gmail (`gmail`)" in tools_response


def test_which_tool_recommends_matching_tool(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUD_TOOLS_CONFIG_PATH", str(tmp_path / "agent_tools.json"))
    monkeypatch.setenv("CLOUD_TOOLKITS", "github")
    monkeypatch.delenv("CLOUD_TOOLS_API_KEY", raising=False)
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
    clear_cloud_tool_catalog_cache()

    response = handle_tool_command("/which-tool check my unread emails")

    assert "Gmail (`gmail`)" in response
    assert "not enabled" in response


def test_cloud_tools_uses_remote_catalog(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUD_TOOLS_CONFIG_PATH", str(tmp_path / "agent_tools.json"))
    monkeypatch.setenv("CLOUD_TOOLS_API_KEY", "test-key")
    clear_cloud_tool_catalog_cache()

    class FakeMeta:
        description = "Manage Airtable bases and records."
        tools_count = 12
        triggers_count = 2

    class FakeToolkit:
        slug = "airtable"
        name = "Airtable"
        meta = FakeMeta()

    class FakeResponse:
        items = [FakeToolkit()]

    class FakeToolkits:
        def list(self, **kwargs):
            assert kwargs["limit"] == 1000
            assert kwargs["managed_by"] == "all"
            return FakeResponse()

    class FakeComposio:
        def __init__(self, api_key, timeout):
            assert api_key == "test-key"
            assert timeout == 12
            self.toolkits = FakeToolkits()

    fake_module = types.ModuleType("composio_client")
    fake_module.Composio = FakeComposio
    monkeypatch.setitem(sys.modules, "composio_client", fake_module)

    catalog, remote_available = load_cloud_tool_catalog(force_refresh=True)
    response = handle_tool_command("/cloud-tools")

    assert remote_available is True
    assert "airtable" in catalog
    assert "Airtable (`airtable`) [available] (12 tools, 2 triggers)" in response
    assert "cloud catalog" in response
