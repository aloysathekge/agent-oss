from agent_tools_config import (
    handle_tool_command,
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

    response = handle_tool_command("/which-tool check my unread emails")

    assert "Gmail (`gmail`)" in response
    assert "not enabled" in response
