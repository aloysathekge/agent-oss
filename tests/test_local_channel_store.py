import asyncio


def test_chat_history_persists_and_windows(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "agent_one")

    from local_channel_store import append_chat_pair, get_recent_history_items

    for index in range(5):
        append_chat_pair("cli", f"user {index}", f"agent {index}")

    recent = get_recent_history_items("cli", limit=8)

    assert len(recent) == 8
    assert recent[0]["content"] == "user 1"
    assert recent[-1]["content"] == "agent 4"


def test_attachment_storage_extracts_text(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "agent_one")

    from local_channel_store import render_attachment_context, store_attachment_from_bytes

    record = asyncio.run(
        store_attachment_from_bytes(
            b"hello from the uploaded file",
            filename="note.txt",
            mime_type="text/plain",
            channel_type="telegram",
            conversation_id="chat-1",
        )
    )

    context = render_attachment_context([record["id"]])

    assert "note.txt" in context
    assert "text/plain" in context
    assert "hello from the uploaded file" in context


def test_attachment_note_is_added_to_history_messages(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "agent_one")

    from local_channel_store import append_attachment_note, store_attachment_from_bytes

    record = asyncio.run(
        store_attachment_from_bytes(
            b"attachment text",
            filename="brief.md",
            mime_type="text/markdown",
            channel_type="cli",
        )
    )
    content = append_attachment_note("summarize this", [record["id"]])

    assert "summarize this" in content
    assert "brief.md" in content
    assert record["id"] in content
