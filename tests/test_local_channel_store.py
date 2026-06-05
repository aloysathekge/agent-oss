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


def test_successful_ai_attachment_extract_clears_base_error(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "agent_one")

    import local_channel_store

    def fake_basic_extract(path, mime_type, max_chars):
        return {
            "extract_type": "pdf",
            "text": "",
            "error": "PDF text extraction requires pypdf.",
        }

    async def fake_ai_extract(record, path, max_chars):
        return {"text": "Extracted resume content", "ai_extract_type": "pdf_vision_ocr"}

    monkeypatch.setattr(local_channel_store, "basic_extract_attachment", fake_basic_extract)
    monkeypatch.setattr(local_channel_store, "enrich_attachment_if_supported", fake_ai_extract)

    record = asyncio.run(
        local_channel_store.store_attachment_from_bytes(
            b"%PDF fake",
            filename="resume.pdf",
            mime_type="application/pdf",
            channel_type="telegram",
        )
    )

    assert record["extract"]["text"] == "Extracted resume content"
    assert record["extract"]["ai_extract_type"] == "pdf_vision_ocr"
    assert "error" not in record["extract"]


def test_refresh_reprocesses_failed_attachment(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("AGENT_ID", "agent_one")

    import local_channel_store

    calls = {"count": 0}

    def fake_basic_extract(path, mime_type, max_chars):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "extract_type": "pdf",
                "text": "",
                "error": "PDF text extraction requires pypdf.",
            }
        return {"extract_type": "pdf", "text": "Reprocessed resume text", "truncated": False}

    async def fake_ai_extract(*args, **kwargs):
        return {}

    monkeypatch.setattr(local_channel_store, "basic_extract_attachment", fake_basic_extract)
    monkeypatch.setattr(local_channel_store, "enrich_attachment_if_supported", fake_ai_extract)

    record = asyncio.run(
        local_channel_store.store_attachment_from_bytes(
            b"%PDF fake",
            filename="resume.pdf",
            mime_type="application/pdf",
            channel_type="telegram",
        )
    )

    assert record["extract"]["error"]

    asyncio.run(local_channel_store.refresh_attachments_for_context([record["id"]]))
    refreshed = local_channel_store.get_attachment_record(record["id"])

    assert refreshed["extract"]["text"] == "Reprocessed resume text"
    assert refreshed["reprocessed_at"]
