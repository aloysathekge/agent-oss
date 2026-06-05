def test_telegram_download_limit_message_is_user_facing(monkeypatch):
    import main

    monkeypatch.setattr(main, "CHANNEL_FILE_MAX_BYTES", 20_000_000)
    message = main.telegram_download_limit_message(25_000_000)

    assert "25.0 MB" in message
    assert "20.0 MB" in message
    assert "CHANNEL_FILE_MAX_BYTES" not in message
    assert "Please send a smaller file" in message


def test_attachment_failure_message_lists_files():
    import main

    message = main.format_attachment_failure_message(
        [
            {
                "filename": "large-video.mp4",
                "message": "That file is above this agent's Telegram download limit of 20.0 MB.",
            }
        ]
    )

    assert "large-video.mp4" in message
    assert "20.0 MB" in message
