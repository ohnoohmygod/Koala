from koala.session.models import SessionMeta


def test_default_timestamps():
    meta = SessionMeta(session_id="abc")
    assert meta.created_at != ""
    assert meta.updated_at != ""
    assert meta.agent_config == {}
    assert meta.message_count == 0


def test_touch_updates_updated_at():
    meta = SessionMeta(session_id="abc")
    old_updated = meta.updated_at
    meta.touch()
    assert meta.updated_at >= old_updated
    assert meta.created_at == meta.created_at  # unchanged


def test_to_dict_from_dict_roundtrip():
    original = SessionMeta(session_id="test", agent_config={"model": "gpt-4"})
    d = original.to_dict()
    restored = SessionMeta.from_dict(d)
    assert restored.session_id == original.session_id
    assert restored.created_at == original.created_at
    assert restored.agent_config == original.agent_config
    assert restored.message_count == original.message_count


def test_custom_timestamps_preserved():
    meta = SessionMeta(
        session_id="x",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )
    assert meta.created_at == "2026-01-01T00:00:00"
    assert meta.updated_at == "2026-01-01T00:00:00"
