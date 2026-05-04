import asyncio
import json

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from koala.session.models import SessionMeta
from koala.session.store import SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(str(tmp_path / "test-session"))


@pytest.fixture
def meta():
    return SessionMeta(session_id="test-session")


@pytest.mark.asyncio
async def test_save_and_load_meta(store, meta):
    await store.save_meta(meta)
    loaded = await store.get_meta()
    assert loaded is not None
    assert loaded.session_id == "test-session"
    assert loaded.created_at == meta.created_at


@pytest.mark.asyncio
async def test_append_single_message(store, meta):
    await store.save_meta(meta)
    await store.append_message(HumanMessage(content="Hello"), meta)
    messages = await store.load_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hello"


@pytest.mark.asyncio
async def test_append_multiple_messages(store, meta):
    await store.save_meta(meta)
    msgs = [
        SystemMessage(content="You are a helper."),
        HumanMessage(content="Hi"),
        AIMessage(content="Hello!"),
    ]
    await store.append_messages(msgs, meta)
    loaded = await store.load_messages()
    assert len(loaded) == 3
    assert isinstance(loaded[0], SystemMessage)
    assert isinstance(loaded[1], HumanMessage)
    assert isinstance(loaded[2], AIMessage)
    assert loaded[1].content == "Hi"


@pytest.mark.asyncio
async def test_append_does_not_rewrite(store, meta):
    await store.save_meta(meta)
    await store.append_message(HumanMessage(content="first"), meta)
    await store.append_message(HumanMessage(content="second"), meta)

    with open(store._messages_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["message"]["data"]["content"] == "first"
    assert json.loads(lines[1])["message"]["data"]["content"] == "second"


@pytest.mark.asyncio
async def test_message_with_tool_calls(store, meta):
    await store.save_meta(meta)
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"id": "tc1", "name": "run_code", "args": {"code": "1+1"}}],
    )
    tool_msg = ToolMessage(content="2", tool_call_id="tc1")
    await store.append_messages([ai_msg, tool_msg], meta)

    loaded = await store.load_messages()
    assert len(loaded) == 2
    assert isinstance(loaded[0], AIMessage)
    assert loaded[0].tool_calls[0]["name"] == "run_code"
    assert isinstance(loaded[1], ToolMessage)
    assert loaded[1].tool_call_id == "tc1"


@pytest.mark.asyncio
async def test_load_empty(store):
    messages = await store.load_messages()
    assert messages == []


@pytest.mark.asyncio
async def test_message_count_increments(store, meta):
    await store.save_meta(meta)
    assert meta.message_count == 0
    await store.append_message(HumanMessage(content="a"), meta)
    assert meta.message_count == 1
    await store.append_messages([HumanMessage(content="b"), HumanMessage(content="c")], meta)
    assert meta.message_count == 3


@pytest.mark.asyncio
async def test_concurrent_appends(store, meta):
    await store.save_meta(meta)
    tasks = [
        store.append_message(HumanMessage(content=f"msg-{i}"), meta)
        for i in range(10)
    ]
    await asyncio.gather(*tasks)

    with open(store._messages_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 10
    for line in lines:
        json.loads(line)  # all valid JSON


@pytest.mark.asyncio
async def test_delete(store, meta):
    await store.save_meta(meta)
    await store.append_message(HumanMessage(content="bye"), meta)
    assert store._dir.exists()
    await store.delete()
    assert not store._dir.exists()
