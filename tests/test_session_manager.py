import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from koala.context.memory import ShortTermMemory
from koala.session.manager import SessionManager


@pytest.fixture
def manager(tmp_path):
    return SessionManager(base_dir=str(tmp_path / "sessions"))


@pytest.mark.asyncio
async def test_create_session(manager):
    sid, store = await manager.create_session()
    meta = await store.get_meta()
    assert meta is not None
    assert meta.session_id == sid


@pytest.mark.asyncio
async def test_create_session_with_custom_id(manager):
    sid, store = await manager.create_session(session_id="my-session")
    assert sid == "my-session"
    meta = await store.get_meta()
    assert meta.session_id == "my-session"


@pytest.mark.asyncio
async def test_create_session_with_agent_config(manager):
    sid, _ = await manager.create_session(
        session_id="cfg-test",
        agent_config={"model": "gpt-4", "system_prompt": "Be helpful"},
    )
    _, store = await manager.get_session(sid)
    meta = await store.get_meta()
    assert meta.agent_config["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_list_sessions_empty(manager):
    sessions = await manager.list_sessions()
    assert sessions == []


@pytest.mark.asyncio
async def test_list_sessions(manager):
    await manager.create_session(session_id="s1")
    await manager.create_session(session_id="s2")
    await manager.create_session(session_id="s3")
    sessions = await manager.list_sessions()
    ids = [s.session_id for s in sessions]
    assert sorted(ids) == ["s1", "s2", "s3"]


@pytest.mark.asyncio
async def test_get_session(manager):
    sid, _ = await manager.create_session(session_id="find-me")
    result = await manager.get_session("find-me")
    assert result is not None
    meta, store = result
    assert meta.session_id == "find-me"


@pytest.mark.asyncio
async def test_get_session_not_found(manager):
    result = await manager.get_session("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_delete_session(manager):
    await manager.create_session(session_id="del-me")
    deleted = await manager.delete_session("del-me")
    assert deleted is True
    result = await manager.get_session("del-me")
    assert result is None


@pytest.mark.asyncio
async def test_delete_session_not_found(manager):
    deleted = await manager.delete_session("nope")
    assert deleted is False


@pytest.mark.asyncio
async def test_save_and_restore_memory(manager):
    sid, _ = await manager.create_session(session_id="round-trip")

    original = ShortTermMemory(system_prompt="You are a helper.")
    original.add_user("Hello")
    original.add_ai("Hi there!")

    await manager.save_memory(sid, original)

    restored = ShortTermMemory()
    await manager.restore_memory(sid, restored)
    assert len(restored.messages) == 3
    assert isinstance(restored.messages[0], SystemMessage)
    assert restored.messages[0].content == "You are a helper."
    assert isinstance(restored.messages[1], HumanMessage)
    assert restored.messages[1].content == "Hello"
    assert isinstance(restored.messages[2], AIMessage)
    assert restored.messages[2].content == "Hi there!"


@pytest.mark.asyncio
async def test_full_round_trip_with_tool_messages(manager):
    sid, _ = await manager.create_session(session_id="tools-test")

    memory = ShortTermMemory(system_prompt="You are a coder.")
    memory.add_user("Run 1+1")
    memory.add(AIMessage(
        content="",
        tool_calls=[{"id": "tc1", "name": "bash", "args": {"command": "echo 2"}}],
    ))
    memory.add(ToolMessage(content="2", tool_call_id="tc1"))
    memory.add_ai("The answer is 2.")

    await manager.save_memory(sid, memory)

    restored = ShortTermMemory()
    await manager.restore_memory(sid, restored)
    assert len(restored.messages) == 5

    ai_msg = restored.messages[2]
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.tool_calls[0]["name"] == "bash"

    tool_msg = restored.messages[3]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "tc1"
    assert tool_msg.content == "2"


@pytest.mark.asyncio
async def test_save_messages_incremental(manager):
    sid, _ = await manager.create_session(session_id="incr")

    await manager.save_messages(sid, [
        HumanMessage(content="msg1"),
        AIMessage(content="reply1"),
    ])
    await manager.save_messages(sid, [
        HumanMessage(content="msg2"),
    ])

    _, store = await manager.get_session(sid)
    messages = await store.load_messages()
    assert len(messages) == 3
    assert messages[0].content == "msg1"
    assert messages[2].content == "msg2"


@pytest.mark.asyncio
async def test_save_messages_session_not_found(manager):
    with pytest.raises(ValueError, match="not found"):
        await manager.save_messages("nonexistent", [HumanMessage(content="x")])


@pytest.mark.asyncio
async def test_save_memory_session_not_found(manager):
    with pytest.raises(ValueError, match="not found"):
        await manager.save_memory("nonexistent", ShortTermMemory())
