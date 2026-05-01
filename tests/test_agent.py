import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, ToolMessage

from koala.agent.agent import Agent
from koala.tools.builtin.search import search, calculator
from koala.llm import LLMClient


def _make_agent() -> tuple[Agent, MagicMock]:
    mock_llm = MagicMock(spec=LLMClient)
    agent = Agent(llm=mock_llm, tools=[search, calculator], system_prompt="test")
    return agent, mock_llm


class TestAgentRun:
    def test_direct_reply(self):
        agent, mock_llm = _make_agent()
        mock_llm.invoke.return_value = AIMessage(content="你好！")

        result = agent.run("你好")

        assert result == "你好！"
        assert len(agent.memory.messages) == 3  # system + user + ai
        mock_llm.invoke.assert_called_once()

    def test_tool_call_then_reply(self):
        agent, mock_llm = _make_agent()
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "calculator", "args": {"expression": "2+3"}, "id": "tc1"}],
        )
        final_msg = AIMessage(content="2+3=5")
        mock_llm.invoke.side_effect = [tool_call_msg, final_msg]

        result = agent.run("2+3等于几")

        assert result == "2+3=5"
        assert len(agent.memory.messages) == 5  # system + user + ai(tool_call) + tool_result + ai(final)
        assert mock_llm.invoke.call_count == 2

    def test_dynamic_add_remove_tool(self):
        agent, _ = _make_agent()
        assert agent._find_tool("search") is not None
        assert agent._find_tool("calculator") is not None

        agent.tools = [t for t in agent.tools if t.name != "calculator"]
        assert agent._find_tool("calculator") is None
        assert agent._find_tool("search") is not None


class TestAgentAsyncRun:
    @pytest.mark.asyncio
    async def test_direct_reply_async(self):
        agent, mock_llm = _make_agent()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="你好！"))

        result = await agent.arun("你好")

        assert result == "你好！"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        agent, mock_llm = _make_agent()
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "nonexistent_tool", "args": {}, "id": "tc1"}],
        )
        final_msg = AIMessage(content="抱歉，工具不可用")
        mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_msg, final_msg])

        result = await agent.arun("测试")

        assert result == "抱歉，工具不可用"
        tool_msgs = [m for m in agent.memory.messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "未知工具" in tool_msgs[0].content
