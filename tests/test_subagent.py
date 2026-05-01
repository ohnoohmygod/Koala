import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from koala.llm import LLMClient


def _make_mock_llm():
    return MagicMock(spec=LLMClient)


class TestSubAgentToolSingleTask:
    @pytest.mark.asyncio
    async def test_single_task_returns_result(self):
        mock_llm = _make_mock_llm()

        with patch("koala.tools.subagent.Agent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.arun.return_value = "任务A的结果总结"
            MockAgent.return_value = mock_instance

            from koala.tools.subagent import SubAgentTool
            tool = SubAgentTool(llm=mock_llm, tools=[])

            result = await tool.ainvoke({"tasks": ["任务A"]})

            assert "任务A的结果总结" in result

    @pytest.mark.asyncio
    async def test_single_task_creates_agent_with_sub_prompt(self):
        mock_llm = _make_mock_llm()

        with patch("koala.tools.subagent.Agent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.arun.return_value = "结果"
            MockAgent.return_value = mock_instance

            from koala.tools.subagent import SubAgentTool
            tool = SubAgentTool(llm=mock_llm, tools=[])
            await tool.ainvoke({"tasks": ["做某事"]})

            # 验证创建了 Agent，且 system_prompt 包含子任务相关提示
            MockAgent.assert_called_once()
            call_kwargs = MockAgent.call_args[1]
            assert "子任务" in call_kwargs["system_prompt"] or "总结" in call_kwargs["system_prompt"]


class TestSubAgentToolMultipleTasks:
    @pytest.mark.asyncio
    async def test_multiple_tasks_returns_all_results(self):
        mock_llm = _make_mock_llm()

        with patch("koala.tools.subagent.Agent") as MockAgent:
            # 每次 new Agent 返回不同的 mock
            instances = [
                AsyncMock(arun=AsyncMock(return_value="结果A")),
                AsyncMock(arun=AsyncMock(return_value="结果B")),
                AsyncMock(arun=AsyncMock(return_value="结果C")),
            ]
            MockAgent.side_effect = instances

            from koala.tools.subagent import SubAgentTool
            tool = SubAgentTool(llm=mock_llm, tools=[])
            result = await tool.ainvoke({"tasks": ["任务A", "任务B", "任务C"]})

            assert "结果A" in result
            assert "结果B" in result
            assert "结果C" in result
            # 验证创建了 3 个 Agent
            assert MockAgent.call_count == 3


class TestSubAgentToolError:
    @pytest.mark.asyncio
    async def test_one_task_error_others_still_run(self):
        mock_llm = _make_mock_llm()

        with patch("koala.tools.subagent.Agent") as MockAgent:
            instances = [
                AsyncMock(arun=AsyncMock(side_effect=Exception("子任务崩溃"))),
                AsyncMock(arun=AsyncMock(return_value="结果B")),
            ]
            MockAgent.side_effect = instances

            from koala.tools.subagent import SubAgentTool
            tool = SubAgentTool(llm=mock_llm, tools=[])
            result = await tool.ainvoke({"tasks": ["任务A", "任务B"]})

            # 任务B 的结果仍然在
            assert "结果B" in result
            # 任务A 的错误也在
            assert "错误" in result or "崩溃" in result

    @pytest.mark.asyncio
    async def test_empty_tasks(self):
        mock_llm = _make_mock_llm()

        from koala.tools.subagent import SubAgentTool
        tool = SubAgentTool(llm=mock_llm, tools=[])
        result = await tool.ainvoke({"tasks": []})

        assert result == "" or "无任务" in result


class TestSubAgentToolForkMode:
    @pytest.mark.asyncio
    async def test_fork_mode_passes_parent_messages(self):
        mock_llm = _make_mock_llm()
        parent_messages = [
            SystemMessage(content="主助手"),
            HumanMessage(content="之前的问题"),
            AIMessage(content="之前的回答"),
        ]

        with patch("koala.tools.subagent.Agent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.arun.return_value = "Fork结果"
            MockAgent.return_value = mock_instance

            from koala.tools.subagent import SubAgentTool
            tool = SubAgentTool(llm=mock_llm, tools=[], parent_messages=parent_messages)
            await tool.ainvoke({"tasks": ["继续任务"], "is_fork": True})

            # 验证 Agent 创建时传入了 parent_messages
            call_kwargs = MockAgent.call_args[1]
            assert call_kwargs.get("messages") == parent_messages
