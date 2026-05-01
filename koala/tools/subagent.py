import asyncio
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from koala.agent.agent import Agent
from koala.llm import LLMClient


class SubAgentTool(BaseTool):
    name: str = "sub_agent"
    description: str = "并发执行多个子任务。tasks: 任务列表。is_fork: 是否继承父Agent上下文。"

    llm: LLMClient
    tools: list[BaseTool] = []
    parent_messages: list[BaseMessage] | None = None

    def _run(self, tasks: list[str], is_fork: bool = False) -> str:
        raise NotImplementedError("请使用异步版本 arun")

    async def _arun(self, tasks: list[str], is_fork: bool = False) -> str:
        if not tasks:
            return "无任务"

        sub_prompt = "你是一个子任务执行者。完成指定任务后，用简洁的文字总结结果。"

        async def run_task(task: str) -> str:
            try:
                if is_fork and self.parent_messages:
                    agent = Agent(llm=self.llm, tools=self.tools, messages=self.parent_messages)
                else:
                    agent = Agent(llm=self.llm, tools=self.tools, system_prompt=sub_prompt)
                return await agent.arun(task)
            except Exception as e:
                return f"任务 '{task}' 执行错误: {e}"

        results = await asyncio.gather(*[run_task(t) for t in tasks])
        return "\n---\n".join(results)
