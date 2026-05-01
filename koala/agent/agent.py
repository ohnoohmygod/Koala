from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from koala.llm import LLMClient
from koala.context.memory import ShortTermMemory


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: list[BaseTool] | None = None,
        system_prompt: str = "你是一个有用的AI助手。",
        messages: list | None = None,
    ):
        self.llm = llm
        self.tools = tools or []
        if messages is not None:
            self.memory = ShortTermMemory()
            self.memory.extend(messages)
        else:
            self.memory = ShortTermMemory(system_prompt)

    def _find_tool(self, name: str) -> BaseTool | None:
        return next((t for t in self.tools if t.name == name), None)

    async def arun(self, user_input: str) -> str:
        self.memory.add_user(user_input)

        while True:
            response = await self.llm.ainvoke(self.memory.messages, tools=self.tools or None)
            self.memory.add(response)

            if not response.tool_calls:
                return response.content

            for tc in response.tool_calls:
                result = await self._execute_tool(tc)
                self.memory.add(ToolMessage(content=result, tool_call_id=tc["id"]))

    def run(self, user_input: str) -> str:
        self.memory.add_user(user_input)

        while True:
            response = self.llm.invoke(self.memory.messages, tools=self.tools or None)
            self.memory.add(response)

            if not response.tool_calls:
                return response.content

            for tc in response.tool_calls:
                result = self._execute_tool_sync(tc)
                self.memory.add(ToolMessage(content=result, tool_call_id=tc["id"]))

    async def _execute_tool(self, tc: dict) -> str:
        tool = self._find_tool(tc["name"])
        if tool is None:
            return f"未知工具: {tc['name']}"
        try:
            result = await tool.ainvoke(tc["args"])
            return str(result)
        except Exception as e:
            return f"工具执行错误: {e}"

    def _execute_tool_sync(self, tc: dict) -> str:
        tool = self._find_tool(tc["name"])
        if tool is None:
            return f"未知工具: {tc['name']}"
        try:
            result = tool.invoke(tc["args"])
            return str(result)
        except Exception as e:
            return f"工具执行错误: {e}"
