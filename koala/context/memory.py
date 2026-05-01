from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage


class ShortTermMemory:
    def __init__(self, system_prompt: str | None = None):
        self._messages: list[BaseMessage] = []
        if system_prompt:
            self._messages.append(SystemMessage(content=system_prompt))

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def add_user(self, content: str):
        self._messages.append(HumanMessage(content=content))

    def add_ai(self, content: str):
        self._messages.append(AIMessage(content=content))

    def add(self, message: BaseMessage):
        self._messages.append(message)

    def extend(self, messages: list[BaseMessage]):
        self._messages.extend(messages)

    def clear(self):
        system = [m for m in self._messages if isinstance(m, SystemMessage)]
        self._messages = system

    def token_count(self) -> int:
        """粗略估算 token 数，约 4 字符 ≈ 1 token。"""
        total = sum(len(m.content) for m in self._messages if m.content)
        return total // 4
