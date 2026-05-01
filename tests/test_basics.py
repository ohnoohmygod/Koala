import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from koala.tools.builtin.search import search, calculator
from koala.context.memory import ShortTermMemory


class TestShortTermMemory:
    def test_init_with_system_prompt(self):
        mem = ShortTermMemory("你是助手")
        assert len(mem.messages) == 1
        assert isinstance(mem.messages[0], SystemMessage)
        assert mem.messages[0].content == "你是助手"

    def test_init_without_system_prompt(self):
        mem = ShortTermMemory()
        assert len(mem.messages) == 0

    def test_add_user(self):
        mem = ShortTermMemory()
        mem.add_user("你好")
        assert len(mem.messages) == 1
        assert isinstance(mem.messages[0], HumanMessage)

    def test_add_ai(self):
        mem = ShortTermMemory()
        mem.add_ai("你好！")
        assert len(mem.messages) == 1
        assert isinstance(mem.messages[0], AIMessage)

    def test_add(self):
        mem = ShortTermMemory()
        msg = HumanMessage(content="test")
        mem.add(msg)
        assert len(mem.messages) == 1

    def test_extend(self):
        mem = ShortTermMemory()
        mem.extend([HumanMessage(content="a"), HumanMessage(content="b")])
        assert len(mem.messages) == 2

    def test_clear_keeps_system(self):
        mem = ShortTermMemory("system")
        mem.add_user("hello")
        mem.add_ai("hi")
        mem.clear()
        assert len(mem.messages) == 1
        assert isinstance(mem.messages[0], SystemMessage)

    def test_clear_no_system(self):
        mem = ShortTermMemory()
        mem.add_user("hello")
        mem.clear()
        assert len(mem.messages) == 0

    def test_token_count(self):
        mem = ShortTermMemory()
        mem.add_user("a" * 100)
        assert mem.token_count() == 25


class TestBuiltinTools:
    def test_search_tool(self):
        result = search.invoke({"query": "test"})
        assert "test" in result

    def test_calculator_tool(self):
        result = calculator.invoke({"expression": "2 + 3"})
        assert result == "5"

    def test_calculator_error(self):
        result = calculator.invoke({"expression": "1/0"})
        assert "错误" in result
