import pytest
from langchain_core.messages import HumanMessage
from koala.llm import LLMClient


@pytest.mark.parametrize("provider", ["glm", "deepseek"])
@pytest.mark.asyncio
async def test_llm_connection(provider):
    llm = LLMClient(provider)
    response = await llm.ainvoke([HumanMessage(content="hi")], tools=None)
    assert response.content is not None
    assert len(response.content) > 0
    print(f"\n{provider}: {response.content[:100]}")
