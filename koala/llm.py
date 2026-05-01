from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import os

load_dotenv()

_PROVIDERS = {
    "glm": {
        "api_key_env": "GLM_API_KEY",
        "base_url_env": "GLM_BASE_URL",
        "model_env": "GLM_MODEL",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "model_env": "DEEPSEEK_MODEL",
    },
}


class LLMClient:
    def __init__(self, provider: str = "glm", **overrides):
        self._provider = provider
        self._overrides = overrides
        self._llm = self._create_llm(provider, overrides)

    @staticmethod
    def _create_llm(provider: str, overrides: dict) -> ChatOpenAI:
        if provider not in _PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(_PROVIDERS.keys())}")
        cfg = _PROVIDERS[provider]
        kwargs = {
            "api_key": os.getenv(cfg["api_key_env"]),
            "base_url": os.getenv(cfg["base_url_env"]),
            "model": os.getenv(cfg["model_env"]),
        }
        kwargs.update(overrides)
        return ChatOpenAI(**kwargs)

    @property
    def provider(self) -> str:
        return self._provider

    def switch_provider(self, provider: str, **overrides):
        self._provider = provider
        self._overrides = overrides
        self._llm = self._create_llm(provider, overrides)

    async def ainvoke(self, messages: list[BaseMessage], tools: list[BaseTool] | None = None):
        llm = self._llm.bind_tools(tools) if tools else self._llm
        return await llm.ainvoke(messages)

    def invoke(self, messages: list[BaseMessage], tools: list[BaseTool] | None = None):
        llm = self._llm.bind_tools(tools) if tools else self._llm
        return llm.invoke(messages)
