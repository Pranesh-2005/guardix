"""OpenAI provider adapter.

Also works for any OpenAI-compatible provider: Azure OpenAI, Groq,
OpenRouter, Together, Gemini's OpenAI-compat endpoint, etc. Pass
``provider_name`` so logs identify the actual provider:

    OpenAIAdapter(groq_client, provider_name="groq")
"""

from typing import Any, Optional
from .base import ProviderAdapter, is_guarded_role
from ..core import Guardial
from ..responses import openai_blocked_response


class OpenAIAdapter(ProviderAdapter):
    """Wraps an OpenAI(-compatible) client to guard prompts."""

    def __init__(
        self,
        client: Any,
        guardial: Optional[Guardial] = None,
        provider_name: str = "openai",
    ) -> None:
        super().__init__(client, guardial)
        self.provider_name = provider_name
        self._chat = _GuardedChat(self)

    @property
    def chat(self) -> "_GuardedChat":
        return self._chat

    def _extract_prompt(self, *args: Any, **kwargs: Any) -> str:
        messages = kwargs.get("messages", args[0] if args else [])
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                if not is_guarded_role(msg.get("role", "user")):
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                else:
                    parts.append(content)
            else:
                parts.append(str(msg))
        return "\n".join(parts)


class _GuardedChat:
    """Proxy for chat.completions with guard injection."""

    def __init__(self, adapter: OpenAIAdapter) -> None:
        self.adapter = adapter
        self._completions = _GuardedChatCompletions(self.adapter)

    @property
    def completions(self) -> "_GuardedChatCompletions":
        return self._completions


class _GuardedChatCompletions:
    """Proxy for chat.completions.create with guard injection."""

    def __init__(self, adapter: OpenAIAdapter) -> None:
        self.adapter = adapter

    def create(self, *args: Any, **kwargs: Any) -> Any:
        prompt = self.adapter._extract_prompt(*args, **kwargs)
        decision = self.adapter._guard_prompt(prompt, provider_name=self.adapter.provider_name)
        if self.adapter._should_mock(decision):
            return self.adapter._blocked_mock(
                decision, self.adapter.provider_name,
                openai_blocked_response, model=kwargs.get("model", "unknown"),
            )
        return self.adapter.client.chat.completions.create(*args, **kwargs)

    # Some SDKs expose acreate for async
    async def acreate(self, *args: Any, **kwargs: Any) -> Any:
        prompt = self.adapter._extract_prompt(*args, **kwargs)
        decision = self.adapter._guard_prompt(prompt, provider_name=self.adapter.provider_name)
        if self.adapter._should_mock(decision):
            return self.adapter._blocked_mock(
                decision, self.adapter.provider_name,
                openai_blocked_response, model=kwargs.get("model", "unknown"),
            )
        return await self.adapter.client.chat.completions.acreate(*args, **kwargs)
