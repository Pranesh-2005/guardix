"""Generic provider adapter for any client with chat.completions.create."""

from typing import Any, Optional
from .base import ProviderAdapter, is_guarded_role
from ..core import Guardial
from ..responses import openai_blocked_response


class GenericAdapter(ProviderAdapter):
    """Wraps any client that has a chat.completions.create method."""

    def __init__(self, client: Any, guardial: Optional[Guardial] = None) -> None:
        super().__init__(client, guardial)
        self._chat = _GuardedGenericChat(self)

    @property
    def chat(self) -> "_GuardedGenericChat":
        return self._chat

    def _provider_name(self) -> str:
        return getattr(self.client, "__class__", object).__name__.lower()

    def _extract_prompt(self, *args: Any, **kwargs: Any) -> str:
        messages = kwargs.get("messages", args[0] if args else [])
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                if not is_guarded_role(msg.get("role", "user")):
                    continue
                content = msg.get("content", "")
                parts.append(content)
            else:
                parts.append(str(msg))
        return "\n".join(parts)


class _GuardedGenericChat:
    """Proxy for chat.completions.create."""

    def __init__(self, adapter: GenericAdapter) -> None:
        self.adapter = adapter
        self._completions = _GuardedGenericCompletions(self.adapter)

    @property
    def completions(self) -> "_GuardedGenericCompletions":
        return self._completions


class _GuardedGenericCompletions:
    """Proxy for completions.create."""

    def __init__(self, adapter: GenericAdapter) -> None:
        self.adapter = adapter

    def create(self, *args: Any, **kwargs: Any) -> Any:
        prompt = self.adapter._extract_prompt(*args, **kwargs)
        provider_name = self.adapter._provider_name()
        decision = self.adapter._guard_prompt(prompt, provider_name=provider_name)
        if self.adapter._should_mock(decision):
            return self.adapter._blocked_mock(
                decision, provider_name,
                openai_blocked_response, model=kwargs.get("model", "unknown"),
            )
        return self.adapter.client.chat.completions.create(*args, **kwargs)

    async def acreate(self, *args: Any, **kwargs: Any) -> Any:
        prompt = self.adapter._extract_prompt(*args, **kwargs)
        provider_name = self.adapter._provider_name()
        decision = self.adapter._guard_prompt(prompt, provider_name=provider_name)
        if self.adapter._should_mock(decision):
            return self.adapter._blocked_mock(
                decision, provider_name,
                openai_blocked_response, model=kwargs.get("model", "unknown"),
            )
        return await self.adapter.client.chat.completions.acreate(*args, **kwargs)
