"""Anthropic provider adapter."""

from typing import Any, Optional
from .base import ProviderAdapter, is_guarded_role
from ..core import Guardial
from ..responses import anthropic_blocked_response


class AnthropicAdapter(ProviderAdapter):
    """Wraps an Anthropic client to guard prompts."""

    def __init__(self, client: Any, guardial: Optional[Guardial] = None) -> None:
        super().__init__(client, guardial)
        self._messages = _GuardedMessages(self)

    @property
    def messages(self) -> "_GuardedMessages":
        return self._messages

    def _extract_prompt(self, *args: Any, **kwargs: Any) -> str:
        messages = kwargs.get("messages", [])
        # The Anthropic ``system`` kwarg is a trusted developer prompt, not
        # untrusted input — it is deliberately not scanned.
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                if not is_guarded_role(msg.get("role", "user")):
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Anthropic content blocks: [{"type": "text", "text": ...}, ...]
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                else:
                    parts.append(content)
            else:
                parts.append(str(msg))
        return "\n".join(parts)


class _GuardedMessages:
    """Proxy for messages.create with guard injection."""

    def __init__(self, adapter: AnthropicAdapter) -> None:
        self.adapter = adapter

    def create(self, *args: Any, **kwargs: Any) -> Any:
        prompt = self.adapter._extract_prompt(*args, **kwargs)
        decision = self.adapter._guard_prompt(prompt, provider_name="anthropic")
        if self.adapter._should_mock(decision):
            return self.adapter._blocked_mock(
                decision, "anthropic",
                anthropic_blocked_response, model=kwargs.get("model", "unknown"),
            )
        return self.adapter.client.messages.create(*args, **kwargs)

    async def create_async(self, *args: Any, **kwargs: Any) -> Any:
        prompt = self.adapter._extract_prompt(*args, **kwargs)
        decision = self.adapter._guard_prompt(prompt, provider_name="anthropic")
        if self.adapter._should_mock(decision):
            return self.adapter._blocked_mock(
                decision, "anthropic",
                anthropic_blocked_response, model=kwargs.get("model", "unknown"),
            )
        return await self.adapter.client.messages.create(*args, **kwargs)
