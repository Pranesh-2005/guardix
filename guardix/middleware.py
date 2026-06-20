"""Middleware / interceptor pattern for provider-agnostic guarding."""

from typing import Any, Callable, Dict, Optional
from functools import wraps

from .core import Guardial, Decision
from .exceptions import GuardBlocked
from .responses import openai_blocked_response
from .providers.base import is_guarded_role


class LLMInterceptor:
    """Interceptor that wraps any provider client and guards every call."""

    def __init__(
        self,
        provider_client: Any,
        guardial: Optional[Guardial] = None,
        provider_name: str = "unknown",
    ) -> None:
        self.client = provider_client
        self.guardial = guardial or Guardial()
        self.provider_name = provider_name
        self._intercepted = False
        self._original_methods: Dict[str, Any] = {}

    def intercept(self) -> "LLMInterceptor":
        """Dynamically intercept the provider's chat.completions.create method."""
        if self._intercepted:
            return self
        try:
            chat = getattr(self.client, "chat", None)
            if chat is None:
                return self
            completions = getattr(chat, "completions", None)
            if completions is None:
                return self
            original = getattr(completions, "create", None)
            if original is None:
                return self
            self._original_methods["create"] = original
            completions.create = self._make_guarded_create(original)
            self._intercepted = True
        except Exception:
            pass
        return self

    def restore(self) -> "LLMInterceptor":
        """Restore the original provider method."""
        if not self._intercepted:
            return self
        try:
            chat = getattr(self.client, "chat", None)
            if chat is None:
                return self
            completions = getattr(chat, "completions", None)
            if completions is None:
                return self
            if "create" in self._original_methods:
                completions.create = self._original_methods["create"]
            self._intercepted = False
        except Exception:
            pass
        return self

    def _make_guarded_create(self, original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            prompt = self._extract_prompt(*args, **kwargs)
            decision = self.guardial.guard(prompt, provider=self.provider_name)
            if decision.decision == "BLOCK" and self.guardial.config.block_mode == "mock":
                self.guardial.logger.log_block_action(
                    decision.prompt_id, self.provider_name, "mock_response", decision.reason
                )
                return openai_blocked_response(
                    decision,
                    model=kwargs.get("model", "unknown"),
                    message=self.guardial.config.block_message,
                )
            return original(*args, **kwargs)

        return guarded

    def _extract_prompt(self, *args: Any, **kwargs: Any) -> str:
        messages = kwargs.get("messages", args[0] if args else [])
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                if not is_guarded_role(msg.get("role", "user")):
                    continue
                parts.append(msg.get("content", ""))
            else:
                parts.append(str(msg))
        return "\n".join(parts)

    def __enter__(self) -> "LLMInterceptor":
        self.intercept()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.restore()
