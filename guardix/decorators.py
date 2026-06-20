"""Decorators for easy guarding of LLM call functions."""

from functools import wraps
from typing import Any, Callable, Optional

from .core import Guardial, Decision
from .exceptions import GuardBlocked
from .providers.base import is_guarded_role
from .responses import (
    anthropic_blocked_response,
    gemini_blocked_response,
    openai_blocked_response,
)

_RESPONSE_BUILDERS = {
    "openai": openai_blocked_response,
    "anthropic": anthropic_blocked_response,
    "gemini": gemini_blocked_response,
}


def guardial_guard(
    policy: Optional[str] = None,
    threshold: Optional[float] = None,
    fail_mode: Optional[str] = None,
    on_block: Optional[Callable[[Decision], Any]] = None,
    provider: str = "unknown",
    block_mode: str = "mock",
    response_format: str = "openai",
    block_message: Optional[str] = None,
) -> Callable[..., Any]:
    """Decorator that guards a function taking a prompt or messages argument.

    By default a blocked prompt does NOT raise: the wrapped function is
    skipped and a provider-shaped mock response (``response_format``:
    "openai", "anthropic", or "gemini") is returned, so the pipeline keeps
    flowing. Pass ``block_mode="raise"`` for the old GuardBlocked behavior.

    Usage:
        @guardial_guard(policy="strict")
        def chat(messages):
            return openai_client.chat.completions.create(model="gpt-4", messages=messages)
    """
    g = Guardial(
        policy=policy,
        threshold=threshold,
        fail_mode=fail_mode,
        block_mode=block_mode,
        block_message=block_message,
    )
    builder = _RESPONSE_BUILDERS.get(response_format, openai_blocked_response)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            messages = kwargs.get("messages")
            if messages is None and args:
                first = args[0]
                if isinstance(first, str):
                    messages = [{"role": "user", "content": first}]
                elif isinstance(first, list):
                    messages = first
            prompt = _messages_to_prompt(messages)
            decision = g.guard(prompt, provider=provider, on_block=on_block)
            if decision.decision == "BLOCK" and on_block is None and block_mode == "mock":
                g.logger.log_block_action(
                    decision.prompt_id, provider, "mock_response", decision.reason
                )
                if response_format == "gemini":
                    return builder(decision, message=block_message)
                return builder(decision, model=kwargs.get("model", "unknown"), message=block_message)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def guardial_audit(
    policy: Optional[str] = None,
    provider: str = "unknown",
    log_sink: Optional[Callable[[Any], None]] = None,
) -> Callable[..., Any]:
    """Decorator that only audits (never blocks) LLM calls.

    Usage:
        @guardial_audit(policy="standard")
        def chat(messages):
            return openai_client.chat.completions.create(model="gpt-4", messages=messages)
    """
    g = Guardial(policy=policy, fail_mode="open", log_sink=log_sink)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            messages = kwargs.get("messages")
            if messages is None and args:
                first = args[0]
                if isinstance(first, str):
                    messages = [{"role": "user", "content": first}]
                elif isinstance(first, list):
                    messages = first
            prompt = _messages_to_prompt(messages)
            g.analyze(prompt, provider=provider)  # audit only, never blocks
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _messages_to_prompt(messages: Any) -> str:
    if messages is None:
        return ""
    if isinstance(messages, str):
        return messages
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
