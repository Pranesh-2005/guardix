"""guardix — Universal LLM prompt guard against injection attacks."""

from typing import Any, Optional

from .core import Guardial, Policy, Decision
from .exceptions import GuardBlocked, GuardError
from .config import Config
from .responses import is_blocked_response

__version__ = "1.0.0"
__all__ = [
    "Guardial",
    "Policy",
    "Decision",
    "GuardBlocked",
    "GuardError",
    "Config",
    "guard_client",
    "is_blocked_response",
]


def guard_client(client: Any, guardial: Optional[Guardial] = None, provider: Optional[str] = None) -> Any:
    """Wrap any supported LLM client with prompt guarding in one line.

    Auto-detects the client type:
    - ``messages.create``            -> Anthropic
    - ``models.generate_content``    -> Gemini (google-genai)
    - ``chat.completions.create``    -> OpenAI and all OpenAI-compatible
      providers (Azure OpenAI, Groq, OpenRouter, Together, ...)

    ``provider`` overrides the name used in logs (e.g. "groq", "openrouter").

    Usage:
        from guardix import guard_client
        client = guard_client(OpenAI())
        client.chat.completions.create(...)  # guarded, never raises on block
    """
    from .providers import AnthropicAdapter, GeminiAdapter, OpenAIAdapter

    messages = getattr(client, "messages", None)
    if messages is not None and callable(getattr(messages, "create", None)):
        return AnthropicAdapter(client, guardial=guardial)

    models = getattr(client, "models", None)
    if models is not None and callable(getattr(models, "generate_content", None)):
        return GeminiAdapter(client, guardial=guardial)

    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None) if chat is not None else None
    if completions is not None and callable(getattr(completions, "create", None)):
        return OpenAIAdapter(client, guardial=guardial, provider_name=provider or "openai")

    raise TypeError(
        "Unsupported client: expected an object with messages.create (Anthropic), "
        "models.generate_content (Gemini), or chat.completions.create (OpenAI-compatible)."
    )
