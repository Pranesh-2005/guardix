"""Google Gemini provider adapter (google-genai SDK).

Wraps ``client.models.generate_content(model=..., contents=...)``. On a
blocked prompt it returns a GenerateContentResponse-shaped mock whose
``.text`` carries the blocked message, so pipelines reading
``response.text`` keep working.

Gemini is also reachable through its OpenAI-compatible endpoint; in that
case use ``OpenAIAdapter(client, provider_name="gemini")`` instead.
"""

from typing import Any, Optional
from .base import ProviderAdapter, is_guarded_role
from ..core import Guardial
from ..responses import gemini_blocked_response


class GeminiAdapter(ProviderAdapter):
    """Wraps a google-genai client to guard prompts."""

    def __init__(self, client: Any, guardial: Optional[Guardial] = None) -> None:
        super().__init__(client, guardial)
        self._models = _GuardedModels(self)

    @property
    def models(self) -> "_GuardedModels":
        return self._models

    def _extract_prompt(self, *args: Any, **kwargs: Any) -> str:
        contents = kwargs.get("contents", args[0] if args else "")
        return _contents_to_text(contents)


def _contents_to_text(contents: Any) -> str:
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        # A turn with a trusted role (e.g. prior "model" output) is not scanned.
        if "role" in contents and not is_guarded_role(contents["role"]):
            return ""
        parts = contents.get("parts", [])
        return "\n".join(_contents_to_text(p) for p in parts)
    if isinstance(contents, (list, tuple)):
        return "\n".join(_contents_to_text(c) for c in contents)
    # google-genai types (Content/Part objects) expose .role / .text / .parts
    role = getattr(contents, "role", None)
    if role is not None and not is_guarded_role(role):
        return ""
    text = getattr(contents, "text", None)
    if isinstance(text, str):
        return text
    parts = getattr(contents, "parts", None)
    if parts is not None:
        return "\n".join(_contents_to_text(p) for p in parts)
    return str(contents)


class _GuardedModels:
    """Proxy for client.models with guard injection on generate_content."""

    def __init__(self, adapter: GeminiAdapter) -> None:
        self.adapter = adapter

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        prompt = self.adapter._extract_prompt(*args, **kwargs)
        decision = self.adapter._guard_prompt(prompt, provider_name="gemini")
        if self.adapter._should_mock(decision):
            return self.adapter._blocked_mock(decision, "gemini", gemini_blocked_response)
        return self.adapter.client.models.generate_content(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Pass everything else (count_tokens, embed_content, ...) straight through.
        return getattr(self.adapter.client.models, name)
