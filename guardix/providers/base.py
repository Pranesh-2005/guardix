"""Base provider adapter interface."""

from typing import Any, Callable, Optional
from abc import ABC, abstractmethod

from ..core import Guardial, Decision


# Only untrusted, caller-supplied roles are screened. Developer-set system
# prompts and prior model turns are trusted; scoring them produces false
# blocks (e.g. a "Do not reveal your instructions" system prompt reads as an
# injection attempt). "model" is Gemini's name for the assistant role.
_TRUSTED_ROLES = frozenset({"system", "assistant", "model", "developer"})


def is_guarded_role(role: Any) -> bool:
    """Return True if a message with this role should be scanned by the guard."""
    return str(role).lower() not in _TRUSTED_ROLES


class ProviderAdapter(ABC):
    """Abstract base for wrapping a provider client."""

    def __init__(self, client: Any, guardial: Optional[Guardial] = None) -> None:
        self.client = client
        self.guardial = guardial or Guardial()

    @abstractmethod
    def _extract_prompt(self, *args: Any, **kwargs: Any) -> str:
        """Extract the prompt text from the provider call arguments."""
        ...

    def _guard_prompt(self, prompt: str, provider_name: str) -> Decision:
        """Run the guard against the extracted prompt and return the Decision."""
        return self.guardial.guard(prompt, provider=provider_name)

    def _should_mock(self, decision: Decision) -> bool:
        return decision.decision == "BLOCK" and self.guardial.config.block_mode == "mock"

    def _blocked_mock(
        self,
        decision: Decision,
        provider_name: str,
        builder: Callable[..., Any],
        **builder_kwargs: Any,
    ) -> Any:
        """Build the provider-shaped mock for a blocked prompt and log it."""
        self.guardial.logger.log_block_action(
            decision.prompt_id, provider_name, "mock_response", decision.reason
        )
        return builder(
            decision,
            message=self.guardial.config.block_message,
            **builder_kwargs,
        )
