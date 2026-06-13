"""Configuration for Gaudrial guard engine."""

from typing import Any, Callable, Dict, List, Optional

from .detectors.base import BaseDetector


class Config:
    """Guard configuration."""

    DEFAULT_POLICIES: Dict[str, Dict[str, Any]] = {
        "permissive": {"threshold": 0.9, "fail_mode": "open", "log_level": "INFO"},
        "standard": {"threshold": 0.7, "fail_mode": "open", "log_level": "INFO"},
        "strict": {"threshold": 0.5, "fail_mode": "closed", "log_level": "DEBUG"},
    }

    def __init__(
        self,
        policy: str = "standard",
        threshold: Optional[float] = None,
        fail_mode: Optional[str] = None,
        log_level: Optional[str] = None,
        log_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        log_file: Optional[str] = "logs/promptgaurd.jsonl",
        custom_detectors: Optional[List[BaseDetector]] = None,
        mask_raw_prompt: bool = True,
        block_mode: str = "mock",
        block_message: Optional[str] = None,
    ) -> None:
        defaults = self.DEFAULT_POLICIES.get(policy, self.DEFAULT_POLICIES["standard"])
        self.policy = policy
        self.threshold = threshold if threshold is not None else defaults["threshold"]
        self.fail_mode = fail_mode if fail_mode is not None else defaults["fail_mode"]
        self.log_level = log_level if log_level is not None else defaults["log_level"]
        self.log_sink = log_sink
        # When set, structured logs go to this file (folder auto-created)
        # and console output is suppressed.
        self.log_file = log_file
        self.custom_detectors = custom_detectors or []
        self.mask_raw_prompt = mask_raw_prompt
        # "mock": blocked calls return a provider-shaped mock response so the
        # pipeline never breaks. "raise": blocked calls raise GuardBlocked.
        self.block_mode = block_mode
        self.block_message = block_message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "threshold": self.threshold,
            "fail_mode": self.fail_mode,
            "log_level": self.log_level,
            "mask_raw_prompt": self.mask_raw_prompt,
            "block_mode": self.block_mode,
        }
