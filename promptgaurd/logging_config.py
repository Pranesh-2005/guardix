"""Structured logging with decision trails for promptgaurd."""

import itertools
import logging
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from dataclasses import asdict, is_dataclass


class DecisionEncoder(json.JSONEncoder):
    """Encode dataclasses and exceptions for JSON serialization."""

    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Exception):
            return str(obj)
        return super().default(obj)


class StructuredLogger:
    """Logger that emits structured decision trail entries.

    By default entries go to the console. Pass ``log_file`` to write JSON
    lines to a file instead (the parent folder is created automatically and
    console output is suppressed, keeping user-facing output clean).
    """

    _ids = itertools.count()

    def __init__(
        self,
        level: str = "INFO",
        sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        log_file: Optional[str] = None,
        console: Optional[bool] = None,
    ) -> None:
        # Unique child logger per instance so one engine's file handler does
        # not leak into another engine's console (and vice versa).
        self.logger = logging.getLogger(f"promptgaurd.{next(self._ids)}")
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.propagate = False
        if console is None:
            console = log_file is None
        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(file_handler)
        if console:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())
        self.sink = sink

    def _emit(self, entry: Dict[str, Any]) -> None:
        line = json.dumps(entry, cls=DecisionEncoder, default=str)
        self.logger.log(getattr(logging, entry.get("level", "INFO").upper(), logging.INFO), line)
        if self.sink:
            try:
                self.sink(entry)
            except Exception:
                pass

    def log_decision(
        self,
        prompt_id: str,
        provider: str,
        detector_results: Dict[str, Any],
        decision: str,
        reason: str,
        latency_ms: float,
        raw_prompt: Optional[str] = None,
        level: str = "INFO",
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "level": level,
            "prompt_id": prompt_id,
            "provider": provider,
            "detector_results": detector_results,
            "decision": decision,
            "reason": reason,
            "latency_ms": latency_ms,
        }
        if raw_prompt is not None:
            entry["raw_prompt"] = raw_prompt
        self._emit(entry)

    def log_block_action(self, prompt_id: str, provider: str, action: str, reason: str) -> None:
        """Trace how a BLOCK was enforced: mock_response or exception_raised."""
        self._emit({
            "timestamp": time.time(),
            "level": "WARNING",
            "prompt_id": prompt_id,
            "provider": provider,
            "action": action,
            "reason": reason,
        })

    def log_error(self, prompt_id: str, provider: str, error: str, latency_ms: float) -> None:
        self._emit({
            "timestamp": time.time(),
            "level": "ERROR",
            "prompt_id": prompt_id,
            "provider": provider,
            "error": error,
            "latency_ms": latency_ms,
        })
