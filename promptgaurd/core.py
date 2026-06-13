"""Core Gaudrial engine, policies, and decisions."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from .config import Config
from .logging_config import StructuredLogger
from .exceptions import GuardBlocked, GuardError
from .detectors.base import BaseDetector
from .detectors.bert_detector import BertDetector


@dataclass
class Decision:
    """Result of a guard scan."""

    prompt_id: str
    decision: str  # "ALLOW", "WARN", "BLOCK"
    scores: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.7
    reason: str = ""
    latency_ms: float = 0.0
    provider: str = "unknown"
    raw_prompt: Optional[str] = None
    class_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "decision": self.decision,
            "scores": self.scores,
            "threshold": self.threshold,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
            "provider": self.provider,
            "class_name": self.class_name,
        }


class Policy:
    """Policy rules for guard decisions."""

    def __init__(self, threshold: float = 0.7, warn_threshold: Optional[float] = None) -> None:
        self.threshold = threshold
        self.warn_threshold = warn_threshold if warn_threshold is not None else threshold * 0.85

    def evaluate(self, max_score: float) -> str:
        if max_score >= self.threshold:
            return "BLOCK"
        if max_score >= self.warn_threshold:
            return "WARN"
        return "ALLOW"


class Gaudrial:
    """Guard engine powered by fine-tuned BERT-mini."""

    def __init__(
        self,
        policy: Optional[str] = None,
        threshold: Optional[float] = None,
        fail_mode: Optional[str] = None,
        log_level: Optional[str] = None,
        log_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        log_file: Optional[str] = "logs/promptgaurd.jsonl",
        custom_detectors: Optional[List[BaseDetector]] = None,
        mask_raw_prompt: bool = True,
        block_mode: str = "mock",
        block_message: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = Config(
                policy=policy or "standard",
                threshold=threshold,
                fail_mode=fail_mode,
                log_level=log_level,
                log_sink=log_sink,
                log_file=log_file,
                custom_detectors=custom_detectors,
                mask_raw_prompt=mask_raw_prompt,
                block_mode=block_mode,
                block_message=block_message,
            )

        self.policy = Policy(threshold=self.config.threshold)
        self.logger = StructuredLogger(
            level=self.config.log_level,
            sink=self.config.log_sink,
            log_file=getattr(self.config, "log_file", None),
        )

        self.detectors: List[BaseDetector] = [BertDetector()]
        if self.config.custom_detectors:
            self.detectors.extend(self.config.custom_detectors)

    def analyze(self, prompt: str, provider: str = "unknown") -> Decision:
        """Run BERT-mini detector and return a Decision. Never raises."""
        prompt_id = str(uuid.uuid4())
        start = time.perf_counter()
        scores: Dict[str, float] = {}
        class_name = ""

        try:
            for detector in self.detectors:
                try:
                    if hasattr(detector, "detect_and_classify"):
                        score, class_name, _ = detector.detect_and_classify(prompt)
                        scores[detector.name] = score
                    else:
                        score = detector.detect(prompt)
                        scores[detector.name] = score
                        if hasattr(detector, "classify"):
                            try:
                                cn, _ = detector.classify(prompt)
                                class_name = cn
                            except Exception:
                                pass
                except Exception as e:
                    scores[detector.name] = 0.0
                    self.logger.log_error(prompt_id, provider, f"{detector.name} failed: {e}", 0.0)
                    if self.config.fail_mode == "closed":
                        raise GuardError(f"{detector.name} failed: {e}", original_error=e)

            max_score = max(scores.values()) if scores else 0.0
            decision_label = self.policy.evaluate(max_score)

            reasons = []
            for name, score in scores.items():
                if score >= self.policy.threshold:
                    reasons.append(f"{name}={score:.2f}")
            reason = f"Threshold exceeded by {', '.join(reasons)}" if reasons else "No detectors flagged"

            latency_ms = (time.perf_counter() - start) * 1000

            decision = Decision(
                prompt_id=prompt_id,
                decision=decision_label,
                scores=scores,
                threshold=self.config.threshold,
                reason=reason,
                latency_ms=latency_ms,
                provider=provider,
                raw_prompt=prompt if not self.config.mask_raw_prompt else None,
                class_name=class_name,
            )

            self.logger.log_decision(
                prompt_id=prompt_id,
                provider=provider,
                detector_results=scores,
                decision=decision_label,
                reason=reason,
                latency_ms=latency_ms,
                raw_prompt=prompt if not self.config.mask_raw_prompt else None,
                level="DEBUG" if decision_label == "ALLOW" else "WARNING",
            )

            return decision

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            self.logger.log_error(prompt_id, provider, f"Guard engine failed: {e}", latency_ms)
            if self.config.fail_mode == "closed":
                raise GuardError(f"Guard engine failed: {e}", original_error=e)
            return Decision(
                prompt_id=prompt_id,
                decision="ALLOW",
                scores={},
                threshold=self.config.threshold,
                reason=f"Guard engine error (fail_open): {e}",
                latency_ms=latency_ms,
                provider=provider,
            )

    def guard(
        self,
        prompt: str,
        provider: str = "unknown",
        on_block: Optional[Callable[[Decision], Any]] = None,
    ) -> Decision:
        """Analyze the prompt and act on a BLOCK according to block_mode.

        - ``on_block`` callback given: call it with the Decision.
        - ``block_mode="raise"``: raise GuardBlocked (legacy behavior).
        - ``block_mode="mock"`` (default): return the Decision unraised so
          the caller (adapter/middleware/decorator) can substitute a
          provider-shaped mock response and keep the pipeline alive.
        """
        decision = self.analyze(prompt, provider=provider)
        if decision.decision == "BLOCK":
            if on_block:
                on_block(decision)
            elif self.config.block_mode == "raise":
                self.logger.log_block_action(
                    decision.prompt_id, provider, "exception_raised", decision.reason
                )
                raise GuardBlocked(f"Prompt blocked: {decision.reason}", decision=decision)
        return decision
