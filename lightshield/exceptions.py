"""Custom exceptions raised by the LightShield middleware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .core.hierarchy import TrustLevel


@dataclass
class LightShieldViolationException(Exception):
    """Raised when LightShield detects a high-confidence security violation.

    Parameters
    ----------
    message:
        Human-readable summary of the violation.
    violations:
        A list of specific violation codes or descriptions.
    confidence:
        A float in the range [0.0, 1.0] indicating LightShield's confidence that
        the response is unsafe.
    violated_uuid_level:
        The trust level (1-4) that was violated, if applicable.
    triggering_pattern:
        The pattern or rule that triggered the violation.
    raw_content:
        The raw content that caused the violation (may be truncated).
    recommended_action:
        Suggested remediation for the developer.
    """

    message: str
    violations: List[str]
    confidence: float
    violated_uuid_level: Optional[TrustLevel] = None
    triggering_pattern: Optional[str] = None
    raw_content: Optional[str] = None
    recommended_action: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"{self.message} (confidence={self.confidence:.2f})"]
        if self.violations:
            parts.append(f"violations={self.violations[:3]}{'...' if len(self.violations) > 3 else ''}")
        if self.triggering_pattern:
            parts.append(f"pattern={self.triggering_pattern}")
        return " ".join(parts)


__all__ = ["LightShieldViolationException"]
