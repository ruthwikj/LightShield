"""Custom exceptions raised by the LightShield middleware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


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
    """

    message: str
    violations: List[str]
    confidence: float

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.message} (confidence={self.confidence:.2f}, violations={self.violations})"


__all__ = ["LightShieldViolationException"]
