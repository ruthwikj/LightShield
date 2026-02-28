"""Behavioral anomaly detection for LightShield.

This module implements lightweight, rule-based checks for obviously unsafe
behaviours in model outputs. It is intentionally simple to keep latency
under 50ms while still catching high-signal anomalies.
"""

from __future__ import annotations

from typing import List


SUSPICIOUS_PATTERNS = [
    "here is the system prompt you provided",
    "i will reveal your system prompt",
    "lightshield boundary rules",
    "ls_",
    "i will bypass these restrictions",
    "to exfiltrate the data",
]


def detect_behavioral_anomalies(content: str) -> List[str]:
    """Return a list of behavioural anomaly violation codes.

    The detector currently looks for simple high-risk phrases related to
    data exfiltration or explicit attempts to bypass safety rules.
    """

    lowered = content.lower()
    return [f"behaviour_anomaly:{p}" for p in SUSPICIOUS_PATTERNS if p in lowered]


__all__ = ["detect_behavioral_anomalies"]
