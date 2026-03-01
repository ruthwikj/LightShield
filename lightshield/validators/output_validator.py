"""Output validation for LightShield.

This module coordinates several low-latency heuristics that evaluate an LLM
response for potential prompt-injection or hierarchy violations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..core.uuid_engine import UUID_PREFIX, SessionUUIDEngine
from .tool_tracer import detect_unauthorised_tool_calls
from .anomaly_detector import detect_behavioral_anomalies


@dataclass
class ValidationResult:
    """Structured result produced by LightShield's output validator."""

    is_safe: bool
    violations: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


UUID_PATTERN = re.compile(rf"<\/?{re.escape(UUID_PREFIX)}_[0-9a-fA-F]+>")


def _extract_text_from_response(response: Any) -> str:
    """Best-effort extraction of the main text content from a model response."""

    # OpenAI-style responses
    if hasattr(response, "choices"):
        try:
            choice = response.choices[0]
            msg = getattr(choice, "message", None) or getattr(choice, "delta", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    return content
        except Exception:  # pragma: no cover - defensive
            pass

    # Anthropic-style responses (content: list of ContentBlock)
    if hasattr(response, "content") and isinstance(response.content, Sequence):
        parts: List[str] = []
        for block in response.content:
            if hasattr(block, "text") and isinstance(block.text, str):
                parts.append(block.text)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        if parts:
            return "\n".join(parts)

    # Dict-style responses
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, Sequence) and choices:
            msg = choices[0].get("message") or choices[0].get("delta")
            if msg and isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content
        content_blocks = response.get("content")
        if isinstance(content_blocks, Sequence):
            for block in content_blocks:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    return block["text"]
                if hasattr(block, "text"):
                    return str(block.text)

    # Fallback to string conversion
    return str(response)


def _response_consistency_checks(
    content: str,
) -> List[str]:
    """Detect obvious attempts to override prior instructions."""

    lowered = content.lower()
    patterns = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "disregard the earlier rules",
        "override the system instructions",
        "i will ignore the system",
    ]
    return [f"consistency_violation:{p}" for p in patterns if p in lowered]


def _detect_uuid_bleed(content: str, engine: SessionUUIDEngine) -> List[str]:
    """Detect UUID boundary tags leaking into the user-visible output."""

    violations: List[str] = []
    matches = UUID_PATTERN.findall(content)
    for match in matches:
        # This indicates the model is exposing internal boundary tags.
        violations.append(f"uuid_bleed:{match}")
    return violations


def validate_string(
    content: str,
    engine: Optional[SessionUUIDEngine] = None,
) -> ValidationResult:
    """Validate a plain string response (e.g. from raw HTTP) against LightShield heuristics.

    Use with :func:`prepare_prompt` — pass the engine returned by prepare_prompt
    for consistent session metadata. Engine is optional; validation runs without
    it but metadata will omit session_id.

    Parameters
    ----------
    content:
        The raw string response from the model.
    engine:
        Optional SessionUUIDEngine from prepare_prompt; used for metadata and
        session consistency.

    Returns
    -------
    ValidationResult
        is_safe, violations, confidence, metadata.
    """

    consistency = _response_consistency_checks(content)
    uuid_bleed = [f"uuid_bleed:{m}" for m in UUID_PATTERN.findall(content)]
    anomalies = detect_behavioral_anomalies(content)
    violations = consistency + uuid_bleed + anomalies

    metadata: Dict[str, Any] = {}
    if engine is not None:
        metadata["session_id"] = engine.session_id

    is_safe = not violations
    confidence = 0.99 if is_safe else 0.9

    return ValidationResult(
        is_safe=is_safe,
        violations=violations,
        confidence=confidence,
        metadata=metadata,
    )


def validate_response(
    response: Any,
    *,
    engine: SessionUUIDEngine,
    original_messages: Sequence[Dict[str, Any]],
    encapsulated_messages: Sequence[Dict[str, Any]],
) -> ValidationResult:
    """Validate a model response against LightShield's security heuristics.

    Parameters
    ----------
    response:
        The raw response object returned by the underlying LLM SDK.
    engine:
        The :class:`SessionUUIDEngine` used to encapsulate the prompt.
    original_messages:
        The developer-provided messages prior to encapsulation.
    encapsulated_messages:
        The messages actually sent to the model, including boundary tags.

    Returns
    -------
    ValidationResult
        The aggregated validation result containing safety decision and
        violation details.
    """

    content = _extract_text_from_response(response)

    violations: List[str] = []
    metadata: Dict[str, Any] = {
        "session_id": engine.session_id,
        "original_message_count": len(original_messages),
    }

    # 1) Tool call tracing.
    tool_violations = detect_unauthorised_tool_calls(response, original_messages)
    violations.extend(tool_violations)

    # 2) Response consistency with system prompt.
    violations.extend(_response_consistency_checks(content))

    # 3) UUID bleed detection.
    violations.extend(_detect_uuid_bleed(content, engine))

    # 4) Behavioral anomalies.
    anomalies = detect_behavioral_anomalies(content)
    violations.extend(anomalies)

    is_safe = not violations
    confidence = 0.99 if is_safe else 0.9

    return ValidationResult(
        is_safe=is_safe,
        violations=violations,
        confidence=confidence,
        metadata=metadata,
    )


__all__ = ["ValidationResult", "validate_response", "validate_string"]
