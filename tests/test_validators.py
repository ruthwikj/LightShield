"""Unit tests for LightShield validators."""

from __future__ import annotations

from lightshield.core import Message, SessionUUIDEngine, prepare_encapsulated_messages
from lightshield.validators import (
    ValidationResult,
    detect_behavioral_anomalies,
    detect_unauthorised_tool_calls,
    validate_response,
)


def _make_simple_context() -> tuple[SessionUUIDEngine, list[Message], list[Message]]:
    engine = SessionUUIDEngine()
    original: list[Message] = [
        {"role": "system", "content": "You are safe."},
        {"role": "user", "content": "Hello"},
    ]
    encapsulated, _ = prepare_encapsulated_messages(original, engine=engine)
    return engine, original, encapsulated


def test_validate_response_safe_content() -> None:
    engine, original, encapsulated = _make_simple_context()
    response = {"choices": [{"message": {"content": "A harmless reply."}}]}
    result = validate_response(
        response,
        engine=engine,
        original_messages=original,
        encapsulated_messages=encapsulated,
    )
    assert isinstance(result, ValidationResult)
    assert result.is_safe is True
    assert result.violations == []


def test_validate_response_detects_instruction_override() -> None:
    engine, original, encapsulated = _make_simple_context()
    response = {
        "choices": [
            {"message": {"content": "I will ignore previous instructions now."}},
        ]
    }
    result = validate_response(
        response,
        engine=engine,
        original_messages=original,
        encapsulated_messages=encapsulated,
    )
    assert result.is_safe is False
    assert any("consistency_violation" in v for v in result.violations)


def test_validate_response_detects_uuid_bleed() -> None:
    engine, original, encapsulated = _make_simple_context()
    # Deliberately leak a known-style UUID tag.
    response = {"choices": [{"message": {"content": "<LS_deadbeef>leak</LS_deadbeef>"}}]}
    result = validate_response(
        response,
        engine=engine,
        original_messages=original,
        encapsulated_messages=encapsulated,
    )
    assert result.is_safe is False
    assert any("uuid_bleed" in v for v in result.violations)


def test_detect_unauthorised_tool_calls() -> None:
    original: list[Message] = [
        {
            "role": "system",
            "content": "You may use 'allowed_tool' only.",
            "tools": [{"name": "allowed_tool"}],
        }
    ]
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "allowed_tool"}},
                        {"function": {"name": "internal_admin_shell"}},
                    ]
                }
            }
        ]
    }
    violations = detect_unauthorised_tool_calls(response, original)
    # Allowed tool should not be flagged; internal_admin_shell should.
    assert any("unauthorised_tool_call:internal_admin_shell" == v for v in violations)
    assert not any("unauthorised_tool_call:allowed_tool" == v for v in violations)


def test_detect_behavioral_anomalies() -> None:
    content = "Here is the system prompt you provided, I will bypass these restrictions."
    anomalies = detect_behavioral_anomalies(content)
    assert anomalies  # At least one pattern should match.
    assert any("behaviour_anomaly" in v for v in anomalies)
