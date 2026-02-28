"""Unit tests for LightShield core modules."""

from __future__ import annotations

from lightshield.core import (
    Message,
    SessionUUIDEngine,
    TrustLevel,
    build_system_prompt_with_trust_map,
    get_trust_level_for_role,
    prepare_encapsulated_messages,
)


def test_trust_level_mapping_and_instructional_flag() -> None:
    assert get_trust_level_for_role("system") is TrustLevel.SYSTEM
    assert get_trust_level_for_role("user") is TrustLevel.USER
    assert get_trust_level_for_role("retrieved") is TrustLevel.RETRIEVED
    assert get_trust_level_for_role("tool") is TrustLevel.TOOL
    # Unknown roles default to USER.
    assert get_trust_level_for_role("unknown") is TrustLevel.USER
    assert TrustLevel.SYSTEM.is_instructional() is True
    assert TrustLevel.USER.is_instructional() is True
    assert TrustLevel.RETRIEVED.is_instructional() is False


def test_session_uuid_engine_generates_distinct_tags() -> None:
    engine = SessionUUIDEngine()
    tags = list(engine.tags)
    assert len(tags) == len(TrustLevel)
    # Tokens should be unique per level in a session.
    tokens = {t.token for t in tags}
    assert len(tokens) == len(TrustLevel)
    # Encapsulation should include matching open/close tags.
    sample = "hello"
    wrapped = engine.encapsulate(sample, TrustLevel.USER)
    tag = engine.tag_for_level(TrustLevel.USER)
    assert tag.open in wrapped
    assert tag.close in wrapped
    assert sample in wrapped


def test_build_system_prompt_with_trust_map_includes_boundary_rules() -> None:
    engine = SessionUUIDEngine()
    base = "You are a helpful assistant."
    combined = build_system_prompt_with_trust_map(base, engine)
    assert "LIGHTSHIELD BOUNDARY RULES" in combined
    # At least one opening boundary tag should appear in the text.
    any_tag_open = any(tag.open in combined for tag in engine.tags)
    assert any_tag_open


def test_prepare_encapsulated_messages_injects_synthetic_system_when_missing() -> None:
    engine = SessionUUIDEngine()
    messages: list[Message] = [
        {"role": "user", "content": "Hello"},
    ]
    encapsulated, _ = prepare_encapsulated_messages(messages, engine=engine)
    # A synthetic system message should have been prepended.
    assert encapsulated[0]["role"] == "system"
    assert "LIGHTSHIELD BOUNDARY RULES" in encapsulated[0]["content"]
    # User content should still be present and wrapped.
    assert "Hello" in encapsulated[-1]["content"]


def test_prepare_encapsulated_messages_augments_existing_system_prompt() -> None:
    engine = SessionUUIDEngine()
    messages: list[Message] = [
        {"role": "system", "content": "You are system."},
        {"role": "user", "content": "Hi"},
    ]
    encapsulated, _ = prepare_encapsulated_messages(messages, engine=engine)
    # First system message content should include boundary rules.
    system_content = encapsulated[0]["content"]
    assert "LIGHTSHIELD BOUNDARY RULES" in system_content
    assert "You are system." in system_content
