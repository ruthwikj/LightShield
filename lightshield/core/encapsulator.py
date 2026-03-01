"""Prompt encapsulation utilities for LightShield.

This module ties together the trust hierarchy and UUID engine to build fully
encapsulated prompts for downstream LLM providers. It is backend-agnostic and
operates on simple ``messages`` lists compatible with common chat APIs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .hierarchy import TrustLevel, get_trust_level_for_role
from .uuid_engine import SessionUUIDEngine


Message = Dict[str, Any]


def build_system_prompt_with_trust_map(
    base_system_prompt: str, engine: SessionUUIDEngine
) -> str:
    """Append LightShield's trust-map instructions to a base system prompt.

    Parameters
    ----------
    base_system_prompt:
        The developer-provided system prompt.
    engine:
        The active :class:`SessionUUIDEngine` instance whose tags should be
        described to the model.

    Returns
    -------
    str
        The combined system prompt including the trust-map instructions.
    """

    boundary_rules = engine.build_trust_map_instructions()
    return f"{base_system_prompt.rstrip()}\n\n---\nLIGHTSHIELD BOUNDARY RULES\n\n{boundary_rules}"


def prepare_encapsulated_messages(
    messages: List[Message],
    engine: Optional[SessionUUIDEngine] = None,
) -> Tuple[List[Message], SessionUUIDEngine]:
    """Encapsulate chat messages and inject the LightShield trust map.

    This function is the primary entry point used by adapters. It:

    1. Creates a new :class:`SessionUUIDEngine` if one is not provided.
    2. Encapsulates every message content with the appropriate boundary tags
       based on its role-derived :class:`TrustLevel`.
    3. Appends the trust-map instructions to the first system-level message,
       or injects a synthetic system message if none exist.

    Parameters
    ----------
    messages:
        A list of message dictionaries, each with at least ``role`` and
        ``content`` keys.
    engine:
        Optional existing :class:`SessionUUIDEngine` to reuse; if omitted a
        fresh engine is created for this call.

    Returns
    -------
    tuple
        A pair ``(encapsulated_messages, engine)`` where ``engine`` is the
        :class:`SessionUUIDEngine` used for this preparation.
    """

    if engine is None:
        engine = SessionUUIDEngine()

    encapsulated: List[Message] = []
    trust_map_injected = False

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        level: TrustLevel = get_trust_level_for_role(role)
        wrapped_content = engine.encapsulate(str(content), level)

        if level == TrustLevel.SYSTEM and not trust_map_injected:
            base_system_prompt = str(content)
            wrapped_with_map = engine.encapsulate(
                build_system_prompt_with_trust_map(base_system_prompt, engine),
                TrustLevel.SYSTEM,
            )
            wrapped_content = wrapped_with_map
            trust_map_injected = True

        encapsulated.append({**msg, "content": wrapped_content})

    if not trust_map_injected:
        # No user-provided system prompt; inject a synthetic one that only
        # describes the trust map and boundary rules.
        synthetic_base = (
            "You are a helpful assistant operating behind the LightShield firewall."
        )
        synthetic_content = engine.encapsulate(
            build_system_prompt_with_trust_map(synthetic_base, engine),
            TrustLevel.SYSTEM,
        )
        encapsulated.insert(
            0,
            {
                "role": "system",
                "content": synthetic_content,
            },
        )
    print(encapsulated)

    return encapsulated, engine


def prepare_prompt(
    system: str,
    user: str,
    retrieved: Optional[str] = None,
    tool: Optional[str] = None,
    engine: Optional[SessionUUIDEngine] = None,
) -> Tuple[str, SessionUUIDEngine]:
    """Build a single encapsulated prompt string for raw HTTP APIs (e.g. Ollama).

    Accepts plain strings and returns a plain string prompt usable with any
    model endpoint. Use with :func:`validate_string` for the full flow.

    Parameters
    ----------
    system:
        System prompt (Level 1).
    user:
        User input (Level 2).
    retrieved:
        Optional retrieved/RAG content (Level 3).
    tool:
        Optional tool output (Level 4).
    engine:
        Optional engine to reuse; creates a fresh one if omitted.

    Returns
    -------
    tuple
        (prompt_string, engine) — pass engine to validate_string for validation.
    """

    if engine is None:
        engine = SessionUUIDEngine()

    system_with_map = build_system_prompt_with_trust_map(system, engine)
    parts: List[str] = []
    parts.append(engine.encapsulate(system_with_map, TrustLevel.SYSTEM))
    parts.append(engine.encapsulate(user, TrustLevel.USER))
    if retrieved:
        parts.append(engine.encapsulate(retrieved, TrustLevel.RETRIEVED))
    if tool:
        parts.append(engine.encapsulate(tool, TrustLevel.TOOL))

    return "\n\n".join(parts), engine


__all__ = [
    "Message",
    "build_system_prompt_with_trust_map",
    "prepare_encapsulated_messages",
    "prepare_prompt",
]
