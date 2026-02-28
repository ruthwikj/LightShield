"""Trust hierarchy definitions for the LightShield prompt injection firewall.

This module defines the strict instruction hierarchy used by LightShield to
differentiate between trusted instructions and untrusted data. The hierarchy
is enforced structurally via UUID encapsulation and is consulted by both the
encapsulation engine and the output validators.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict


class TrustLevel(IntEnum):
    """Ordered trust levels for different prompt segments.

    The numeric values encode precedence: lower numbers indicate higher
    authority within the instruction hierarchy.
    """

    SYSTEM = 1
    USER = 2
    RETRIEVED = 3
    TOOL = 4

    def is_instructional(self) -> bool:
        """Return True if this level is allowed to carry authoritative instructions.

        Currently, only SYSTEM and USER levels are considered instructional;
        retrieved content and tool outputs are treated as data-only.
        """

        return self in {TrustLevel.SYSTEM, TrustLevel.USER}


ROLE_TO_LEVEL: Dict[str, TrustLevel] = {
    "system": TrustLevel.SYSTEM,
    "developer": TrustLevel.SYSTEM,  # OpenAI "developer" role
    "user": TrustLevel.USER,
    "retrieved": TrustLevel.RETRIEVED,
    "rag": TrustLevel.RETRIEVED,
    "context": TrustLevel.RETRIEVED,
    "tool": TrustLevel.TOOL,
    "tool_output": TrustLevel.TOOL,
}


def get_trust_level_for_role(role: str) -> TrustLevel:
    """Map a chat message role string to a :class:`TrustLevel`.

    Any unknown roles default to :class:`TrustLevel.USER` to avoid accidentally
    treating unrecognised segments as data-only.

    Parameters
    ----------
    role:
        The role field from a chat message dictionary.

    Returns
    -------
    TrustLevel
        The inferred trust level for the given role.
    """

    normalized = (role or "").lower()
    return ROLE_TO_LEVEL.get(normalized, TrustLevel.USER)


__all__ = ["TrustLevel", "get_trust_level_for_role", "ROLE_TO_LEVEL"]
