"""Core primitives for the LightShield firewall."""

from .hierarchy import TrustLevel, get_trust_level_for_role
from .uuid_engine import SessionUUIDEngine, UUIDTag, UUID_PREFIX
from .encapsulator import Message, build_system_prompt_with_trust_map, prepare_encapsulated_messages

__all__ = [
    "TrustLevel",
    "get_trust_level_for_role",
    "SessionUUIDEngine",
    "UUIDTag",
    "UUID_PREFIX",
    "Message",
    "build_system_prompt_with_trust_map",
    "prepare_encapsulated_messages",
]
