"""Unit tests for LightShield public API and exceptions."""

from __future__ import annotations

from lightshield import LightShield
from lightshield.exceptions import LightShieldViolationException


def test_lightshield_wrap_duck_typing_order() -> None:
    class CompletionsClient:
        def create(self, *args, **kwargs):
            return type("R", (), {"choices": []})()

    class ChatClient:
        def __init__(self) -> None:
            self.chat = type("C", (), {"completions": CompletionsClient()})()

    class MessagesClient:
        def __init__(self) -> None:
            self.messages = type("M", (), {"create": lambda *a, **k: None})()

    class ChainClient:
        def invoke(self, *args, **kwargs):
            return "ok"

    s = LightShield()
    wrapped_chat = s.wrap(ChatClient())
    wrapped_messages = s.wrap(MessagesClient())
    wrapped_chain = s.wrap(ChainClient())

    # Chat client should gain a .chat with .completions.create.
    assert hasattr(wrapped_chat, "chat")
    # Messages client should gain a .messages namespace.
    assert hasattr(wrapped_messages, "messages")
    # Chain client becomes a LangChainAdapter-like object with invoke.
    assert hasattr(wrapped_chain, "invoke")


def test_lightshield_violation_exception_str_contains_details() -> None:
    exc = LightShieldViolationException(
        message="Violation",
        violations=["rule1", "rule2"],
        confidence=0.95,
    )
    text = str(exc)
    assert "Violation" in text
    assert "rule1" in text
    assert "0.95" in text
