"""Tool call tracing for LightShield.

This module inspects model responses for tool calls and determines whether they
appear authorised based on the original prompt context. The implementation is
intentionally conservative and lightweight.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _extract_declared_tools(messages: Sequence[Dict[str, Any]]) -> List[str]:
    """Extract tool names that appear to be authorised in the prompt.

    This is a heuristic that scans for a ``tools`` field or for textual
    mentions in system/user messages.
    """

    declared: List[str] = []
    for msg in messages:
        tools = msg.get("tools")
        if isinstance(tools, list):
            for t in tools:
                name = getattr(t, "name", None) or t.get("name") if isinstance(t, dict) else None
                if isinstance(name, str):
                    declared.append(name)
    return list(dict.fromkeys(declared))


def _iter_tool_calls_from_response(response: Any) -> List[str]:
    """Return a list of tool names that the model attempted to call."""

    tool_names: List[str] = []

    # OpenAI-style structured tool calls on choices[].message.tool_calls
    choices = getattr(response, "choices", None)
    if choices:
        for choice in choices:
            msg = getattr(choice, "message", None) or getattr(choice, "delta", None)
            if msg is None:
                continue
            tool_calls = getattr(msg, "tool_calls", None) or getattr(
                msg, "function_call", None
            )
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    name = getattr(tc, "function", None)
                    if name and hasattr(name, "name"):
                        tool_names.append(str(name.name))
            elif isinstance(tool_calls, dict):
                # function_call style
                name = tool_calls.get("name")
                if isinstance(name, str):
                    tool_names.append(name)

    # Dict-style responses
    if isinstance(response, dict):
        choices_dict = response.get("choices") or []
        for choice in choices_dict:
            msg = choice.get("message") or choice.get("delta") or {}
            tool_calls = msg.get("tool_calls") or msg.get("function_call")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    if isinstance(name, str):
                        tool_names.append(name)
            elif isinstance(tool_calls, dict):
                name = tool_calls.get("name")
                if isinstance(name, str):
                    tool_names.append(name)

    return list(dict.fromkeys(tool_names))


def detect_unauthorised_tool_calls(
    response: Any, original_messages: Sequence[Dict[str, Any]]
) -> List[str]:
    """Return violation codes for tool calls that appear unauthorised.

    A tool call is considered authorised if its name appears in a ``tools``
    list attached to any original message. This is intentionally simple and
    side-steps provider-specific policy models while still catching obvious
    "spontaneous" tool invocations.
    """

    declared_tools = _extract_declared_tools(original_messages)
    called_tools = _iter_tool_calls_from_response(response)

    violations: List[str] = []
    for name in called_tools:
        if name not in declared_tools:
            violations.append(f"unauthorised_tool_call:{name}")

    return violations


__all__ = ["detect_unauthorised_tool_calls"]
