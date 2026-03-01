"""Ollama client wrapper that runs chat through LightShield (tagged layers + response sanitization)."""

import re
from typing import Any, Callable, Optional, Sequence, Union

from .system import LayerPrompt


def _parse_messages(messages: Optional[Sequence[Any]]) -> tuple[str, str]:
    """Extract system and user content from a list of message dicts (role + content)."""
    system_parts: list[str] = []
    user_content = ""
    if messages:
        for m in messages:
            role = (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) or ""
            content = (m.get("content") if isinstance(m, dict) else getattr(m, "content", None)) or ""
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in content
                )
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                user_content = content
        system_content = "\n\n".join(system_parts) if system_parts else ""
    else:
        system_content = ""
    return system_content, user_content


def _assemble_system_content(layer_prompt: LayerPrompt, system_content: str) -> str:
    """Build the system message: define tag behavior first (untagged), then the SYSTEM layer content in its tag. No tag is used before the model knows what tags mean."""
    tags = layer_prompt.tags
    authority = layer_prompt.authority_text()
    system_block = tags["system"].wrap(system_content or "")
    return authority + "\n\n" + system_block


def _sanitize_content(content: str, layer_prompt: LayerPrompt) -> str:
    """Remove LightShield tag markers from text so they are not exposed to the application."""
    if not content:
        return content
    text = content
    for tag in layer_prompt.tags.values():
        sid = tag.short_id
        text = text.replace(f"<LS_{sid}>", "").replace(f"</LS_{sid}>", "")
    text = re.sub(r"<LS_[0-9a-f]{8}>", "", text)
    text = re.sub(r"</LS_[0-9a-f]{8}>", "", text)
    return text


def _sanitize_response(response: Any, layer_prompt: LayerPrompt) -> Any:
    """Return a copy of the response with message content sanitized (no tag leakage)."""
    if response is None:
        return response
    if isinstance(response, dict):
        out = dict(response)
        if "message" in out and isinstance(out["message"], dict):
            msg = dict(out["message"])
            if "content" in msg and msg["content"] is not None:
                msg["content"] = _sanitize_content(str(msg["content"]), layer_prompt)
            out["message"] = msg
        return out
    if hasattr(response, "message") and hasattr(response.message, "content"):
        content = getattr(response.message, "content", None)
        if content is not None:
            setattr(response.message, "content", _sanitize_content(str(content), layer_prompt))
    return response


class Shield:
    """Wraps Ollama's module-level chat so calls go through LightShield. Uses ollama.chat(model, messages) directly; no Client instance."""

    def __init__(self, engine: str = "ollama") -> None:
        if engine == "ollama":
            try:
                import ollama as _ollama
                self._ollama = _ollama
            except ImportError as e:
                raise ImportError(
                    "'ollama' package not installed. Install with: pip install ollama"
                ) from e
        else:
            raise ValueError(f"Invalid engine: {engine}")

    def chat(
        self,
        model: str,
        messages: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run chat through LightShield (non-streaming only). Response is always sanitized. Streaming is not supported so we can strip tag leakage."""
        system_content, user_content = _parse_messages(messages)
        layer_prompt = LayerPrompt()
        tags = layer_prompt.tags
        system_body = _assemble_system_content(layer_prompt, system_content)
        user_body = tags["user"].wrap(user_content or "")
        transformed_messages = [
            {"role": "system", "content": system_body},
            {"role": "user", "content": user_body},
        ]
        kwargs_no_stream = {k: v for k, v in kwargs.items() if k != "stream"}
        resp = self._ollama.chat(
            model=model,
            messages=transformed_messages,
            stream=False,
            **kwargs_no_stream,
        )
        return _sanitize_response(resp, layer_prompt)

    def __getattr__(self, name: str) -> Any:
        """Delegate other names (generate, list, pull, etc.) to the ollama module."""
        return getattr(self._ollama, name)


class RagShield:
    """Engine-agnostic RAG prompt-injection protection.

    Prepares tagged messages and a sanitizer closure. The caller sends messages
    through their own LLM client (OpenAI, Anthropic, Ollama, etc.).
    """

    def prepare(
        self,
        system: str,
        context: Union[str, Sequence[str]],
        query: str,
    ) -> tuple[list[dict[str, str]], Callable[[str], str]]:
        """Build tagged messages for a RAG call and return a response sanitizer.

        Returns:
            (messages, sanitizer) — *messages* is a list of role/content dicts
            ready for any chat-completions API; *sanitizer* is a function that
            strips LightShield tag markers from response text.
        """
        layer_prompt = LayerPrompt()
        tags = layer_prompt.tags

        # Normalise context to a list of strings
        if isinstance(context, str):
            chunks = [context]
        else:
            chunks = list(context)

        # Wrap each retrieved doc individually in the "retrieved" tag
        retrieved_parts = [tags["retrieved"].wrap(chunk) for chunk in chunks]

        # System content: authority text + system-tagged instructions + retrieved docs
        system_content = "\n\n".join(
            [
                layer_prompt.authority_text(),
                tags["system"].wrap(system),
                *retrieved_parts,
            ]
        )

        # User query wrapped in "user" tag
        user_content = tags["user"].wrap(query)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        def sanitizer(text: str) -> str:
            return _sanitize_content(text, layer_prompt)

        return messages, sanitizer
