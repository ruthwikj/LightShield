"""LangChain adapter for LightShield.

This adapter wraps a LangChain ``Chain``-like object and automatically
encapsulates retrieved documents (RAG context) as Level-3 data segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

from ..core.encapsulator import Message, prepare_encapsulated_messages
from ..core.hierarchy import TrustLevel
from ..core.uuid_engine import SessionUUIDEngine
from ..validators.output_validator import ValidationResult, validate_response


def _wrap_rag_documents(
    docs: Iterable[Any], engine: SessionUUIDEngine
) -> List[str]:
    """Encapsulate LangChain ``Document`` objects as retrieved content."""

    wrapped: List[str] = []
    for doc in docs:
        pagestr = getattr(doc, "page_content", None)
        if pagestr is None:
            pagestr = str(doc)
        wrapped.append(engine.encapsulate(pagestr, TrustLevel.RETRIEVED))
    return wrapped


@dataclass
class LangChainResult:
    """Wrapper around a LangChain result with validation metadata."""

    raw: Any
    lightshield: ValidationResult

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self.raw, item)

    def __getitem__(self, item: Any) -> Any:  # pragma: no cover - passthrough
        return self.raw[item]


class LangChainAdapter:
    """Adapter that wraps a LangChain chain object.

    The adapter expects the underlying chain's LLM to receive standard
    ``messages`` input; for many RAG chains this is exposed via the ``input``
    and ``context`` keys.
    """

    def __init__(self, chain: Any) -> None:
        self._chain = chain

    def invoke(self, input: Any, **kwargs: Any) -> LangChainResult:  # noqa: A002
        """Invoke the underlying chain with automatic RAG encapsulation.

        The adapter looks for a ``documents`` or ``context`` kwarg containing
        LangChain ``Document`` instances and rewrites them as encapsulated
        retrieved-content messages.
        """

        engine = SessionUUIDEngine()

        # Prepare pseudo-messages when passing through to an LLM-backed chain.
        messages: List[Message] = []
        messages.append({"role": "user", "content": input})

        docs = kwargs.get("documents") or kwargs.get("context")
        if docs:
            wrapped_docs = _wrap_rag_documents(docs, engine)
            for content in wrapped_docs:
                messages.append({"role": "retrieved", "content": content})

        encapsulated_messages, _ = prepare_encapsulated_messages(messages, engine=engine)

        # Pass encapsulated messages through in a provider-agnostic way.
        call_kwargs = dict(kwargs)
        call_kwargs["messages"] = encapsulated_messages

        raw_result = self._chain.invoke(input, **call_kwargs)
        validation = validate_response(
            raw_result,
            engine=engine,
            original_messages=messages,
            encapsulated_messages=encapsulated_messages,
        )
        return LangChainResult(raw=raw_result, lightshield=validation)


__all__ = ["LangChainAdapter", "LangChainResult"]
