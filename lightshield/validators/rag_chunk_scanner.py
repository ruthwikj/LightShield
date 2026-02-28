"""RAG chunk scanning for LightShield.

Scans retrieved document chunks and metadata for instruction-like patterns
that could be used for prompt injection. Used at retrieval time before
context assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union


INSTRUCTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard the earlier rules",
    "override the system",
    "you must now",
    "you are now",
    "follow these instructions",
    "follow these new instructions",
    "new instructions:",
    "system override",
    "bypass the",
    "ignore the system prompt",
    "reveal the system prompt",
    "leak the",
    "exfiltrate",
    "output the system prompt",
    "print the system prompt",
    "execute the following",
    "run the following",
    "call the function",
    "call the tool",
]


@dataclass
class FlaggedChunk:
    """A chunk flagged as potentially malicious."""

    id: Optional[str]
    score: Optional[float]
    content_preview: str
    triggering_patterns: List[str]
    metadata_keys_scanned: List[str]
    risk_score: float


def _extract_text_from_chunk(chunk: Any) -> str:
    """Extract all text content from a Pinecone match for scanning."""

    parts: List[str] = []
    if isinstance(chunk, dict):
        metadata = chunk.get("metadata") or {}
        if isinstance(metadata, dict):
            for v in metadata.values():
                if isinstance(v, str):
                    parts.append(v)
        text = chunk.get("content") or chunk.get("text") or chunk.get("chunk")
        if isinstance(text, str):
            parts.append(text)
    else:
        metadata = getattr(chunk, "metadata", None)
        if metadata and isinstance(metadata, dict):
            for v in metadata.values():
                if isinstance(v, str):
                    parts.append(v)
        for attr in ("content", "text", "chunk", "page_content"):
            val = getattr(chunk, attr, None)
            if isinstance(val, str):
                parts.append(val)
    return "\n".join(parts)


def _get_chunk_id(chunk: Any) -> Optional[str]:
    """Extract chunk ID from a Pinecone match."""
    if isinstance(chunk, dict):
        return chunk.get("id")
    return getattr(chunk, "id", None)


def _get_chunk_score(chunk: Any) -> Optional[float]:
    """Extract similarity score from a Pinecone match."""
    if isinstance(chunk, dict):
        return chunk.get("score")
    return getattr(chunk, "score", None)


def _metadata_dict(chunk: Any) -> Dict[str, Any]:
    """Get metadata as a flat dict for scanning."""
    if isinstance(chunk, dict):
        m = chunk.get("metadata") or {}
    else:
        m = getattr(chunk, "metadata", None) or {}
    return m if isinstance(m, dict) else {}


def _scan_text_for_patterns(text: str) -> List[str]:
    """Return list of instruction-like patterns found in text."""

    lowered = text.lower()
    found: List[str] = []
    for p in INSTRUCTION_PATTERNS:
        if p in lowered:
            found.append(p)
    return found


def scan_chunk(chunk: Any) -> Optional[FlaggedChunk]:
    """Scan a single Pinecone match for instruction-like patterns.

    Parameters
    ----------
    chunk:
        A Pinecone match (dict or object with id, score, metadata).

    Returns
    -------
    FlaggedChunk or None
        If suspicious patterns are found, returns a FlaggedChunk; else None.
    """

    text = _extract_text_from_chunk(chunk)
    metadata = _metadata_dict(chunk)
    metadata_texts: List[str] = []
    for v in metadata.values():
        if isinstance(v, str):
            metadata_texts.append(v)
    combined = f"{text}\n" + "\n".join(metadata_texts) if metadata_texts else text
    patterns = _scan_text_for_patterns(combined)
    if not patterns:
        return None
    preview = (text or "".join(str(v) for v in metadata.values() if isinstance(v, str)))[:200]
    return FlaggedChunk(
        id=_get_chunk_id(chunk),
        score=_get_chunk_score(chunk),
        content_preview=preview,
        triggering_patterns=patterns,
        metadata_keys_scanned=list(metadata.keys()),
        risk_score=min(0.99, 0.5 + 0.1 * len(patterns)),
    )


__all__ = ["FlaggedChunk", "scan_chunk", "INSTRUCTION_PATTERNS"]
