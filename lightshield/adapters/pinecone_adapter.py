"""Pinecone RAG adapter for LightShield.

Intercepts Pinecone query results at retrieval time, scans chunks for
instruction-like patterns, quarantines suspicious content, and UUID-tags
safe chunks before they reach the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from ..core.hierarchy import TrustLevel
from ..core.uuid_engine import SessionUUIDEngine
from ..exceptions import LightShieldViolationException
from ..validators.rag_chunk_scanner import FlaggedChunk, scan_chunk


@dataclass
class LightShieldRAGResult:
    """Result of sanitizing Pinecone query results.

    Attributes
    ----------
    safe_chunks:
        Chunks that passed scanning, UUID-tagged at Level 3 (retrieved).
    flagged_chunks:
        Chunks quarantined due to instruction-like patterns.
    risk_score:
        Aggregate risk [0.0, 1.0] based on flagged content.
    engine:
        Session UUID engine used for tagging; can be passed to wrap() for
        consistent trust map.
    """

    safe_chunks: List[Dict[str, Any]] = field(default_factory=list)
    flagged_chunks: List[FlaggedChunk] = field(default_factory=list)
    risk_score: float = 0.0
    engine: Optional[SessionUUIDEngine] = field(default=None, repr=False)

    def to_messages(self) -> List[Dict[str, Any]]:
        """Convert safe_chunks to chat messages with role='retrieved'."""

        return [
            {"role": "retrieved", "content": c.get("content", c.get("text", ""))}
            for c in self.safe_chunks
        ]


def _extract_matches(raw_results: Any) -> List[Any]:
    """Extract list of matches from Pinecone query result."""

    if isinstance(raw_results, list):
        return raw_results
    if hasattr(raw_results, "matches"):
        return list(raw_results.matches)
    if isinstance(raw_results, dict):
        return raw_results.get("matches") or raw_results.get("results") or []
    return []


def _chunk_to_content(chunk: Any) -> str:
    """Extract displayable text content from a match."""

    if isinstance(chunk, dict):
        return chunk.get("content") or chunk.get("text") or str(chunk.get("metadata", ""))
    return (
        getattr(chunk, "content", None)
        or getattr(chunk, "text", None)
        or str(getattr(chunk, "metadata", ""))
    )


def _build_tagged_chunk(
    chunk: Any,
    engine: SessionUUIDEngine,
) -> Dict[str, Any]:
    """Build a UUID-tagged chunk dict for safe_chunks."""

    content = _chunk_to_content(chunk)
    tagged = engine.encapsulate(content, TrustLevel.RETRIEVED)
    out: Dict[str, Any] = {"content": tagged, "id": _get_id(chunk), "score": _get_score(chunk)}
    if isinstance(chunk, dict):
        out["metadata"] = chunk.get("metadata")
    elif hasattr(chunk, "metadata"):
        out["metadata"] = chunk.metadata
    return out


def _get_id(chunk: Any) -> Optional[str]:
    if isinstance(chunk, dict):
        return chunk.get("id")
    return getattr(chunk, "id", None)


def _get_score(chunk: Any) -> Optional[float]:
    if isinstance(chunk, dict):
        return chunk.get("score")
    return getattr(chunk, "score", None)


def sanitize_pinecone(
    raw_results: Any,
    *,
    engine: Optional[SessionUUIDEngine] = None,
    raise_on_high_risk: bool = False,
    risk_threshold: float = 0.8,
) -> LightShieldRAGResult:
    """Sanitize Pinecone query results at retrieval time.

    Scans every chunk (including metadata) for instruction-like patterns,
    quarantines suspicious chunks, and UUID-tags safe chunks with Level 3
    (retrieved) before they reach the LLM.

    Parameters
    ----------
    raw_results:
        Raw Pinecone query response (QueryResponse, dict with "matches", or
        list of matches).
    engine:
        Optional SessionUUIDEngine to reuse; creates a fresh one if omitted.
    raise_on_high_risk:
        If True and risk_score >= risk_threshold, raises
        LightShieldViolationException.
    risk_threshold:
        Threshold above which to raise when raise_on_high_risk=True.

    Returns
    -------
    LightShieldRAGResult
        safe_chunks (UUID-tagged), flagged_chunks, risk_score, engine.
    """

    if engine is None:
        engine = SessionUUIDEngine()

    matches = _extract_matches(raw_results)
    safe: List[Dict[str, Any]] = []
    flagged: List[FlaggedChunk] = []
    max_risk = 0.0

    for chunk in matches:
        flag = scan_chunk(chunk)
        if flag:
            flagged.append(flag)
            max_risk = max(max_risk, flag.risk_score)
        else:
            safe.append(_build_tagged_chunk(chunk, engine))

    total = len(matches)
    risk_score = max_risk if total > 0 else 0.0
    if flagged and total > 0:
        risk_score = max(risk_score, len(flagged) / total * 0.5 + max_risk * 0.5)

    result = LightShieldRAGResult(
        safe_chunks=safe,
        flagged_chunks=flagged,
        risk_score=min(1.0, risk_score),
        engine=engine,
    )

    if raise_on_high_risk and risk_score >= risk_threshold:
        first = flagged[0] if flagged else None
        raise LightShieldViolationException(
            message="LightShield quarantined high-risk RAG chunks.",
            violations=[f"rag_quarantine:{f.id or i}" for i, f in enumerate(flagged)],
            confidence=risk_score,
            violated_uuid_level=TrustLevel.RETRIEVED,
            triggering_pattern=first.triggering_patterns[0] if first else "high_risk",
            raw_content=first.content_preview if first else "",
            recommended_action="Review flagged_chunks and remove or redact malicious content.",
        )

    return result


__all__ = ["LightShieldRAGResult", "sanitize_pinecone"]
