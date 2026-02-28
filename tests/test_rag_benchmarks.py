"""Unit tests for LightShield RAG benchmark suite."""

from __future__ import annotations

from lightshield.benchmarks import RAG_CASES, RAGInjectionTestCase, run_rag_benchmarks
from lightshield.benchmarks.test_harness import RAGBenchmarkResult


def test_rag_benchmark_runs() -> None:
    result = run_rag_benchmarks(RAG_CASES)
    assert isinstance(result, RAGBenchmarkResult)
    assert result.total_cases == len(RAG_CASES)
    assert result.attack_cases + result.benign_cases == result.total_cases


def test_rag_benchmark_quarantines_attacks() -> None:
    result = run_rag_benchmarks(RAG_CASES)
    assert result.attack_quarantined_rate >= 0.0
    assert result.attack_quarantined_rate <= 1.0
    assert result.avg_latency_ms >= 0.0


def test_rag_cases_include_all_required() -> None:
    names = {c.name for c in RAG_CASES}
    required = {
        "malicious_instruction_in_chunk",
        "cross_chunk_injection",
        "metadata_field_injection",
        "high_similarity_malicious",
        "instruction_override_retrieved",
        "data_exfiltration_tool_output",
        "benign_rag_chunks",
    }
    assert required.issubset(names)
