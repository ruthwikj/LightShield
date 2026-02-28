"""Benchmark harness for evaluating LightShield's effectiveness and latency."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List

from .. import LightShield
from ..adapters.openai_adapter import LightShieldResponse
from .injection_test_cases import CASES, InjectionTestCase


@dataclass
class BenchmarkResult:
    """Aggregated metrics from running the benchmark suite."""

    injection_success_rate: float
    false_positive_rate: float
    avg_latency_overhead_ms: float
    total_cases: int
    attack_cases: int
    benign_cases: int
    case_details: List[dict] = field(default_factory=list)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeResponse:
    def __init__(self, choices: List[_FakeChoice]) -> None:
        self.choices = choices


class _FakeChatCompletions:
    def create(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        messages = kwargs.get("messages") or []
        # Echo the last non-system message content as the "model" response.
        non_system = [m for m in messages if m.get("role") != "system"]
        content = non_system[-1]["content"] if non_system else ""
        return _FakeResponse([_FakeChoice(_FakeMessage(str(content)))])


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeChatCompletions()


class FakeOpenAIClient:
    """Minimal OpenAI-compatible client used for offline benchmarking."""

    def __init__(self) -> None:
        self.chat = _FakeChat()


def run_benchmarks(
    cases: Iterable[InjectionTestCase] = CASES,
) -> BenchmarkResult:
    """Run the canonical benchmark suite and return aggregate metrics.

    The harness uses a deterministic, in-process fake client to avoid network
    calls. Latency overhead is measured as the additional time spent when
    routing through :class:`lightshield.LightShield` compared to the raw client.
    """

    lightshield = LightShield()
    base_client = FakeOpenAIClient()
    wrapped_client = lightshield.wrap(base_client)

    attack_cases = 0
    benign_cases = 0
    attack_successes = 0
    false_positives = 0
    raw_latencies: List[float] = []
    wrapped_latencies: List[float] = []
    details: List[dict] = []

    for case in cases:
        is_attack = case.is_attack
        if is_attack:
            attack_cases += 1
        else:
            benign_cases += 1

        # Baseline call without LightShield.
        t0 = time.perf_counter()
        base_client.chat.completions.create(model="dummy", messages=case.messages)
        t1 = time.perf_counter()
        raw_lat = t1 - t0
        raw_latencies.append(raw_lat)

        # Call through LightShield-wrapped client.
        t2 = time.perf_counter()
        wrapped_resp: LightShieldResponse = wrapped_client.chat.completions.create(
            model="dummy", messages=case.messages
        )
        t3 = time.perf_counter()
        wrapped_lat = t3 - t2
        wrapped_latencies.append(wrapped_lat)

        is_safe = wrapped_resp.lightshield.is_safe

        if is_attack and is_safe:
            # Attacker "succeeded": LightShield did not flag it.
            attack_successes += 1
        if not is_attack and not is_safe:
            # Benign input flagged as unsafe.
            false_positives += 1

        details.append(
            {
                "name": case.name,
                "is_attack": is_attack,
                "is_safe": is_safe,
                "violations": list(wrapped_resp.lightshield.violations),
                "confidence": wrapped_resp.lightshield.confidence,
                "raw_latency_ms": raw_lat * 1000.0,
                "wrapped_latency_ms": wrapped_lat * 1000.0,
            }
        )

    total_cases = attack_cases + benign_cases
    attack_success_rate = (attack_successes / attack_cases) if attack_cases else 0.0
    false_positive_rate = (false_positives / benign_cases) if benign_cases else 0.0

    if raw_latencies and wrapped_latencies:
        avg_raw = statistics.fmean(raw_latencies)
        avg_wrapped = statistics.fmean(wrapped_latencies)
        overhead_ms = max(0.0, (avg_wrapped - avg_raw) * 1000.0)
    else:
        overhead_ms = 0.0

    return BenchmarkResult(
        injection_success_rate=attack_success_rate,
        false_positive_rate=false_positive_rate,
        avg_latency_overhead_ms=overhead_ms,
        total_cases=total_cases,
        attack_cases=attack_cases,
        benign_cases=benign_cases,
        case_details=details,
    )


__all__ = ["BenchmarkResult", "run_benchmarks", "FakeOpenAIClient"]
