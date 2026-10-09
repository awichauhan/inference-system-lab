import math
from dataclasses import dataclass

from benchmark.models import RequestResult


@dataclass
class BenchmarkSummary:
    total_requests: int
    successful_requests: int
    failed_requests: int

    duration_seconds: float

    requests_per_second: float
    input_tokens_per_second: float
    output_tokens_per_second: float

    mean_ttft_ms: float | None
    p50_ttft_ms: float | None
    p95_ttft_ms: float | None
    p99_ttft_ms: float | None

    mean_tpot_ms: float | None
    mean_e2e_ms: float | None

    success_rate: float


def percentile(
    values: list[float],
    percentile_value: float,
) -> float | None:

    if not values:
        return None

    sorted_values = sorted(values)

    index = math.ceil(
        percentile_value / 100 * len(sorted_values)
    ) - 1

    index = max(0, index)

    return sorted_values[index]


def mean(values: list[float]) -> float | None:

    if not values:
        return None

    return sum(values) / len(values)


def summarize(
    results: list[RequestResult],
    duration_seconds: float,
) -> BenchmarkSummary:

    successful = [
        result
        for result in results
        if result.error is None
    ]

    failed = [
        result
        for result in results
        if result.error is not None
    ]

    ttfts = [
        result.ttft
        for result in successful
        if result.ttft is not None
    ]

    tpots = [
        result.tpot
        for result in successful
        if result.tpot is not None
    ]

    e2e_latencies = [
        result.e2e_latency
        for result in successful
    ]

    total_input_tokens = sum(
        result.prompt_tokens
        for result in successful
    )

    total_output_tokens = sum(
        result.output_tokens
        for result in successful
    )

    total_requests = len(results)

    success_rate = (
        len(successful) / total_requests
        if total_requests
        else 0
    )

    return BenchmarkSummary(
        total_requests=total_requests,
        successful_requests=len(successful),
        failed_requests=len(failed),

        duration_seconds=duration_seconds,

        requests_per_second=(
            len(successful) / duration_seconds
        ),

        input_tokens_per_second=(
            total_input_tokens / duration_seconds
        ),

        output_tokens_per_second=(
            total_output_tokens / duration_seconds
        ),

        mean_ttft_ms=to_ms(mean(ttfts)),
        p50_ttft_ms=to_ms(percentile(ttfts, 50)),
        p95_ttft_ms=to_ms(percentile(ttfts, 95)),
        p99_ttft_ms=to_ms(percentile(ttfts, 99)),

        mean_tpot_ms=to_ms(mean(tpots)),
        mean_e2e_ms=to_ms(mean(e2e_latencies)),

        success_rate=success_rate,
    )


def to_ms(value: float | None) -> float | None:
    if value is None:
        return None

    return value * 1000