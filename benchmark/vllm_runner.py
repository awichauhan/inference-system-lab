
import csv
import json
from dataclasses import asdict
from pathlib import Path

import httpx

from benchmark.adapters.vllm import (
    VLLMRequest,
    VLLMResult,
    send_vllm_request,
)
from benchmark.client import run_with_concurrency
from benchmark.metrics import (
    BenchmarkSummary,
    mean,
    percentile,
    to_ms,
)


async def run_vllm_requests(
    requests: list[VLLMRequest],
    base_url: str,
    model: str,
    concurrency: int,
    arrival_rate: float | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[VLLMResult], float]:

    async def send(
        client: httpx.AsyncClient,
        request: VLLMRequest,
    ) -> VLLMResult:

        return await send_vllm_request(
            client,
            request,
            model,
        )

    return await run_with_concurrency(
        requests=requests,
        base_url=base_url,
        concurrency=concurrency,
        send=send,
        arrival_rate=arrival_rate,
        transport=transport,
    )


def summarize_vllm(
    results: list[VLLMResult],
    duration_seconds: float,
) -> BenchmarkSummary:

    if duration_seconds <= 0:
        raise ValueError(
            "duration_seconds must be positive"
        )

    successful = [
        result
        for result in results
        if result.error is None
        and result.status_code == 200
        and result.prompt_tokens is not None
        and result.completion_tokens is not None
    ]

    ttfts = [
        result.ttft
        for result in successful
        if result.ttft is not None
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
        result.completion_tokens
        for result in successful
    )

    return BenchmarkSummary(
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(results) - len(successful),

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

        # SSE chunks are not necessarily individual tokens.
        mean_tpot_ms=None,

        mean_e2e_ms=to_ms(mean(e2e_latencies)),

        success_rate=(
            len(successful) / len(results)
            if results else 0.0
        ),
    )


def save_vllm_report(
    results: list[VLLMResult],
    summary: BenchmarkSummary,
    config: dict,
    output_stem: Path,
) -> None:

    output_stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_stem.with_suffix(".csv")

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "request_id",
                "prompt_tokens",
                "completion_tokens",
                "content_chunks",
                "ttft_ms",
                "e2e_ms",
                "status_code",
                "error",
            ],
        )

        writer.writeheader()

        for result in results:

            writer.writerow({
                "request_id": result.request_id,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "content_chunks": len(
                    result.content_chunk_times
                ),
                "ttft_ms": to_ms(result.ttft),
                "e2e_ms": to_ms(result.e2e_latency),
                "status_code": result.status_code,
                "error": result.error,
            })

    json_path = output_stem.with_name(
        output_stem.name + "-summary.json"
    )

    metadata = {
        "config": config,
        "summary": asdict(summary),
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)
