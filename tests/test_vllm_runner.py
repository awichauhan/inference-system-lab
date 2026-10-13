
import asyncio
import json

import httpx
import pytest

from benchmark.adapters.vllm import VLLMRequest
from benchmark.vllm_runner import (
    run_vllm_requests,
    save_vllm_report,
    summarize_vllm,
)


def event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@pytest.mark.asyncio
async def test_vllm_runner_integration(tmp_path):

    received = []

    active = 0
    peak_active = 0

    async def handler(request: httpx.Request):

        nonlocal active, peak_active

        active += 1
        peak_active = max(peak_active, active)

        received.append(json.loads(request.content))

        await asyncio.sleep(0.01)

        active -= 1

        body = (
            event({
                "choices": [
                    {"delta": {"role": "assistant"}}
                ]
            })
            + event({
                "choices": [
                    {"delta": {"content": "Hello"}}
                ]
            })
            + event({
                "choices": [
                    {"delta": {"content": " world"}}
                ]
            })
            + event({
                "choices": [],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "total_tokens": 9,
                }
            })
            + "data: [DONE]\n\n"
        )

        return httpx.Response(
            200,
            text=body,
            headers={
                "content-type": "text/event-stream"
            },
        )

    requests = [
        VLLMRequest(i, "Hello", 8)
        for i in range(4)
    ]

    results, duration = await run_vllm_requests(
        requests=requests,
        base_url="http://testserver",
        model="test-model",
        concurrency=2,
        transport=httpx.MockTransport(handler),
    )

    summary = summarize_vllm(results, duration)

    assert len(received) == 4
    assert len(results) == 4

    # Verify bounded concurrency.
    assert peak_active == 2

    # Verify request results.
    assert summary.successful_requests == 4
    assert summary.failed_requests == 0

    assert all(
        result.completion_tokens == 2
        for result in results
    )

    # Server-reported tokens drive throughput.
    assert summary.input_tokens_per_second > 0
    assert summary.output_tokens_per_second > 0

    # Text chunks are not guaranteed to be tokens.
    assert summary.mean_tpot_ms is None

    output_stem = tmp_path / "smoke"

    save_vllm_report(
        results=results,
        summary=summary,
        config={"model": "test-model"},
        output_stem=output_stem,
    )

    assert (tmp_path / "smoke.csv").exists()

    json_path = tmp_path / "smoke-summary.json"

    assert json_path.exists()

    saved = json.loads(json_path.read_text())

    assert saved["summary"]["successful_requests"] == 4
