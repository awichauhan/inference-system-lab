
import json

import httpx
import pytest

from benchmark.adapters.vllm import (
    VLLMRequest,
    parse_sse_line,
    send_vllm_request,
)


def make_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def test_sse_parser():
    event = parse_sse_line(
        'data: {"choices": []}'
    )

    assert event == {"choices": []}
    assert parse_sse_line("data: [DONE]") == "[DONE]"
    assert parse_sse_line("") is None


@pytest.mark.asyncio
async def test_vllm_streaming():

    def handler(request: httpx.Request):
        payload = json.loads(request.content)

        assert payload["stream"] is True
        assert payload["stream_options"]["include_usage"]
        assert payload["max_tokens"] == 8

        body = (
            make_event({
                "choices": [
                    {"delta": {"role": "assistant"}}
                ]
            })
            + make_event({
                "choices": [
                    {"delta": {"content": "Hello"}}
                ]
            })
            + make_event({
                "choices": [
                    {"delta": {"content": " world"}}
                ]
            })
            + make_event({
                "choices": [],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 2,
                    "total_tokens": 14
                }
            })
            + "data: [DONE]\n\n"
        )

        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        base_url="http://testserver",
        transport=transport,
    ) as client:

        result = await send_vllm_request(
            client=client,
            request=VLLMRequest(
                request_id=1,
                prompt="Say hello",
                max_tokens=8,
            ),
            model="test-model",
        )

    assert result.error is None
    assert result.status_code == 200

    assert result.prompt_tokens == 12
    assert result.completion_tokens == 2

    assert len(result.content_chunk_times) == 2

    assert result.ttft is not None
    assert result.e2e_latency >= result.ttft


@pytest.mark.asyncio
async def test_vllm_http_error():

    def handler(request: httpx.Request):
        return httpx.Response(
            500,
            json={"error": "server failure"},
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        base_url="http://testserver",
        transport=transport,
    ) as client:

        result = await send_vllm_request(
            client,
            VLLMRequest(2, "Hello", 8),
            "test-model",
        )

    assert result.status_code == 500
    assert result.error is not None
