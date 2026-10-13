
import json
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class VLLMRequest:
    request_id: int
    prompt: str
    max_tokens: int = 32


@dataclass
class VLLMResult:
    request_id: int
    start_time: float
    end_time: float = 0.0

    first_content_time: float | None = None

    content_chunk_times: list[float] = field(
        default_factory=list
    )

    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    status_code: int | None = None
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        if self.first_content_time is None:
            return None

        return self.first_content_time - self.start_time

    @property
    def e2e_latency(self) -> float:
        return self.end_time - self.start_time

    @property
    def chunk_intervals(self) -> list[float]:
        times = self.content_chunk_times

        return [
            times[i] - times[i - 1]
            for i in range(1, len(times))
        ]


def parse_sse_line(line: str) -> dict | str | None:
    if not line.startswith("data:"):
        return None

    payload = line[5:].strip()

    if payload == "[DONE]":
        return "[DONE]"

    if not payload:
        return None

    return json.loads(payload)


async def send_vllm_request(
    client: httpx.AsyncClient,
    request: VLLMRequest,
    model: str,
) -> VLLMResult:

    if request.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    result = VLLMResult(
        request_id=request.request_id,
        start_time=time.perf_counter(),
    )

    done_received = False

    try:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": request.prompt,
                    }
                ],
                "max_tokens": request.max_tokens,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {
                    "include_usage": True
                },
            },
        ) as response:

            result.status_code = response.status_code
            response.raise_for_status()

            async for line in response.aiter_lines():

                event = parse_sse_line(line)

                if event is None:
                    continue

                if event == "[DONE]":
                    done_received = True
                    break

                usage = event.get("usage")

                if isinstance(usage, dict):
                    result.prompt_tokens = usage.get(
                        "prompt_tokens"
                    )
                    result.completion_tokens = usage.get(
                        "completion_tokens"
                    )

                for choice in event.get("choices", []):

                    delta = choice.get("delta") or {}
                    content = delta.get("content")

                    if not content:
                        continue

                    now = time.perf_counter()

                    if result.first_content_time is None:
                        result.first_content_time = now

                    result.content_chunk_times.append(now)

            if not done_received:
                raise ValueError(
                    "Streaming response ended without [DONE]"
                )

            if result.prompt_tokens is None:
                raise ValueError(
                    "Missing prompt-token usage"
                )

            if result.completion_tokens is None:
                raise ValueError(
                    "Missing completion-token usage"
                )

            if result.first_content_time is None:
                raise ValueError(
                    "No visible generated content received"
                )

    except Exception as exc:
        result.error = str(exc)

    finally:
        result.end_time = time.perf_counter()

    return result
