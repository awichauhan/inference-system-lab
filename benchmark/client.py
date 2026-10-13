from collections.abc import Awaitable, Callable
from typing import TypeVar

import asyncio
import time

import httpx

from benchmark.models import RequestResult, RequestSpec

async def send_request(
    client: httpx.AsyncClient,
    request: RequestSpec,
) -> RequestResult:
    start_time = time.perf_counter()
    first_token_time = None
    token_times = []

    status_code = None
    error = None

    try:
        async with client.stream(
                "POST",
                "/generate",
                json={
                    "prompt_tokens": request.prompt_tokens,
                    "output_tokens": request.output_tokens,
                    "priority": request.priority,
                },
        ) as response:
            status_code = response.status_code
            response.raise_for_status()

            async for line in response.aiter_lines():

                if not line:
                    continue

                now = time.perf_counter()

                if first_token_time is None:
                    first_token_time = now

                token_times.append(now)

    except Exception as exc:
        error = str(exc)

    end_time = time.perf_counter()

    return RequestResult(
        request_id=request.request_id,
        prompt_tokens=request.prompt_tokens,
        requested_output_tokens=request.output_tokens,
        start_time=start_time,
        first_token_time=first_token_time,
        end_time=end_time,
        token_times=token_times,
        status_code=status_code,
        error=error,
    )

TRequest = TypeVar("TRequest")
TResult = TypeVar("TResult")


async def run_with_concurrency(
    requests: list[TRequest],
    base_url: str,
    concurrency: int,
    send: Callable[
        [httpx.AsyncClient, TRequest],
        Awaitable[TResult],
    ],
    arrival_rate: float | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[TResult], float]:

    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    if arrival_rate is not None and arrival_rate <= 0:
        raise ValueError("arrival_rate must be positive")

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=None,
        transport=transport,
    ) as client:

        async def controlled_request(
            request: TRequest,
            delay: float,
        ) -> TResult:

            if delay > 0:
                await asyncio.sleep(delay)

            async with semaphore:
                return await send(client, request)

        benchmark_start = time.perf_counter()

        tasks = [
            asyncio.create_task(
                controlled_request(
                    request,
                    0.0 if arrival_rate is None
                    else index / arrival_rate,
                )
            )
            for index, request in enumerate(requests)
        ]

        results = await asyncio.gather(*tasks)

        duration = time.perf_counter() - benchmark_start

    return results, duration


async def run_requests(
    requests: list[RequestSpec],
    base_url: str,
    concurrency: int,
    arrival_rate: float | None = None,
) -> tuple[list[RequestResult], float]:

    return await run_with_concurrency(
        requests=requests,
        base_url=base_url,
        concurrency=concurrency,
        send=send_request,
        arrival_rate=arrival_rate,
    )

