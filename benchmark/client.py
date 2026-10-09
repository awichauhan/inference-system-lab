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

async def run_requests(
    requests: list[RequestSpec],
    base_url: str,
    concurrency: int,
    arrival_rate: float | None = None,
) -> tuple[list[RequestResult], float]:

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=None,
    ) as client:

        async def controlled_request(
            request: RequestSpec,
            delay: float,
        ) -> RequestResult:

            if delay > 0:
                await asyncio.sleep(delay)

            async with semaphore:
                return await send_request(
                    client,
                    request,
                )

        tasks = []

        for index, request in enumerate(requests):

            if arrival_rate is None:
                delay = 0
            else:
                delay = index / arrival_rate

            task = asyncio.create_task(
                controlled_request(
                    request,
                    delay,
                )
            )

            tasks.append(task)

        benchmark_start = time.perf_counter()

        results = await asyncio.gather(*tasks)

        benchmark_end = time.perf_counter()

    total_duration = (
        benchmark_end - benchmark_start
    )

    return results, total_duration



