
import asyncio
import json
import sys
from pathlib import Path

from benchmark.adapters.vllm import VLLMRequest
from benchmark.report import print_summary
from benchmark.vllm_runner import (
    run_vllm_requests,
    save_vllm_report,
    summarize_vllm,
)


async def main(config_path: str) -> None:

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    if config["total_requests"] <= 0:
        raise ValueError("total_requests must be positive")

    if config["max_tokens"] <= 0:
        raise ValueError("max_tokens must be positive")

    if config["concurrency"] <= 0:
        raise ValueError("concurrency must be positive")

    requests = [
        VLLMRequest(
            request_id=index,
            prompt=config["prompt"],
            max_tokens=config["max_tokens"],
        )
        for index in range(config["total_requests"])
    ]

    results, duration = await run_vllm_requests(
        requests=requests,
        base_url=config["base_url"],
        model=config["model"],
        concurrency=config["concurrency"],
        arrival_rate=config.get("arrival_rate"),
    )

    summary = summarize_vllm(
        results=results,
        duration_seconds=duration,
    )

    print_summary(summary)

    experiment_name = Path(config_path).stem

    output_stem = (
        Path("results/lab01") / experiment_name
    )

    save_vllm_report(
        results=results,
        summary=summary,
        config=config,
        output_stem=output_stem,
    )

    print()
    print(
        f"Saved: {output_stem}.csv"
    )
    print(
        f"Saved: {output_stem}-summary.json"
    )


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: poetry run python "
            "run_vllm_benchmark.py "
            "configs/lab01/smoke.json"
        )

    asyncio.run(main(sys.argv[1]))
