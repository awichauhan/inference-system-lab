import asyncio
import json
import sys

from benchmark.client import run_requests
from benchmark.metrics import summarize
from benchmark.report import print_summary, save_results_csv
from benchmark.workloads import fixed_workload, mixed_workload


async def main(config_path: str):

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    workload_type = config["workload"]

    if workload_type == "fixed":

        requests = fixed_workload(
            total_requests=config["total_requests"],
            prompt_tokens=config["prompt_tokens"],
            output_tokens=config["output_tokens"],
        )

    elif workload_type == "mixed":

        requests = mixed_workload(
            total_requests=config["total_requests"],
            seed=config.get("seed", 42),
        )

    else:
        raise ValueError(
            f"Unknown workload: {workload_type}"
        )

    results, duration = await run_requests(
        requests=requests,
        base_url=config["base_url"],
        concurrency=config["concurrency"],
        arrival_rate=config.get("arrival_rate"),
    )

    summary = summarize(
        results=results,
        duration_seconds=duration,
    )

    print_summary(summary)

    output_path = (
        f"results/"
        f"{workload_type}"
        f"-c{config['concurrency']}.csv"
    )

    save_results_csv(
        results=results,
        output_path=output_path,
    )

    print()
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python run_benchmark.py configs/baseline.json"
        )

    asyncio.run(
        main(sys.argv[1])
    )