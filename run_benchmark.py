import asyncio
import json
import sys

from benchmark.client import run_requests
from benchmark.metrics import summarize
from benchmark.report import print_summary, save_results_csv
from benchmark.workloads import fixed_workload, mixed_workload


async def main(config_path: str):

    # -----------------------------
    # 1. Load experiment config
    # -----------------------------
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    # -----------------------------
    # 2. Basic config validation
    # -----------------------------
    if config["total_requests"] <= 0:
        raise ValueError("total_requests must be greater than 0")

    if config["concurrency"] <= 0:
        raise ValueError("concurrency must be greater than 0")

    if config["concurrency"] > config["total_requests"]:
        print(
            "Warning: concurrency is greater than total_requests. "
            "Effective concurrency cannot exceed request count."
        )

    # -----------------------------
    # 3. Build workload
    # -----------------------------
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
            f"Unknown workload type: {workload_type}"
        )

    # -----------------------------
    # 4. Run benchmark
    # -----------------------------
    results, duration = await run_requests(
        requests=requests,
        base_url=config["base_url"],
        concurrency=config["concurrency"],
        arrival_rate=config.get("arrival_rate"),
    )

    # -----------------------------
    # 5. Calculate metrics
    # -----------------------------
    summary = summarize(
        results=results,
        duration_seconds=duration,
    )

    # -----------------------------
    # 6. Print benchmark summary
    # -----------------------------
    print_summary(summary)

    # -----------------------------
    # 7. Build unique result filename
    # -----------------------------

    # Prefer explicit experiment name.
    #
    # Example:
    # "name": "prompt-512"
    #
    # becomes:
    # results/prompt-512.csv

    experiment_name = config.get("name")

    # Backward compatibility for older configs
    # that do not yet contain "name".
    if experiment_name is None:
        experiment_name = (
            f"{workload_type}"
            f"-c{config['concurrency']}"
        )

    output_path = (
        f"results/{experiment_name}.csv"
    )

    # -----------------------------
    # 8. Save per-request results
    # -----------------------------
    save_results_csv(
        results=results,
        output_path=output_path,
    )

    print()
    print(
        f"Saved results to: {output_path}"
    )


# ---------------------------------
# Program entry point
# ---------------------------------
if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: "
            "python run_benchmark.py "
            "configs/<config-file>.json"
        )

    asyncio.run(
        main(sys.argv[1])
    )