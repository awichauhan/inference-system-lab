import csv
from pathlib import Path

from benchmark.metrics import BenchmarkSummary
from benchmark.models import RequestResult


def print_summary(summary: BenchmarkSummary) -> None:

    print()
    print("========== Benchmark Summary ==========")

    print(f"Requests:             {summary.total_requests}")
    print(f"Successful:           {summary.successful_requests}")
    print(f"Failed:               {summary.failed_requests}")

    print()

    print(
        f"Duration:             "
        f"{summary.duration_seconds:.2f} sec"
    )

    print(
        f"Requests/sec:         "
        f"{summary.requests_per_second:.2f}"
    )

    print(
        f"Input tokens/sec:     "
        f"{summary.input_tokens_per_second:.2f}"
    )

    print(
        f"Output tokens/sec:    "
        f"{summary.output_tokens_per_second:.2f}"
    )

    print()

    print_metric("Mean TTFT", summary.mean_ttft_ms)
    print_metric("p50 TTFT", summary.p50_ttft_ms)
    print_metric("p95 TTFT", summary.p95_ttft_ms)
    print_metric("p99 TTFT", summary.p99_ttft_ms)

    print()

    print_metric("Mean TPOT", summary.mean_tpot_ms)
    print_metric("Mean E2E", summary.mean_e2e_ms)

    print()

    print(
        f"Success rate:         "
        f"{summary.success_rate * 100:.2f}%"
    )


def print_metric(name: str, value: float | None) -> None:

    if value is None:
        print(f"{name:<20} N/A")
    else:
        print(f"{name:<20} {value:.2f} ms")


def save_results_csv(
    results: list[RequestResult],
    output_path: str,
) -> None:

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "request_id",
                "prompt_tokens",
                "output_tokens",
                "ttft_ms",
                "tpot_ms",
                "e2e_ms",
                "status_code",
                "error",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result.request_id,
                    result.prompt_tokens,
                    result.output_tokens,
                    optional_ms(result.ttft),
                    optional_ms(result.tpot),
                    result.e2e_latency * 1000,
                    result.status_code,
                    result.error,
                ]
            )


def optional_ms(value: float | None):
    if value is None:
        return None

    return value * 1000