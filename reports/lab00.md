# Lab 0 — Workload and Benchmarking Harness

## Goal

Build a reproducible asynchronous benchmark client capable of
measuring streamed inference-like responses.

## System

Mock FastAPI inference server
+
async HTTPX benchmark client.

## Metrics

- TTFT
- TPOT
- ITL
- E2E latency
- requests/sec
- input tokens/sec
- output tokens/sec
- p50/p95/p99
- success rate

## Experiment 1 — Concurrency

Constant:
- 50 requests
- 128 input tokens
- 32 output tokens

Observed:
- throughput increased strongly with concurrency
- TPOT remained near 31 ms
- TTFT increased moderately

## Experiment 2 — Prompt length

Observed:
- TTFT increased with prompt length
- TPOT remained approximately constant

Interpretation:
- simulated prefill increased
- simulated decode cost stayed unchanged

## Experiment 3 — Output length

Observed:
- E2E increased with output length
- TTFT remained approximately constant

## Experiment 4 — Arrival rate

Observed:
...

## Experiment 5 — Mixed workload

Observed:
...

## Limitations

The mock server simulates computation using asyncio.sleep().
It therefore does not model finite GPU compute, HBM bandwidth,
KV-cache pressure or real inference scheduling.

## Conclusion

The benchmark harness is validated and ready to target a real
inference server in Lab 1.