# Lab 0 — Workload and Benchmarking Harness

## Goal

Build a reproducible asynchronous benchmark harness capable of measuring streamed inference-like responses before introducing a real LLM, GPU, or inference engine.

Lab 0 establishes the measurement layer that later labs can reuse against real inference runtimes such as vLLM.

---

## System Architecture

Lab 0 uses two local processes:

```text
Benchmark Client
HTTPX + asyncio
        │
        │ HTTP POST /generate
        ▼
FastAPI Mock Inference Server
        │
        ├── Simulated Prefill
        │
        └── Streamed Decode Tokens
```

The mock server intentionally uses deterministic timing:

```text
Prefill delay = prompt_tokens × 0.1 ms
Decode delay  = 30 ms per generated token
```

Responses are streamed token by token.

This gives the benchmark client a predictable target against which measurement correctness can be validated.

---

## Components

### Mock Server

The FastAPI server provides:

```text
GET  /health
POST /generate
```

`/generate` accepts:

```text
prompt_tokens
output_tokens
priority
```

The server simulates prefill using:

```python
prefill_seconds = request.prompt_tokens * 0.0001
```

and decoding using:

```python
decode_seconds = 0.030
```

Tokens are produced through an async generator and returned with `StreamingResponse`.

### Benchmark Client

The benchmark client uses:

- `httpx.AsyncClient`
- HTTP response streaming
- `asyncio.create_task()`
- `asyncio.gather()`
- `asyncio.Semaphore`
- async iteration over streamed tokens
- `time.perf_counter()`

For every request it records:

```text
request start
first token arrival
every token arrival
request end
HTTP status
error state
```

These timestamps are used to derive inference-serving metrics.

---

## Metrics

The harness calculates:

| Metric | Meaning |
|---|---|
| TTFT | Time To First Token |
| TPOT | Time Per Output Token after the first token |
| ITL | Inter-Token Latency |
| E2E | End-to-end request latency |
| Requests/sec | Completed requests per second |
| Input tokens/sec | Prompt tokens processed per second |
| Output tokens/sec | Generated tokens per second |
| p50 TTFT | Median first-token latency |
| p95 TTFT | 95th percentile first-token latency |
| p99 TTFT | 99th percentile first-token latency |
| Success rate | Percentage of requests completed without errors |

The committed CSV files persist per-request:

```text
request_id
prompt_tokens
output_tokens
ttft_ms
tpot_ms
e2e_ms
status_code
error
```

All committed Lab 0 result files contain successful HTTP 200 responses with no recorded request failures.

> Note: aggregate throughput values such as requests/sec and tokens/sec are printed during runtime but are not currently persisted inside the CSV artifacts. The analysis below therefore uses only values that can be reconstructed directly from the committed result files.

---

# Experiment 1 — Concurrency Sweep

## Goal

Observe how request latency changes when more requests are allowed to execute concurrently.

Constant workload:

```text
Total requests = 50
Prompt tokens  = 128
Output tokens  = 32
Arrival rate   = unrestricted
```

Only concurrency was changed.

## Results

| Concurrency | Mean TTFT | p50 TTFT | p95 TTFT | p99 TTFT | Mean TPOT | Mean E2E |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49.85 ms | 48.39 ms | 57.14 ms | 83.73 ms | 31.28 ms | 1021.21 ms |
| 4 | 49.68 ms | 48.76 ms | 52.05 ms | 76.55 ms | 31.22 ms | 1018.56 ms |
| 8 | 53.56 ms | 52.96 ms | 58.28 ms | 75.86 ms | 31.13 ms | 1020.91 ms |
| 16 | 58.81 ms | 54.99 ms | 77.92 ms | 83.86 ms | 31.07 ms | 1024.99 ms |
| 32 | 67.56 ms | 62.37 ms | 89.79 ms | 90.64 ms | 31.18 ms | 1039.74 ms |
| 64* | 69.33 ms | 68.63 ms | 72.47 ms | 95.78 ms | 31.61 ms | 1055.44 ms |

\*The workload contains only 50 requests, so effective active concurrency cannot exceed 50 even when the configured limit is 64.

Result files:

```text
results/concurrency-c1.csv
results/concurrency-c4.csv
results/concurrency-c8.csv
results/concurrency-c16.csv
results/concurrency-c32.csv
results/concurrency-c64.csv
```

## Observation

Mean TTFT increased from approximately `49.85 ms` at concurrency 1 to `69.33 ms` at the highest configured concurrency.

TPOT remained almost constant at approximately `31 ms/token` throughout the experiment, while E2E latency increased only moderately.

This behavior is expected because the mock server simulates computation using:

```python
await asyncio.sleep(...)
```

While one coroutine is sleeping, the event loop can continue progressing other requests. Multiple requests can therefore overlap without competing for a finite compute resource.

Conceptually:

```text
Concurrency 1

R1 ==============================
                                R2 ==============================

Higher concurrency

R1 ==============================
R2 ==============================
R3 ==============================
R4 ==============================
```

This experiment validates async concurrency, semaphore-based concurrency limits, request scheduling, streaming measurement, and latency behavior under increased runtime activity. It does not model GPU saturation.

### Tail-Latency Note

Each concurrency experiment contains only 50 requests. The percentile implementation uses nearest-rank selection.

For p99:

```text
0.99 × 50 = 49.5
```

which effectively selects the slowest observation. The p99 values are therefore useful for learning tail latency, but should not be interpreted as statistically stable production p99 values.

---

# Experiment 2 — Prompt Length vs TTFT

## Goal

Measure how increasing prompt length affects first-token latency while keeping output generation constant.

Constant workload:

```text
Total requests = 50
Concurrency    = 8
Output tokens  = 32
```

Prompt length was varied across `128`, `512`, `2048`, and `4096` tokens.

## Results

| Prompt Tokens | Mean TTFT | p50 TTFT | p95 TTFT | p99 TTFT | Mean TPOT | Mean E2E |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 54.02 ms | 53.35 ms | 57.49 ms | 76.66 ms | 31.17 ms | 1022.43 ms |
| 512 | 94.27 ms | 93.06 ms | 100.06 ms | 124.01 ms | 31.25 ms | 1065.28 ms |
| 2048 | 250.16 ms | 246.36 ms | 277.61 ms | 282.15 ms | 31.30 ms | 1222.98 ms |
| 4096 | 452.79 ms | 452.17 ms | 456.27 ms | 480.07 ms | 31.18 ms | 1421.78 ms |

Result files:

```text
results/prompt-128.csv
results/prompt-512.csv
results/prompt-2048.csv
results/prompt-4096.csv
```

## Observation

TTFT increased strongly with prompt length:

```text
128 tokens   → ~54 ms
512 tokens   → ~94 ms
2048 tokens  → ~250 ms
4096 tokens  → ~453 ms
```

Meanwhile TPOT stayed approximately constant at `~31 ms/token`.

This matches the server implementation:

```python
prefill_seconds = request.prompt_tokens * 0.0001
```

Longer prompts therefore create more simulated prefill work before the first token can be generated.

```text
Prompt length ↑
       ↓
Prefill work ↑
       ↓
TTFT ↑
```

This demonstrates the core distinction that prefill strongly affects TTFT, whereas decode behavior dominates token-generation time.

---

# Experiment 3 — Output Length vs E2E Latency

## Goal

Measure how requested generation length affects total request latency.

Constant workload:

```text
Total requests = 50
Prompt tokens  = 128
Concurrency    = 8
```

Output length was varied across `16`, `32`, `64`, and `128` tokens.

## Results

| Output Tokens | Mean TTFT | p50 TTFT | p95 TTFT | p99 TTFT | Mean TPOT | Mean E2E |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 53.63 ms | 52.69 ms | 57.20 ms | 77.04 ms | 31.09 ms | 522.18 ms |
| 32 | 53.41 ms | 52.92 ms | 57.29 ms | 74.02 ms | 31.11 ms | 1019.81 ms |
| 64 | 53.60 ms | 53.10 ms | 57.45 ms | 74.26 ms | 31.11 ms | 2015.40 ms |
| 128 | 53.47 ms | 53.18 ms | 57.69 ms | 72.46 ms | 31.12 ms | 4008.02 ms |

Result files:

```text
results/output-16.csv
results/output-32.csv
results/output-64.csv
results/output-128.csv
```

## Observation

TTFT remained almost unchanged at `~53–54 ms`, while TPOT remained stable at `~31.1 ms/token`.

E2E latency increased almost linearly:

```text
16 tokens   → ~522 ms
32 tokens   → ~1020 ms
64 tokens   → ~2015 ms
128 tokens  → ~4008 ms
```

The server performs one decode wait for every generated token:

```python
for token_id in range(request.output_tokens):
    await asyncio.sleep(decode_seconds)
```

Therefore:

```text
Output tokens ↑
       ↓
Decode iterations ↑
       ↓
E2E latency ↑
```

This experiment clearly separates TTFT, TPOT, and total E2E latency.

---

# Experiment 4 — Arrival-Rate Sweep

## Goal

Introduce request-arrival pacing instead of launching the entire workload at once.

Constant workload:

```text
Total requests = 100
Prompt tokens  = 128
Output tokens  = 32
Concurrency    = 16
```

Arrival rate was varied across `2`, `5`, `10`, and `20` requests/sec.

The client implements deterministic request pacing using:

```python
delay = index / arrival_rate
```

For example, at `5 req/s`:

```text
R0 → 0.0 s
R1 → 0.2 s
R2 → 0.4 s
R3 → 0.6 s
...
```

## Results

| Arrival Rate | Mean TTFT | p50 TTFT | p95 TTFT | p99 TTFT | Mean TPOT | Mean E2E |
|---:|---:|---:|---:|---:|---:|---:|
| 2 req/s | 48.92 ms | 48.83 ms | 50.20 ms | 51.36 ms | 31.05 ms | 1012.19 ms |
| 5 req/s | 49.72 ms | 49.54 ms | 51.90 ms | 64.31 ms | 31.03 ms | 1012.64 ms |
| 10 req/s | 49.93 ms | 48.75 ms | 53.48 ms | 75.34 ms | 30.89 ms | 1008.82 ms |
| 20 req/s | 49.12 ms | 47.57 ms | 57.55 ms | 74.06 ms | 30.97 ms | 1010.46 ms |

Result files:

```text
results/arrival-2rps.csv
results/arrival-5rps.csv
results/arrival-10rps.csv
results/arrival-20rps.csv
```

## Observation

Mean TTFT remained close to `~49–50 ms`, and TPOT remained near `~31 ms/token` even as the configured arrival rate increased.

Tail TTFT became somewhat more variable at higher arrival rates, but the system did not experience a sustained latency explosion.

This is expected because the mock server has no finite compute resource. Its simulated model work uses `await asyncio.sleep(...)`, allowing requests to overlap cheaply.

The experiment therefore validates arrival pacing, request scheduling, traffic generation, and interaction between pacing and bounded concurrency, but it does not demonstrate true queue buildup under GPU saturation.

---

# Experiment 5 — Mixed Workload

## Goal

Move from uniform synthetic traffic to heterogeneous request shapes.

Configuration:

```text
Total requests = 100
Concurrency    = 16
Random seed    = 42
```

Possible prompt lengths:

```text
128
512
2048
4096
```

Possible output lengths:

```text
32
64
128
256
```

Possible priorities:

```text
0
1
2
```

The fixed seed makes the generated workload reproducible.

## Actual Prompt Distribution

| Prompt Tokens | Requests |
|---:|---:|
| 128 | 28 |
| 512 | 28 |
| 2048 | 20 |
| 4096 | 24 |

## Actual Output Distribution

| Output Tokens | Requests |
|---:|---:|
| 32 | 31 |
| 64 | 27 |
| 128 | 22 |
| 256 | 20 |

## Aggregate Results

| Metric | Value |
|---|---:|
| Successful Requests | 100 / 100 |
| Mean TTFT | 195.30 ms |
| p50 TTFT | 89.60 ms |
| p95 TTFT | 447.69 ms |
| p99 TTFT | 458.75 ms |
| Mean TPOT | 30.93 ms |
| Mean E2E | 3460.19 ms |

Result file:

```text
results/mixed-c16.csv
```

## Observation

The mixed workload produced a much broader latency distribution.

```text
Mean TTFT = 195.30 ms
p50       = 89.60 ms
p95       = 447.69 ms
p99       = 458.75 ms
```

This spread appears because requests no longer have identical prompt sizes. Long prompts increase TTFT, while long outputs significantly increase E2E latency.

TPOT still remained close to `~31 ms/token` because the simulated decode delay is fixed.

This experiment introduces an important scheduler concept: real inference servers must handle heterogeneous short and long requests simultaneously, so a scheduler cannot assume that every request has the same computational cost.

---

# Validation

The repository contains tests for the core metric calculations.

## Percentile Tests

`tests/test_metrics.py` validates p50 and p95 using known ordered values.

## Request Metric Tests

`tests/test_models.py` validates TTFT, TPOT, ITL, and E2E latency using deterministic timestamps.

Example synthetic request:

```text
start = 10.0

token timestamps:
10.2
10.3
10.4
10.5

end = 10.5
```

Expected:

```text
TTFT = 0.2 sec
TPOT = 0.1 sec
ITL  = [0.1, 0.1, 0.1]
E2E  = 0.5 sec
```

---

# Measurement Sanity Check

The mock server intentionally uses:

```python
decode_seconds = 0.030
```

Therefore theoretical token spacing is `30 ms`.

Across the experiments, measured TPOT remained close to `~31 ms`:

```text
Concurrency sweep: ~31.1–31.6 ms
Prompt sweep:      ~31.2–31.3 ms
Output sweep:      ~31.1 ms
Arrival sweep:     ~30.9–31.1 ms
Mixed workload:    ~30.93 ms
```

The small difference from the configured 30 ms delay represents runtime overhead from Python, asyncio scheduling, HTTP streaming, FastAPI, Uvicorn, and operating-system scheduling.

This provides a useful sanity check that the benchmark instrumentation is measuring streamed token arrivals correctly.

---

# Python Concepts Used

| Concept | Where It Appears |
|---|---|
| Modules and packages | `benchmark`, `mock_server`, `tests` |
| Dataclasses | `RequestSpec`, `RequestResult`, `BenchmarkSummary` |
| Type hints | request/result definitions |
| Properties | TTFT, TPOT, ITL, E2E calculations |
| `async def` / `await` | server and client |
| Coroutines | HTTP requests |
| Async tasks | concurrent request execution |
| Async iteration | streamed token reading |
| Async generators | mock token generation |
| Async context managers | HTTP streaming lifecycle |
| Semaphore | bounded concurrency |
| Exception handling | per-request failure capture |
| `time.perf_counter()` | latency measurement |
| JSON | experiment configuration |
| CSV | result persistence |
| Random seed | reproducible mixed workload |
| pytest | metric validation |

---

# DSA and Algorithm Concepts Used

| Concept | Usage |
|---|---|
| Dynamic arrays / lists | requests, results, token timestamps |
| Queueing concept | requests waiting for semaphore capacity |
| Linear scan | metric aggregation |
| Sorting | percentile calculation |
| Dictionary / hash-map lookup | parsed JSON configuration |
| State modelling | request/result dataclasses |
| Sequential timestamp differences | ITL calculation |

Relevant complexities:

```text
Appending timestamp:        amortized O(1)
Scanning results:           O(n)
Summing metrics:            O(n)
Sorting percentile data:    O(n log n)
Percentile lookup afterward: O(1)
```

---

# Key Systems Concepts Covered

Lab 0 introduced:

```text
client-server architecture
HTTP streaming
async I/O
event loops
concurrency
bounded concurrency
arrival rate
request pacing
latency
throughput
TTFT
TPOT
ITL
E2E
tail latency
p50 / p95 / p99
controlled experiments
reproducibility
fixed vs mixed workloads
prefill vs decode
```

---

# Limitations

Lab 0 is intentionally a measurement lab. It is not a GPU simulator.

The current mock server does not model:

```text
real neural-network execution
GPU compute saturation
GPU memory bandwidth
HBM pressure
KV-cache allocation
KV-cache fragmentation
continuous batching
inference-engine scheduling
CUDA kernels
tensor parallelism
pipeline parallelism
expert parallelism
NCCL communication
multi-GPU topology
production network latency
```

The largest limitation is:

```python
await asyncio.sleep(...)
```

Simulated requests spend most of their time waiting rather than consuming a finite compute resource. High concurrency therefore does not produce the same resource contention that a real GPU inference workload would experience.

---

# Instrumentation Improvements for Later Labs

The current per-request CSV files preserve latency measurements successfully, but run-level values such as benchmark duration and aggregate throughput are not persisted directly.

These values are printed during execution but are not currently written into a result artifact.

Later labs should consider storing:

```text
run metadata
configuration
benchmark duration
requests/sec
input tokens/sec
output tokens/sec
latency summary
hardware information
model information
engine configuration
```

in a machine-readable file such as:

```text
results/<experiment-name>-summary.json
```

The full ITL series is also calculated in memory but is not currently persisted in the CSV schema. This can be extended when deeper token-level analysis becomes useful.

---

# Lab 0 Conclusions

The benchmark harness successfully demonstrated the expected relationships.

```text
Concurrency ↑
    ↓
more overlapping async work
    ↓
TTFT overhead increases moderately
while TPOT remains stable
```

```text
Prompt length ↑
    ↓
simulated prefill ↑
    ↓
TTFT ↑
```

```text
Output length ↑
    ↓
decode iterations ↑
    ↓
E2E latency ↑
```

```text
Arrival rate ↑
    ↓
request pacing changes
but no real compute saturation occurs
because the mock server uses async sleep
```

```text
Mixed request shapes
    ↓
broader latency distribution
    ↓
scheduler-oriented reasoning becomes necessary
```

The benchmark infrastructure now supports:

```text
reproducible workloads
concurrent streaming requests
arrival-rate control
per-token timestamps
latency metrics
tail percentiles
error tracking
CSV result persistence
unit-tested metric calculations
```

---

# Lab 0 Status

**COMPLETE**

The benchmark client and measurement concepts established here can now be reused against a real inference engine.

---

# Next — Lab 1

Lab 1 replaces:

```text
Benchmark Harness
        ↓
Mock FastAPI Server
```

with:

```text
Benchmark Harness
        ↓
Real Inference Runtime
        ↓
LLM
        ↓
NVIDIA GPU
```

The major new topics will include:

```text
real model serving
GPU utilization
real prefill/decode behavior
throughput saturation
real latency/throughput trade-offs
vLLM
KV-cache behavior
GPU memory constraints
```

The benchmark harness created in Lab 0 becomes the measuring instrument for those experiments.
