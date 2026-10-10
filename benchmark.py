
"""Benchmark end-to-end Memory Orbs retrieval latency.

Run from the project root:
    python benchmark.py

Optional:
    MEMORY_BENCHMARK_RUNS=30 python benchmark.py
"""

import os
import statistics
import time

from store import init_db, get_active_memories
from retrieve import retrieve_memories

RUNS = int(os.environ.get("MEMORY_BENCHMARK_RUNS", "30"))
WARMUP_RUNS = 3

QUERIES = [
    "What operating system do I use?",
    "What are my current projects?",
    "What programming languages do I prefer?",
    "What are my learning goals?",
    "What tools do I use?",
]


def percentile(values, percentile_value):
    """Calculate a percentile using the nearest-rank method."""
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            int((percentile_value / 100) * len(ordered) + 0.999999) - 1,
        ),
    )
    return ordered[index]


def main():
    if RUNS < 1:
        raise ValueError("MEMORY_BENCHMARK_RUNS must be at least 1")

    init_db()
    memory_count = len(get_active_memories())

    if memory_count == 0:
        print("No active memories found.")
        print("Ingest your evaluation messages or sample memories first.")
        return

    print("Memory Orbs retrieval latency benchmark")
    print(f"Active memories: {memory_count}")
    print(f"Warm-up queries: {WARMUP_RUNS}")
    print(f"Timed runs: {RUNS}")
    print("Timing includes query embedding and memory retrieval.")
    print("")

    # Warm up the embedding model and retrieval path.
    for i in range(WARMUP_RUNS):
        retrieve_memories(QUERIES[i % len(QUERIES)])

    latencies_ms = []

    for i in range(RUNS):
        query = QUERIES[i % len(QUERIES)]

        start = time.perf_counter()
        retrieve_memories(query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies_ms.append(elapsed_ms)

    print("=== RESULTS ===")
    print(f"Queries measured: {len(latencies_ms)}")
    print(f"Median (p50):    {statistics.median(latencies_ms):.2f} ms")
    print(f"p95:             {percentile(latencies_ms, 95):.2f} ms")
    print(f"Mean:            {statistics.mean(latencies_ms):.2f} ms")
    print(f"Minimum:         {min(latencies_ms):.2f} ms")
    print(f"Maximum:         {max(latencies_ms):.2f} ms")
    print("")
    print("Note: Warm retrieval measurements on this machine.")
    print("Results depend on hardware, Ollama model state, and database size.")


if __name__ == "__main__":
    main()