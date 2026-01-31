#!/usr/bin/env python3
"""Quick test script for S3FIFOForgive vs S3FIFO vs S3FIFOEmb"""

import subprocess
import re
from multiprocessing import Pool
from pathlib import Path

CACHESIM = Path(__file__).parent.parent / "_build/bin/cachesim"
DATA_DIR = Path(__file__).parent.parent / "data"

# Test configuration
TRACES = [
    ("w30.oracleGeneral.bin.zst", "oracleGeneral"),
    ("wiki_2019t.oracleGeneral.zst", "oracleGeneral"),
]

ALGORITHMS = ["s3fifo", "s3fifoemb", "s3fifoforgive"]

# Cache sizes in bytes (10MB, 50MB, 100MB for w30; 1GB, 5GB, 10GB for wiki)
CACHE_SIZES = {
    "w30": [10 * 1024 * 1024, 50 * 1024 * 1024, 100 * 1024 * 1024],
    "wiki_2019t": [1 * 1024 * 1024 * 1024, 5 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024],
}


def run_cachesim(args):
    trace_file, trace_type, algo, cache_size, trace_name = args
    trace_path = DATA_DIR / trace_file

    cmd = [
        str(CACHESIM),
        str(trace_path),
        trace_type,
        algo,
        str(cache_size),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        output = result.stdout + result.stderr

        # Parse miss ratio
        match = re.search(r'miss ratio\s+([\d.]+)', output)
        if match:
            miss_ratio = float(match.group(1))
            # Parse cache size display
            size_match = re.search(r'cache size\s+(\d+\w+)', output)
            size_str = size_match.group(1) if size_match else f"{cache_size}"
            return (trace_name, algo, size_str, miss_ratio)
    except subprocess.TimeoutExpired:
        return (trace_name, algo, f"{cache_size}", None)
    except Exception as e:
        return (trace_name, algo, f"{cache_size}", None)

    return (trace_name, algo, f"{cache_size}", None)


def main():
    # Build job list
    jobs = []
    for trace_file, trace_type in TRACES:
        trace_name = trace_file.split(".")[0]
        sizes = CACHE_SIZES.get(trace_name, [10*1024*1024, 50*1024*1024, 100*1024*1024])
        for algo in ALGORITHMS:
            for size in sizes:
                jobs.append((trace_file, trace_type, algo, size, trace_name))

    print(f"Running {len(jobs)} jobs with 8 workers...")

    # Run in parallel
    with Pool(8) as pool:
        results = pool.map(run_cachesim, jobs)

    # Organize and print results
    from collections import defaultdict
    by_trace = defaultdict(lambda: defaultdict(dict))

    for trace_name, algo, size_str, miss_ratio in results:
        if miss_ratio is not None:
            by_trace[trace_name][size_str][algo] = miss_ratio

    # Print results as table
    for trace_name in sorted(by_trace.keys()):
        print(f"\n=== {trace_name} ===")
        sizes = sorted(by_trace[trace_name].keys(), key=lambda x: int(re.sub(r'\D', '', x) or 0))

        # Header
        print(f"{'Size':<12} {'S3FIFO':<10} {'S3FIFOEmb':<12} {'S3FIFOForgive':<14} {'Forgive vs Base':<16}")
        print("-" * 66)

        for size_str in sizes:
            algos = by_trace[trace_name][size_str]
            s3fifo = algos.get("s3fifo", float('nan'))
            s3fifoemb = algos.get("s3fifoemb", float('nan'))
            s3fifoforgive = algos.get("s3fifoforgive", float('nan'))

            # Calculate improvement
            if s3fifo and s3fifoforgive:
                diff = (s3fifoforgive - s3fifo) / s3fifo * 100
                diff_str = f"{diff:+.2f}%"
            else:
                diff_str = "N/A"

            print(f"{size_str:<12} {s3fifo:<10.4f} {s3fifoemb:<12.4f} {s3fifoforgive:<14.4f} {diff_str:<16}")


if __name__ == "__main__":
    main()
