#!/usr/bin/env python3
"""
Benchmark script to compare cache eviction policies.

Runs S3-FIFO, SIEVE, SIEVE-emb, and S3-FIFO-emb across traces,
then generates comparison plots.
"""

import argparse
import csv
import logging
import multiprocessing
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("benchmark_policies")

# Path configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent
BUILD_DIR = BASE_DIR / "_build"
CACHESIM_PATH = BUILD_DIR / "bin" / "cachesim"
DATA_DIR = BASE_DIR / "data"

# Default configuration
DEFAULT_POLICIES = ["s3fifo", "sieve", "sieveemb", "s3fifoemb"]
DEFAULT_CACHE_SIZES = [0.01, 0.05, 0.1]  # 1%, 5%, 10% of working set
DEFAULT_NUM_WORKERS = 16

# CSV trace format parameters
CSV_PARAMS = {
    "twitter_cluster52": "time-col=1,obj-id-col=2,obj-size-col=3,delimiter=,,obj-id-is-num=1",
    "twitter_cluster52_10m": "time-col=1,obj-id-col=2,obj-size-col=3,delimiter=,,obj-id-is-num=1",
    "cloudPhysicsIO": "time-col=2,obj-id-col=5,obj-size-col=4,delimiter=,",
}


def detect_trace_format(trace_path: Path) -> Tuple[str, Optional[str]]:
    """
    Detect trace format from filename.

    Returns:
        Tuple of (format_type, format_params)
    """
    name = trace_path.name
    stem = trace_path.stem

    # Handle compressed files
    if name.endswith('.zst'):
        stem = Path(stem).stem
        name = name[:-4]

    if name.endswith('.oracleGeneral') or name.endswith('.oracleGeneral.bin'):
        return "oracleGeneral", None
    elif name.endswith('.vscsi'):
        return "vscsi", None
    elif name.endswith('.csv'):
        # Look for matching CSV params
        for key, params in CSV_PARAMS.items():
            if key in stem:
                return "csv", params
        # Default CSV params
        logger.warning(f"No CSV params found for {name}, using default")
        return "csv", "obj-id-col=1,delimiter=,"
    elif name.endswith('.txt'):
        return "txt", None
    else:
        logger.warning(f"Unknown trace format for {name}, trying oracleGeneral")
        return "oracleGeneral", None


def discover_traces(data_dir: Path) -> List[Path]:
    """Discover all trace files in the data directory."""
    traces = []

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return traces

    # Supported extensions
    extensions = [
        '.oracleGeneral', '.oracleGeneral.bin', '.oracleGeneral.zst',
        '.oracleGeneral.bin.zst', '.vscsi', '.csv', '.csv.zst', '.txt'
    ]

    for f in data_dir.iterdir():
        if f.is_file():
            name = f.name
            for ext in extensions:
                if name.endswith(ext):
                    traces.append(f)
                    break

    logger.info(f"Discovered {len(traces)} traces in {data_dir}")
    for t in traces:
        logger.debug(f"  - {t.name}")

    return sorted(traces)


def validate_algorithm(algo: str) -> bool:
    """Quick test to check if algorithm is available."""
    if not CACHESIM_PATH.exists():
        logger.error(f"cachesim not found at {CACHESIM_PATH}")
        return False

    # Use a small test with the cloudPhysicsIO trace
    test_trace = DATA_DIR / "cloudPhysicsIO.vscsi"
    if not test_trace.exists():
        # Try to find any trace
        traces = discover_traces(DATA_DIR)
        if not traces:
            logger.error("No traces found for validation")
            return False
        test_trace = traces[0]

    trace_fmt, fmt_params = detect_trace_format(test_trace)

    cmd = [
        str(CACHESIM_PATH),
        str(test_trace),
        trace_fmt,
        algo,
        "0.01",
        "--num-thread", "1",
        "-n", "1000"  # Only process 1000 requests for quick test
    ]

    if fmt_params:
        cmd.extend(["--trace-type-params", fmt_params])

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30
        )
        output = result.stdout.decode('utf-8')

        if result.returncode != 0:
            logger.warning(f"Algorithm {algo} validation failed: exit code {result.returncode}")
            return False

        if "miss ratio" in output.lower():
            return True

        logger.warning(f"Algorithm {algo} output unexpected: {output[:200]}")
        return False

    except subprocess.TimeoutExpired:
        logger.warning(f"Algorithm {algo} validation timed out")
        return False
    except Exception as e:
        logger.warning(f"Algorithm {algo} validation error: {e}")
        return False


def run_cachesim_job(args: Tuple) -> List[Dict]:
    """
    Run cachesim for a single (trace, algo) pair with all cache sizes.

    Returns list of result dictionaries.
    """
    trace_path, algo, cache_sizes = args
    results = []

    trace_fmt, fmt_params = detect_trace_format(trace_path)
    trace_name = trace_path.stem
    if trace_name.endswith('.oracleGeneral'):
        trace_name = trace_name[:-14]

    # Format cache sizes as comma-separated string
    sizes_str = ",".join(str(s) for s in cache_sizes)

    cmd = [
        str(CACHESIM_PATH),
        str(trace_path),
        trace_fmt,
        algo,
        sizes_str,
        "--num-thread", "1"
    ]

    if fmt_params:
        cmd.extend(["--trace-type-params", fmt_params])

    logger.info(f"Running: {trace_name} with {algo}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=3600  # 1 hour timeout
        )
        output = result.stdout.decode('utf-8')

        if result.returncode != 0:
            logger.error(f"cachesim failed for {trace_name}/{algo}: exit code {result.returncode}")
            logger.error(f"Output: {output[:500]}")
            return results

        # Parse output lines
        # Format: /path/to/trace AlgoName cache size XX[MiB|GiB], NNNN req, miss ratio 0.XXXX, byte miss ratio 0.XXXX
        pattern = r'cache size\s+(\d+)(MiB|GiB).*?miss ratio\s+([\d.]+)(?:.*?byte miss ratio\s+([\d.]+))?'

        for line in output.split('\n'):
            if 'miss ratio' in line.lower():
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    cache_size_val = int(match.group(1))
                    cache_size_unit = match.group(2)
                    # Convert to MB for consistency
                    if cache_size_unit.lower() == 'gib':
                        cache_size_mb = cache_size_val * 1024
                    else:
                        cache_size_mb = cache_size_val
                    miss_ratio = float(match.group(3))
                    byte_miss_ratio = float(match.group(4)) if match.group(4) else None

                    results.append({
                        'trace': trace_name,
                        'algorithm': algo,
                        'cache_size_mb': cache_size_mb,
                        'miss_ratio': miss_ratio,
                        'byte_miss_ratio': byte_miss_ratio
                    })
                else:
                    logger.warning(f"Could not parse line: {line}")

        if not results:
            logger.warning(f"No results parsed for {trace_name}/{algo}")
            logger.debug(f"Output was: {output}")

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout for {trace_name}/{algo}")
    except Exception as e:
        logger.error(f"Error running {trace_name}/{algo}: {e}")

    return results


def run_benchmarks(
    traces: List[Path],
    algos: List[str],
    cache_sizes: List[float],
    num_workers: int
) -> List[Dict]:
    """Run all benchmarks in parallel."""

    # Create job list: one job per (trace, algo) pair
    jobs = []
    for trace in traces:
        for algo in algos:
            jobs.append((trace, algo, cache_sizes))

    logger.info(f"Running {len(jobs)} jobs with {num_workers} workers")

    all_results = []

    if num_workers == 1:
        # Sequential execution for debugging
        for job in jobs:
            results = run_cachesim_job(job)
            all_results.extend(results)
    else:
        # Parallel execution
        with multiprocessing.Pool(num_workers) as pool:
            job_results = pool.map(run_cachesim_job, jobs)
            for results in job_results:
                all_results.extend(results)

    return all_results


def save_results_csv(results: List[Dict], output_path: Path):
    """Save results to CSV file."""
    if not results:
        logger.warning("No results to save")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['trace', 'algorithm', 'cache_size_mb', 'miss_ratio', 'byte_miss_ratio']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results saved to {output_path}")


def generate_plots(results: List[Dict], output_dir: Path, cache_sizes: List[float]):
    """Generate comparison bar charts."""
    if not results:
        logger.warning("No results to plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique traces and algorithms
    traces = sorted(set(r['trace'] for r in results))
    algorithms = sorted(set(r['algorithm'] for r in results))

    # For each trace, map cache sizes to indices (0=smallest, 1=medium, 2=largest)
    # This handles the fact that different traces have different working set sizes
    trace_size_indices = {}
    for trace in traces:
        trace_results = [r for r in results if r['trace'] == trace]
        trace_sizes = sorted(set(r['cache_size_mb'] for r in trace_results))
        trace_size_indices[trace] = {size: idx for idx, size in enumerate(trace_sizes)}

    # Add size_index to results
    for r in results:
        r['size_index'] = trace_size_indices[r['trace']].get(r['cache_size_mb'], 0)

    # Color scheme for algorithms
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
    algo_colors = dict(zip(algorithms, colors))

    # Generate plots for each metric and cache size index
    metrics = [
        ('miss_ratio', 'Miss Ratio'),
        ('byte_miss_ratio', 'Byte Miss Ratio')
    ]

    # Map size index to percentage label
    size_index_to_pct = {i: int(s * 100) for i, s in enumerate(cache_sizes)}

    for metric_key, metric_label in metrics:
        for size_idx in range(len(cache_sizes)):
            pct = size_index_to_pct.get(size_idx, size_idx)

            # Filter results for this size index
            size_results = [r for r in results if r.get('size_index') == size_idx]

            if not size_results:
                continue

            # Skip if metric not available
            if all(r.get(metric_key) is None for r in size_results):
                logger.warning(f"No {metric_key} data for {pct}% cache size")
                continue

            # Create figure
            fig, ax = plt.subplots(figsize=(max(10, len(traces) * 1.5), 6))

            # Bar positions
            x = np.arange(len(traces))
            width = 0.8 / len(algorithms)

            for i, algo in enumerate(algorithms):
                values = []
                for trace in traces:
                    val = None
                    for r in size_results:
                        if r['trace'] == trace and r['algorithm'] == algo:
                            val = r.get(metric_key)
                            break
                    values.append(val if val is not None else 0)

                offset = (i - len(algorithms) / 2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, label=algo, color=algo_colors[algo])

                # Add value labels on bars
                for bar, val in zip(bars, values):
                    if val > 0:
                        height = bar.get_height()
                        ax.annotate(f'{val:.3f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8, rotation=90)

            ax.set_xlabel('Trace')
            ax.set_ylabel(metric_label)
            ax.set_title(f'{metric_label} at {pct}% Cache Size')
            ax.set_xticks(x)
            ax.set_xticklabels(traces, rotation=45, ha='right')
            ax.legend(loc='upper right')
            ax.set_ylim(0, 1.1)

            plt.tight_layout()

            # Save plot
            filename = f"{metric_key}_{pct}pct.png"
            plot_path = output_dir / filename
            plt.savefig(plot_path, dpi=150)
            plt.close()

            logger.info(f"Saved plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cache eviction policies"
    )
    parser.add_argument(
        "--algos",
        type=str,
        default=",".join(DEFAULT_POLICIES),
        help="Comma-separated list of algorithms to test"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=",".join(str(s) for s in DEFAULT_CACHE_SIZES),
        help="Comma-separated cache sizes as fraction of working set"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory containing trace files"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--traces",
        type=str,
        default=None,
        help="Comma-separated list of specific traces to run (default: all in data dir)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip algorithm validation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Parse arguments
    requested_algos = [a.strip() for a in args.algos.split(",")]
    cache_sizes = [float(s.strip()) for s in args.sizes.split(",")]
    data_dir = Path(args.data_dir)
    output_csv = Path(args.output_csv)
    plots_dir = Path(args.plots_dir)

    # Discover traces
    if args.traces:
        trace_names = [t.strip() for t in args.traces.split(",")]
        all_traces = discover_traces(data_dir)
        traces = []
        for name in trace_names:
            for t in all_traces:
                if name in t.name:
                    traces.append(t)
                    break
            else:
                logger.warning(f"Trace not found: {name}")
    else:
        traces = discover_traces(data_dir)

    if not traces:
        logger.error("No traces found")
        sys.exit(1)

    logger.info(f"Using {len(traces)} traces:")
    for t in traces:
        logger.info(f"  - {t.name}")

    # Validate algorithms
    if args.skip_validation:
        valid_algos = requested_algos
    else:
        logger.info("Validating algorithms...")
        valid_algos = []
        for algo in requested_algos:
            if validate_algorithm(algo):
                valid_algos.append(algo)
                logger.info(f"  {algo}: OK")
            else:
                logger.warning(f"  {algo}: SKIPPED (not available)")

    if not valid_algos:
        logger.error("No valid algorithms found")
        sys.exit(1)

    logger.info(f"Running with algorithms: {', '.join(valid_algos)}")
    logger.info(f"Cache sizes: {cache_sizes}")

    # Run benchmarks
    results = run_benchmarks(traces, valid_algos, cache_sizes, args.num_workers)

    if not results:
        logger.error("No results obtained")
        sys.exit(1)

    logger.info(f"Collected {len(results)} result entries")

    # Save CSV
    save_results_csv(results, output_csv)

    # Generate plots
    generate_plots(results, plots_dir, cache_sizes)

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
