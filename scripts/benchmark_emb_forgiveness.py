#!/usr/bin/env python3
"""
Benchmark script to compare embedding-based forgiveness across workloads.

Compares: LRU, LRU+EmbForgive, S3-FIFO, S3-FIFO+EmbForgive
Across: Wikipedia CDN, CloudPhysics, Meta KVCache, Meta CDN

Features:
- Caps concurrent experiments at N (default 20)
- Outputs CSV with all results
- Supports --num-req to limit requests per trace
"""

import argparse
import csv
import logging
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("benchmark_emb")

# Path configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent
BUILD_DIR = BASE_DIR / "_build"
CACHESIM_PATH = BUILD_DIR / "bin" / "cachesim"
DATA_DIR = BASE_DIR / "data"


@dataclass
class TraceConfig:
    """Configuration for a trace file."""
    path: Path
    trace_type: str
    trace_params: str
    cache_size: str  # Cache size: fraction (e.g., "0.01") or absolute (e.g., "1gb")
    num_req: Optional[int] = None  # Limit requests (None = all)
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.path.stem


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    trace: str
    workload: str
    algorithm: str
    cache_size: str
    num_requests: int
    miss_ratio: float
    byte_miss_ratio: Optional[float] = None
    forgive_pct: Optional[float] = None


def get_wikipedia_traces(cache_size: str = "0.01") -> List[TraceConfig]:
    """Get Wikipedia CDN trace configurations."""
    traces = []
    # cache-t-00 to cache-t-20
    for i in range(21):
        path = DATA_DIR / f"cache-t-{i:02d}"
        if path.exists():
            traces.append(TraceConfig(
                path=path,
                trace_type="csv",
                trace_params="obj-id-col=2,obj-size-col=3,has-header=true",
                cache_size=cache_size,
                name=f"wiki-t-{i:02d}"
            ))
    return traces


def get_cloudphysics_traces() -> List[TraceConfig]:
    """Get CloudPhysics trace configurations."""
    traces = []
    for path in sorted(DATA_DIR.glob("w*.oracleGeneral.bin.zst")):
        # Extract trace number
        match = re.match(r'w(\d+)\.', path.name)
        if match:
            num = int(match.group(1))
            traces.append(TraceConfig(
                path=path,
                trace_type="oracleGeneral",
                trace_params="",
                cache_size=10000,  # Will use default, adjust as needed
                name=f"cloudphysics-w{num:02d}"
            ))
    return traces


def get_kvcache_traces(cache_size: str = "0.01") -> List[TraceConfig]:
    """Get Meta KVCache trace configurations."""
    traces = []
    for i in range(1, 6):
        # Prefer uncompressed if available
        path = DATA_DIR / f"kvcache_traces_{i}.csv"
        if not path.exists():
            path = DATA_DIR / f"kvcache_traces_{i}.csv.zst"
        if path.exists():
            traces.append(TraceConfig(
                path=path,
                trace_type="csv",
                trace_params="obj-id-col=2,obj-size-col=6,has-header=true,obj-id-is-num=0",
                cache_size=cache_size,
                name=f"kvcache-{i}"
            ))
    return traces


def get_metacdn_traces(cache_size: str = "0.01") -> List[TraceConfig]:
    """Get Meta CDN trace configurations."""
    traces = []

    # reag trace
    reag_path = DATA_DIR / "reag0c01_20230315_20230322_0.2000.csv"
    if not reag_path.exists():
        reag_path = DATA_DIR / "reag0c01_20230315_20230322_0.2000.csv.zst"
    if reag_path.exists():
        traces.append(TraceConfig(
            path=reag_path,
            trace_type="csv",
            trace_params="obj-id-col=2,obj-size-col=4,has-header=true,obj-id-is-num=0",
            cache_size=cache_size,
            name="metacdn-reag"
        ))

    # rprn CSV format
    rprn_csv = DATA_DIR / "rprn0c01_20230315_20230322_0.2000.csv"
    if rprn_csv.exists():
        traces.append(TraceConfig(
            path=rprn_csv,
            trace_type="csv",
            trace_params="obj-id-col=2,obj-size-col=4,has-header=true,obj-id-is-num=0",
            cache_size=cache_size,
            name="metacdn-rprn"
        ))

    # rnha trace
    rnha_path = DATA_DIR / "rnha0c01_20230315_20230322_0.8000.csv"
    if not rnha_path.exists():
        rnha_path = DATA_DIR / "rnha0c01_20230315_20230322_0.8000.csv.zst"
    if rnha_path.exists():
        traces.append(TraceConfig(
            path=rnha_path,
            trace_type="csv",
            trace_params="obj-id-col=2,obj-size-col=4,has-header=true,obj-id-is-num=0",
            cache_size=cache_size,
            name="metacdn-rnha"
        ))

    return traces


def get_all_traces(workloads: List[str], cache_size: str = "0.01") -> Dict[str, List[TraceConfig]]:
    """Get all trace configurations organized by workload."""
    all_traces = {}

    if "wikipedia" in workloads or "all" in workloads:
        all_traces["wikipedia"] = get_wikipedia_traces(cache_size)

    if "cloudphysics" in workloads or "all" in workloads:
        all_traces["cloudphysics"] = get_cloudphysics_traces()

    if "kvcache" in workloads or "all" in workloads:
        all_traces["kvcache"] = get_kvcache_traces(cache_size)

    if "metacdn" in workloads or "all" in workloads:
        all_traces["metacdn"] = get_metacdn_traces(cache_size)

    return all_traces


def run_experiment(
    trace: TraceConfig,
    workload: str,
    algorithm: str,
    eviction_params: str = "",
    num_req: Optional[int] = None
) -> Optional[ExperimentResult]:
    """Run a single cachesim experiment."""

    cmd = [
        str(CACHESIM_PATH),
        str(trace.path),
        trace.trace_type,
        algorithm,
        str(trace.cache_size),
        "--ignore-obj-size=false",
        "--num-thread=1"
    ]

    if trace.trace_params:
        cmd.extend(["-t", trace.trace_params])

    if eviction_params:
        cmd.extend(["-e", eviction_params])

    req_limit = num_req or trace.num_req
    if req_limit:
        cmd.extend(["--num-req", str(req_limit)])

    logger.debug(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=3600
        )
        output = result.stdout.decode('utf-8', errors='replace')

        if result.returncode != 0:
            logger.error(f"Failed: {trace.name}/{algorithm}: exit {result.returncode}")
            logger.debug(f"Output: {output[:500]}")
            return None

        # Parse miss ratio
        miss_match = re.search(r'miss ratio\s+([\d.]+)', output)
        if not miss_match:
            logger.error(f"Could not parse miss ratio for {trace.name}/{algorithm}")
            logger.debug(f"Output: {output}")
            return None

        miss_ratio = float(miss_match.group(1))

        # Parse byte miss ratio
        byte_miss_ratio = None
        byte_miss_match = re.search(r'byte miss ratio\s+([\d.]+)', output)
        if byte_miss_match:
            byte_miss_ratio = float(byte_miss_match.group(1))

        # Parse number of requests
        req_match = re.search(r'(\d+)\s+req', output)
        num_requests = int(req_match.group(1)) if req_match else 0

        # Parse forgive percentage if available
        forgive_pct = None
        forgive_match = re.search(r'forgives=\d+\s+\(([\d.]+)%\)', output)
        if forgive_match:
            forgive_pct = float(forgive_match.group(1))

        return ExperimentResult(
            trace=trace.name,
            workload=workload,
            algorithm=algorithm,
            cache_size=trace.cache_size,
            num_requests=num_requests,
            miss_ratio=miss_ratio,
            byte_miss_ratio=byte_miss_ratio,
            forgive_pct=forgive_pct
        )

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout: {trace.name}/{algorithm}")
        return None
    except Exception as e:
        logger.error(f"Error: {trace.name}/{algorithm}: {e}")
        return None


def run_all_experiments(
    traces_by_workload: Dict[str, List[TraceConfig]],
    algorithms: List[Tuple[str, str]],  # (name, eviction_params)
    max_workers: int = 20,
    num_req: Optional[int] = None,
    output_path: Optional[Path] = None
) -> List[ExperimentResult]:
    """Run all experiments with bounded parallelism and incremental saving."""

    # Build job list
    jobs = []
    for workload, traces in traces_by_workload.items():
        for trace in traces:
            for algo_name, algo_params in algorithms:
                jobs.append((trace, workload, algo_name, algo_params, num_req))

    logger.info(f"Running {len(jobs)} experiments with max {max_workers} workers")

    results = []
    completed = 0
    save_interval = 10  # Save every N completions

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_experiment, trace, wl, algo, params, nreq): (trace.name, algo)
            for trace, wl, algo, params, nreq in jobs
        }

        for future in as_completed(futures):
            trace_name, algo = futures[future]
            completed += 1

            try:
                result = future.result()
                if result:
                    results.append(result)
                    logger.info(f"[{completed}/{len(jobs)}] {trace_name}/{algo}: miss={result.miss_ratio:.4f}")
                else:
                    logger.warning(f"[{completed}/{len(jobs)}] {trace_name}/{algo}: FAILED")
            except Exception as e:
                logger.error(f"[{completed}/{len(jobs)}] {trace_name}/{algo}: ERROR {e}")

            # Save incremental results
            if output_path and completed % save_interval == 0:
                save_results(results, output_path)
                logger.info(f"Saved {len(results)} results (checkpoint)")

    # Final save
    if output_path:
        save_results(results, output_path)

    return results


def save_results(results: List[ExperimentResult], output_path: Path):
    """Save results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'workload', 'trace', 'algorithm', 'cache_size',
            'num_requests', 'miss_ratio', 'hit_ratio',
            'byte_miss_ratio', 'byte_hit_ratio', 'forgive_pct'
        ])
        for r in sorted(results, key=lambda x: (x.workload, x.trace, x.algorithm)):
            byte_miss_str = f"{r.byte_miss_ratio:.6f}" if r.byte_miss_ratio is not None else ""
            byte_hit_str = f"{1-r.byte_miss_ratio:.6f}" if r.byte_miss_ratio is not None else ""
            writer.writerow([
                r.workload, r.trace, r.algorithm, r.cache_size,
                r.num_requests, f"{r.miss_ratio:.6f}", f"{1-r.miss_ratio:.6f}",
                byte_miss_str, byte_hit_str,
                f"{r.forgive_pct:.1f}" if r.forgive_pct is not None else ""
            ])

    logger.info(f"Results saved to {output_path}")


def print_summary(results: List[ExperimentResult]):
    """Print summary table."""
    # Group by workload and trace
    from collections import defaultdict

    grouped = defaultdict(dict)
    for r in results:
        key = (r.workload, r.trace)
        grouped[key][r.algorithm] = r

    print("\n" + "="*80)
    print("SUMMARY: Miss Ratio by Trace and Algorithm")
    print("="*80)

    algos = ["lru", "lruforgiveclean", "s3fifo", "s3fifoforgiveclean"]

    # Header
    print(f"{'Workload':<15} {'Trace':<20} ", end="")
    for algo in algos:
        short_name = algo.replace("forgiveclean", "+emb")
        print(f"{short_name:<12} ", end="")
    print("LRU Δ   S3F Δ")
    print("-"*100)

    for (workload, trace), algo_results in sorted(grouped.items()):
        print(f"{workload:<15} {trace:<20} ", end="")

        values = {}
        for algo in algos:
            if algo in algo_results:
                values[algo] = algo_results[algo].miss_ratio
                print(f"{values[algo]:.4f}      ", end="")
            else:
                print(f"{'N/A':<12} ", end="")

        # Calculate deltas
        lru_delta = ""
        s3f_delta = ""
        if "lru" in values and "lruforgiveclean" in values:
            delta = (values["lru"] - values["lruforgiveclean"]) * 100
            lru_delta = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        if "s3fifo" in values and "s3fifoforgiveclean" in values:
            delta = (values["s3fifo"] - values["s3fifoforgiveclean"]) * 100
            s3f_delta = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"

        print(f"{lru_delta:<7} {s3f_delta}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark embedding-based forgiveness across workloads"
    )
    parser.add_argument(
        "--workloads",
        type=str,
        default="all",
        help="Comma-separated workloads: wikipedia,cloudphysics,kvcache,metacdn,all"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Maximum concurrent experiments (default: 20)"
    )
    parser.add_argument(
        "--num-req",
        type=int,
        default=None,
        help="Limit requests per trace (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_emb_forgiveness.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--s3fifo-params",
        type=str,
        default="lr=0.10,ctx-speed=0.001,threshold=0.7,window=40,min-access=2,max-forgives=15",
        help="S3-FIFO forgiveness parameters (default: optimal from tuning)"
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="lru,lruforgiveclean,s3fifo,s3fifoforgiveclean",
        help="Comma-separated algorithms to run (default: all four)"
    )
    parser.add_argument(
        "--cache-size",
        type=str,
        default="0.01",
        help="Cache size as fraction of working set (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Check cachesim exists
    if not CACHESIM_PATH.exists():
        logger.error(f"cachesim not found at {CACHESIM_PATH}")
        logger.error("Run: bash scripts/install_libcachesim.sh")
        sys.exit(1)

    # Parse workloads
    workloads = [w.strip() for w in args.workloads.split(",")]

    # Get traces
    traces_by_workload = get_all_traces(workloads, args.cache_size)

    total_traces = sum(len(t) for t in traces_by_workload.values())
    logger.info(f"Found {total_traces} traces across {len(traces_by_workload)} workloads")
    logger.info(f"Cache size: {args.cache_size}")
    for wl, traces in traces_by_workload.items():
        logger.info(f"  {wl}: {len(traces)} traces")

    if total_traces == 0:
        logger.error("No traces found")
        sys.exit(1)

    # Define algorithms based on --algorithms flag
    all_algorithms = {
        "lru": ("lru", ""),
        "lruforgiveclean": ("lruforgiveclean", ""),
        "s3fifo": ("s3fifo", ""),
        "s3fifoforgiveclean": ("s3fifoforgiveclean", args.s3fifo_params),
    }
    requested_algos = [a.strip() for a in args.algorithms.split(",")]
    algorithms = [all_algorithms[a] for a in requested_algos if a in all_algorithms]

    logger.info(f"Algorithms: {[a[0] for a in algorithms]}")
    if "s3fifoforgiveclean" in requested_algos:
        logger.info(f"S3FIFO params: {args.s3fifo_params}")

    # Run experiments with incremental saving
    output_path = Path(args.output)
    results = run_all_experiments(
        traces_by_workload,
        algorithms,
        max_workers=args.max_workers,
        num_req=args.num_req,
        output_path=output_path
    )

    if not results:
        logger.error("No results collected")
        sys.exit(1)

    logger.info(f"Collected {len(results)} results")

    # Save and summarize
    save_results(results, Path(args.output))
    print_summary(results)


if __name__ == "__main__":
    main()
