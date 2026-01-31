#!/usr/bin/env python3
"""
Two-pass parameter tuning script for S3FIFOForgiveClean.

Pass 1 (Training): Grid search threshold on training traces per workload.
Pass 2 (Evaluation): Evaluate fitted parameters on held-out test traces.

Key insight: Threshold is the dominant parameter (+0.51% impact).
Other params (min-access, max-forgives) have <0.02% impact and are fixed.
"""

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tune_threshold")

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent
BUILD_DIR = BASE_DIR / "_build"
CACHESIM_PATH = BUILD_DIR / "bin" / "cachesim"
DATA_DIR = BASE_DIR / "data"

# Fixed parameters (from sensitivity analysis - these have minimal impact)
FIXED_PARAMS = {
    "lr": 0.10,
    "ctx-speed": 0.001,
    "window": 32,
    "min-access": 2,
    "max-forgives": 15,
}

# Threshold grid to search
THRESHOLD_GRID = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]


@dataclass
class WorkloadConfig:
    """Configuration for a workload including train/test traces."""
    name: str
    train_traces: List[str]
    test_traces: List[str]
    trace_format: str
    trace_params: str


# Workload configurations
WORKLOADS = {
    "wiki_text": WorkloadConfig(
        name="wiki_text",
        train_traces=["cache-t-00", "cache-t-01"],
        test_traces=[f"cache-t-{i:02d}" for i in range(2, 21)],
        trace_format="csv",
        trace_params="obj-id-col=2,obj-size-col=3,has-header=true",
    ),
    "wiki_upload": WorkloadConfig(
        name="wiki_upload",
        train_traces=["cache-u-00", "cache-u-01"],
        test_traces=[f"cache-u-{i:02d}" for i in range(2, 21)],
        trace_format="csv",
        trace_params="obj-id-col=2,obj-size-col=4,has-header=true",
    ),
    "kvcache": WorkloadConfig(
        name="kvcache",
        train_traces=["kvcache_traces_1.csv", "kvcache_traces_2.csv"],
        test_traces=["kvcache_traces_3.csv", "kvcache_traces_4.csv", "kvcache_traces_5.csv"],
        trace_format="csv",
        trace_params="obj-id-col=2,obj-size-col=6,obj-id-is-num=0,has-header=true",
    ),
    "metacdn": WorkloadConfig(
        name="metacdn",
        train_traces=["reag0c01_20230315_20230322_0.2000.csv"],
        test_traces=[
            "rprn0c01_20230315_20230322_0.2000.csv",
            "rnha0c01_20230315_20230322_0.8000.csv",
        ],
        trace_format="csv",
        trace_params="obj-id-col=2,obj-size-col=4,obj-id-is-num=0,has-header=true",
    ),
}


def build_params_str(threshold: float) -> str:
    """Build parameter string for -e flag with given threshold."""
    params = FIXED_PARAMS.copy()
    params["threshold"] = threshold
    return ",".join(f"{k}={v}" for k, v in params.items())


def run_experiment(
    trace_name: str,
    config: WorkloadConfig,
    algo: str,
    cache_size: str,
    threshold: Optional[float] = None,
    num_req: Optional[int] = None,
) -> Optional[Dict]:
    """
    Run a single cachesim experiment.

    Args:
        trace_name: Name of the trace file
        config: Workload configuration
        algo: Algorithm name (s3fifo or s3fifoforgiveclean)
        cache_size: Cache size as fraction (e.g., "0.01")
        threshold: Threshold parameter (only for s3fifoforgiveclean)
        num_req: Optional limit on number of requests to process

    Returns:
        Dictionary with experiment results or None on failure
    """
    trace_path = DATA_DIR / trace_name

    if not trace_path.exists():
        logger.error(f"Trace not found: {trace_path}")
        return None

    cmd = [
        str(CACHESIM_PATH),
        str(trace_path),
        config.trace_format,
        algo,
        cache_size,
        "--ignore-obj-size=false",
        "--num-thread=1",
        "-t", config.trace_params,
    ]

    if num_req is not None:
        cmd.extend(["-n", str(num_req)])

    if algo == "s3fifoforgiveclean" and threshold is not None:
        params_str = build_params_str(threshold)
        cmd.extend(["-e", params_str])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=3600,
        )
        output = result.stdout.decode('utf-8', errors='replace')

        if result.returncode != 0:
            stderr = result.stderr.decode('utf-8', errors='replace')
            logger.error(f"cachesim failed for {trace_name}/{algo}: {stderr[:200]}")
            return None

        # Parse miss ratio from output
        miss_match = re.search(r'miss ratio\s+([\d.]+)', output)
        byte_miss_match = re.search(r'byte miss ratio\s+([\d.]+)', output)

        miss_ratio = float(miss_match.group(1)) if miss_match else None
        byte_miss_ratio = float(byte_miss_match.group(1)) if byte_miss_match else None

        if miss_ratio is None:
            logger.warning(f"Could not parse miss ratio for {trace_name}/{algo}")
            logger.debug(f"Output: {output[:500]}")
            return None

        return {
            "trace": trace_name,
            "workload": config.name,
            "algorithm": algo,
            "threshold": threshold,
            "miss_ratio": miss_ratio,
            "byte_miss_ratio": byte_miss_ratio,
        }

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout for {trace_name}/{algo}")
        return None
    except Exception as e:
        logger.error(f"Error running {trace_name}/{algo}: {e}")
        return None


def run_training_pass(
    workloads: Dict[str, WorkloadConfig],
    cache_size: str,
    max_workers: int,
    num_req: Optional[int] = None,
) -> Dict[str, Dict]:
    """
    Run Pass 1: Grid search threshold on training traces.

    Returns:
        Dictionary mapping workload name to fitted parameters
    """
    logger.info("=" * 60)
    logger.info("PASS 1: Training - Grid search threshold on training traces")
    logger.info("=" * 60)

    # Build experiment list
    experiments = []
    for workload_name, config in workloads.items():
        for trace in config.train_traces:
            for threshold in THRESHOLD_GRID:
                experiments.append((trace, config, "s3fifoforgiveclean", cache_size, threshold))

    logger.info(f"Running {len(experiments)} training experiments...")

    # Run experiments in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_experiment, trace, config, algo, size, thresh, num_req):
            (trace, config.name, thresh)
            for trace, config, algo, size, thresh in experiments
        }

        for i, future in enumerate(as_completed(futures)):
            trace, workload, thresh = futures[future]
            result = future.result()
            if result:
                results.append(result)
                logger.info(
                    f"[{i+1}/{len(experiments)}] {workload}/{trace} "
                    f"threshold={thresh}: miss={result['miss_ratio']:.4f}"
                )

    # Aggregate results and find best threshold per workload
    fitted_params = {}

    for workload_name, config in workloads.items():
        # Get all results for this workload
        workload_results = [r for r in results if r["workload"] == workload_name]

        # Group by threshold and compute average miss ratio
        threshold_scores = {}
        for thresh in THRESHOLD_GRID:
            thresh_results = [r for r in workload_results if r["threshold"] == thresh]
            if thresh_results:
                avg_miss = sum(r["miss_ratio"] for r in thresh_results) / len(thresh_results)
                threshold_scores[thresh] = avg_miss

        if not threshold_scores:
            logger.warning(f"No results for workload {workload_name}")
            continue

        # Find best threshold (lowest miss ratio)
        best_threshold = min(threshold_scores, key=threshold_scores.get)
        best_miss_ratio = threshold_scores[best_threshold]

        fitted_params[workload_name] = {
            "workload": workload_name,
            "best_threshold": best_threshold,
            "fixed_params": FIXED_PARAMS.copy(),
            "full_params_str": build_params_str(best_threshold),
            "train_miss_ratio": best_miss_ratio,
            "train_traces": config.train_traces,
            "grid_search_results": {str(k): v for k, v in sorted(threshold_scores.items())},
        }

        logger.info(
            f"\n{workload_name}: Best threshold = {best_threshold} "
            f"(avg miss ratio = {best_miss_ratio:.4f})"
        )
        logger.info(f"  Grid search results: {threshold_scores}")

    return fitted_params, results


def run_evaluation_pass(
    workloads: Dict[str, WorkloadConfig],
    fitted_params: Dict[str, Dict],
    cache_size: str,
    max_workers: int,
    num_req: Optional[int] = None,
) -> List[Dict]:
    """
    Run Pass 2: Evaluate on held-out test traces.

    Returns:
        List of evaluation results
    """
    logger.info("\n" + "=" * 60)
    logger.info("PASS 2: Evaluation - Test fitted parameters on held-out traces")
    logger.info("=" * 60)

    # Build experiment list: baseline s3fifo + fitted s3fifoforgiveclean
    experiments = []
    for workload_name, config in workloads.items():
        if workload_name not in fitted_params:
            logger.warning(f"No fitted params for {workload_name}, skipping evaluation")
            continue

        best_threshold = fitted_params[workload_name]["best_threshold"]

        for trace in config.test_traces:
            # Baseline: s3fifo
            experiments.append((trace, config, "s3fifo", cache_size, None))
            # Fitted: s3fifoforgiveclean with best threshold
            experiments.append((trace, config, "s3fifoforgiveclean", cache_size, best_threshold))

    logger.info(f"Running {len(experiments)} evaluation experiments...")

    # Run experiments in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_experiment, trace, config, algo, size, thresh, num_req):
            (trace, config.name, algo, thresh)
            for trace, config, algo, size, thresh in experiments
        }

        for i, future in enumerate(as_completed(futures)):
            trace, workload, algo, thresh = futures[future]
            result = future.result()
            if result:
                results.append(result)
                thresh_str = f" (thresh={thresh})" if thresh else ""
                logger.info(
                    f"[{i+1}/{len(experiments)}] {workload}/{trace} "
                    f"{algo}{thresh_str}: miss={result['miss_ratio']:.4f}"
                )

    return results


def save_fitted_params(
    fitted_params: Dict[str, Dict],
    output_dir: Path,
):
    """Save fitted parameters as JSON files."""
    params_dir = output_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    for workload_name, params in fitted_params.items():
        output_path = params_dir / f"{workload_name}.json"
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)
        logger.info(f"Saved fitted params: {output_path}")


def save_results_csv(
    results: List[Dict],
    output_path: Path,
):
    """Save results to CSV file."""
    if not results:
        logger.warning("No results to save")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["workload", "trace", "algorithm", "threshold", "miss_ratio", "byte_miss_ratio"]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x["workload"], x["trace"], x["algorithm"])):
            writer.writerow({k: r.get(k) for k in fieldnames})

    logger.info(f"Saved results: {output_path}")


def generate_summary(
    fitted_params: Dict[str, Dict],
    eval_results: List[Dict],
    output_dir: Path,
):
    """Generate markdown summary report."""
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary.md"

    lines = [
        "# S3FIFOForgiveClean Threshold Tuning Results",
        "",
        "## Fitted Parameters by Workload",
        "",
        "| Workload | Best Threshold | Train Miss Ratio |",
        "|----------|----------------|------------------|",
    ]

    for workload_name, params in sorted(fitted_params.items()):
        lines.append(
            f"| {workload_name} | {params['best_threshold']} | "
            f"{params['train_miss_ratio']:.4f} |"
        )

    lines.extend([
        "",
        "## Evaluation Results (Test Set)",
        "",
    ])

    # Group eval results by workload
    for workload_name in sorted(set(r["workload"] for r in eval_results)):
        workload_results = [r for r in eval_results if r["workload"] == workload_name]

        if not workload_results:
            continue

        lines.extend([
            f"### {workload_name}",
            "",
            "| Trace | S3FIFO | S3FIFOForgiveClean | Improvement |",
            "|-------|--------|--------------------| ------------|",
        ])

        # Group by trace
        traces = sorted(set(r["trace"] for r in workload_results))
        improvements = []

        for trace in traces:
            trace_results = [r for r in workload_results if r["trace"] == trace]
            baseline = next((r for r in trace_results if r["algorithm"] == "s3fifo"), None)
            fitted = next((r for r in trace_results if r["algorithm"] == "s3fifoforgiveclean"), None)

            if baseline and fitted:
                baseline_mr = baseline["miss_ratio"]
                fitted_mr = fitted["miss_ratio"]
                improvement = (baseline_mr - fitted_mr) * 100
                improvements.append(improvement)

                lines.append(
                    f"| {trace} | {baseline_mr:.4f} | {fitted_mr:.4f} | "
                    f"{improvement:+.2f}% |"
                )

        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            lines.extend([
                f"| **Average** | | | **{avg_improvement:+.2f}%** |",
                "",
            ])

    # Overall summary
    lines.extend([
        "## Summary",
        "",
        "### Fixed Parameters (from sensitivity analysis)",
        "```",
        f"lr={FIXED_PARAMS['lr']}, ctx-speed={FIXED_PARAMS['ctx-speed']}, "
        f"window={FIXED_PARAMS['window']},",
        f"min-access={FIXED_PARAMS['min-access']}, max-forgives={FIXED_PARAMS['max-forgives']}",
        "```",
        "",
        "### Threshold Grid Searched",
        f"`{THRESHOLD_GRID}`",
        "",
    ])

    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Saved summary: {summary_path}")

    # Also print summary to console
    print("\n" + "=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)
    print("\nFitted thresholds by workload:")
    for workload_name, params in sorted(fitted_params.items()):
        print(f"  {workload_name}: threshold={params['best_threshold']} "
              f"(train miss={params['train_miss_ratio']:.4f})")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Two-pass threshold tuning for S3FIFOForgiveClean"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.home() / "sigcomm" / "tuning_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--cache-size",
        type=str,
        default="0.01",
        help="Cache size as fraction (e.g., 0.01 for 1%%)",
    )
    parser.add_argument(
        "--workloads",
        type=str,
        default=None,
        help="Comma-separated list of workloads (default: all)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training pass (use existing fitted params)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-n", "--num-req",
        type=int,
        default=None,
        help="Limit number of requests per trace (for dry-run testing)",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select workloads
    if args.workloads:
        workload_names = [w.strip() for w in args.workloads.split(",")]
        workloads = {name: WORKLOADS[name] for name in workload_names if name in WORKLOADS}
    else:
        workloads = WORKLOADS

    logger.info(f"Workloads: {list(workloads.keys())}")
    logger.info(f"Cache size: {args.cache_size}")
    logger.info(f"Output directory: {output_dir}")
    if args.num_req:
        logger.info(f"Request limit (dry-run): {args.num_req}")

    # Check cachesim exists
    if not CACHESIM_PATH.exists():
        logger.error(f"cachesim not found at {CACHESIM_PATH}")
        logger.error("Please build libCacheSim first: cd _build && make")
        sys.exit(1)

    # Pass 1: Training
    if args.skip_training:
        # Load existing fitted params
        params_dir = output_dir / "params"
        fitted_params = {}
        for workload_name in workloads:
            params_path = params_dir / f"{workload_name}.json"
            if params_path.exists():
                with open(params_path) as f:
                    fitted_params[workload_name] = json.load(f)
                logger.info(f"Loaded fitted params for {workload_name}")
            else:
                logger.warning(f"No fitted params found for {workload_name}")
        training_results = []
    else:
        fitted_params, training_results = run_training_pass(
            workloads, args.cache_size, args.max_workers, args.num_req
        )

        # Save fitted params
        save_fitted_params(fitted_params, output_dir)

        # Save training results
        save_results_csv(
            training_results,
            output_dir / "results" / "training_results.csv"
        )

    # Pass 2: Evaluation
    eval_results = run_evaluation_pass(
        workloads, fitted_params, args.cache_size, args.max_workers, args.num_req
    )

    # Save evaluation results
    save_results_csv(
        eval_results,
        output_dir / "results" / "evaluation_results.csv"
    )

    # Generate summary
    generate_summary(fitted_params, eval_results, output_dir)

    logger.info("\nTuning complete!")


if __name__ == "__main__":
    main()
