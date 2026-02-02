#!/usr/bin/env python3
"""
Optuna-based hyperparameter tuning for embedding-based cache policies.

Usage:
    python tune_emb_cache.py --trace /path/to/trace.csv --algo s3fifoforgive-embcache

    # Custom search space
    python tune_emb_cache.py --trace /path/to/trace.csv --algo lruforgiveembcache \
        --lr-range 0.01 0.5 --ctx-range 1e-7 1e-3 --th-range 0.3 0.95
"""

import optuna
import subprocess
import re
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Auto-detect cachesim binary
SCRIPT_DIR = Path(__file__).parent
CACHESIM_PATH = SCRIPT_DIR.parent / "_build" / "bin" / "cachesim"


@dataclass
class TraceConfig:
    path: str
    trace_type: str = "csv"
    trace_params: str = "obj-id-col=2,obj-size-col=3,has-header=true"
    cache_size: Optional[int] = None
    cache_size_ratio: float = 0.01  # 1% of working set
    ignore_obj_size: bool = True


@dataclass
class SearchSpace:
    lr_min: float = 0.01
    lr_max: float = 0.5
    ctx_min: float = 1e-7
    ctx_max: float = 1e-3
    th_min: float = 0.3
    th_max: float = 0.95
    emb_budget_ratio: float = 5.0  # multiplier of cache size


def get_working_set_size(trace_path: str, trace_type: str, trace_params: str) -> Optional[int]:
    """Get working set size from trace using cachesim."""
    cmd = [
        str(CACHESIM_PATH), trace_path, trace_type, "lru", "1000",
        "--trace-type-params", trace_params,
        "--ignore-obj-size=true"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr

    # Try to find working set size in output
    match = re.search(r'working set size:\s+(\d+)\s+object', output)
    if match:
        return int(match.group(1))

    # Fallback: estimate from number of requests (assume ~30% unique)
    match = re.search(r'(\d+)\s+req,', output)
    if match:
        n_requests = int(match.group(1))
        return int(n_requests * 0.3)  # rough estimate

    return None


def run_experiment(
    trace: TraceConfig,
    algo: str,
    lr: float,
    ctx_speed: float,
    threshold: float,
    emb_entries: int,
    window: int = 16,
    min_access: int = 2,
    max_forgives: int = 5,
) -> float:
    """Run a single experiment and return the miss ratio."""

    # Build eviction params based on algorithm
    if "lru" in algo.lower():
        eviction_params = (
            f"lr={lr},ctx-speed={ctx_speed},window={window},"
            f"min-access-count={min_access},max-forgives={max_forgives},"
            f"threshold={threshold},max-emb-entries={emb_entries}"
        )
    else:  # s3fifo variants
        eviction_params = (
            f"lr={lr},ctx-speed={ctx_speed},window={window},"
            f"min-access={min_access},max-forgives={max_forgives},"
            f"threshold={threshold},max-emb-entries={emb_entries}"
        )

    cmd = [
        str(CACHESIM_PATH),
        trace.path,
        trace.trace_type,
        algo,
        str(trace.cache_size),
        "--trace-type-params", trace.trace_params,
        "-e", eviction_params,
    ]

    if trace.ignore_obj_size:
        cmd.append("--ignore-obj-size=true")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr

        match = re.search(r'miss ratio ([0-9.]+)', output)
        if match:
            return float(match.group(1))
        else:
            print(f"Failed to parse output: {output[-500:]}")
            return 1.0
    except subprocess.TimeoutExpired:
        print("Experiment timed out")
        return 1.0
    except Exception as e:
        print(f"Error running experiment: {e}")
        return 1.0


def create_objective(trace: TraceConfig, algo: str, search: SearchSpace):
    """Create Optuna objective function with closure over config."""

    emb_entries = int(trace.cache_size * search.emb_budget_ratio)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", search.lr_min, search.lr_max, log=True)
        ctx_speed = trial.suggest_float("ctx_speed", search.ctx_min, search.ctx_max, log=True)
        threshold = trial.suggest_float("threshold", search.th_min, search.th_max)

        miss_ratio = run_experiment(
            trace=trace,
            algo=algo,
            lr=lr,
            ctx_speed=ctx_speed,
            threshold=threshold,
            emb_entries=emb_entries,
        )

        print(f"Trial {trial.number}: lr={lr:.4f}, ctx={ctx_speed:.2e}, th={threshold:.3f} -> miss={miss_ratio:.4f}")
        return miss_ratio

    return objective


def run_baseline(trace: TraceConfig, baseline_algo: str) -> float:
    """Run baseline algorithm for comparison."""
    cmd = [
        str(CACHESIM_PATH),
        trace.path,
        trace.trace_type,
        baseline_algo,
        str(trace.cache_size),
        "--trace-type-params", trace.trace_params,
    ]
    if trace.ignore_obj_size:
        cmd.append("--ignore-obj-size=true")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr

    match = re.search(r'miss ratio ([0-9.]+)', output)
    if match:
        return float(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Tune embedding cache parameters with Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required
    parser.add_argument("--trace", type=str, required=True, help="Path to trace file")
    parser.add_argument("--algo", type=str, default="s3fifoforgive-embcache",
                        choices=["s3fifoforgive-embcache", "lruforgiveembcache"],
                        help="Algorithm to tune")

    # Trace config
    parser.add_argument("--trace-type", type=str, default="csv", help="Trace type")
    parser.add_argument("--trace-params", type=str,
                        default="obj-id-col=2,obj-size-col=3,has-header=true",
                        help="Trace type params")
    parser.add_argument("--cache-size", type=int, default=None,
                        help="Cache size (auto-detected if not specified)")
    parser.add_argument("--cache-size-ratio", type=float, default=0.01,
                        help="Cache size as ratio of working set (default: 0.01)")
    parser.add_argument("--ignore-obj-size", action="store_true", default=True,
                        help="Ignore object sizes")

    # Search space
    parser.add_argument("--lr-range", type=float, nargs=2, default=[0.01, 0.5],
                        metavar=("MIN", "MAX"), help="Learning rate range")
    parser.add_argument("--ctx-range", type=float, nargs=2, default=[1e-7, 1e-3],
                        metavar=("MIN", "MAX"), help="Context speed range")
    parser.add_argument("--th-range", type=float, nargs=2, default=[0.3, 0.95],
                        metavar=("MIN", "MAX"), help="Threshold range")
    parser.add_argument("--emb-budget", type=float, default=5.0,
                        help="Embedding budget as multiplier of cache size")

    # Optuna config
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--n-jobs", type=int, default=8, help="Parallel jobs")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")

    args = parser.parse_args()

    # Setup trace config
    trace = TraceConfig(
        path=args.trace,
        trace_type=args.trace_type,
        trace_params=args.trace_params,
        cache_size=args.cache_size,
        cache_size_ratio=args.cache_size_ratio,
        ignore_obj_size=args.ignore_obj_size,
    )

    # Auto-detect cache size if not specified
    if trace.cache_size is None:
        print("Detecting working set size...")
        wss = get_working_set_size(trace.path, trace.trace_type, trace.trace_params)
        if wss is None:
            print("ERROR: Could not auto-detect working set size.")
            print("Please specify --cache-size explicitly.")
            return
        trace.cache_size = int(wss * trace.cache_size_ratio)
        print(f"Working set: {wss}, Cache size ({trace.cache_size_ratio*100:.0f}%): {trace.cache_size}")

    # Setup search space
    search = SearchSpace(
        lr_min=args.lr_range[0],
        lr_max=args.lr_range[1],
        ctx_min=args.ctx_range[0],
        ctx_max=args.ctx_range[1],
        th_min=args.th_range[0],
        th_max=args.th_range[1],
        emb_budget_ratio=args.emb_budget,
    )

    # Run baseline
    baseline_algo = "s3fifo" if "s3fifo" in args.algo.lower() else "lru"
    print(f"\nRunning {baseline_algo} baseline...")
    baseline_miss = run_baseline(trace, baseline_algo)
    print(f"Baseline miss ratio: {baseline_miss:.4f}")

    # Create study
    trace_name = Path(args.trace).stem
    study_name = f"{args.algo}-{trace_name}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
    )

    print(f"\nStarting optimization: {args.n_trials} trials, {args.n_jobs} parallel jobs")
    print(f"Embedding budget: {search.emb_budget_ratio}x cache size")
    print("-" * 60)

    objective = create_objective(trace, args.algo, search)
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    # Results
    print("\n" + "=" * 60)
    print("Optimization complete!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best miss ratio: {study.best_value:.4f}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        if "ctx" in key:
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value:.4f}")

    if baseline_miss:
        improvement = (baseline_miss - study.best_value) / baseline_miss * 100
        print(f"\nImprovement over {baseline_algo}: {improvement:.2f}%")

    # Save results
    results = {
        "trace": args.trace,
        "algo": args.algo,
        "baseline_algo": baseline_algo,
        "baseline_miss_ratio": baseline_miss,
        "best_miss_ratio": study.best_value,
        "best_params": study.best_params,
        "improvement_pct": improvement if baseline_miss else None,
        "n_trials": args.n_trials,
        "cache_size": trace.cache_size,
        "emb_entries": int(trace.cache_size * search.emb_budget_ratio),
    }

    output_path = args.output
    if output_path is None:
        output_path = f"tuning_results_{study_name}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
