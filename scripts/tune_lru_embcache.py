#!/usr/bin/env python3
"""
Multi-trace Optuna tuning for LRUForgiveEmbCache.
Optimizes average relative improvement over LRU across training traces.
"""

import optuna
import subprocess
import re
import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional
import time


def run_cachesim(
    cachesim_path: str,
    trace_path: str,
    algo: str,
    cache_size_ratio: float,
    eviction_params: str = "",
    max_requests: Optional[int] = None,
) -> Optional[float]:
    """Run cachesim and return miss ratio."""
    cmd = [
        cachesim_path, trace_path, "oracleGeneral", algo, str(cache_size_ratio),
        "--ignore-obj-size=true"
    ]
    if eviction_params:
        cmd.extend(["--eviction-params", eviction_params])
    if max_requests:
        cmd.extend(["-n", str(max_requests)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        output = result.stdout + result.stderr
        match = re.search(r'miss ratio\s+([0-9.]+)', output)
        if match:
            return float(match.group(1))
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {Path(trace_path).name}")
    except Exception as e:
        print(f"  ERROR: {Path(trace_path).name}: {e}")
    return None


def run_single_trace(args: Tuple) -> Tuple[str, float, float]:
    """Run both baseline and test on a single trace. Returns (trace_name, baseline_mr, test_mr)."""
    (cachesim_path, trace_path, cache_size_ratio, eviction_params, max_requests) = args

    trace_name = Path(trace_path).name

    # Run LRU baseline
    baseline_mr = run_cachesim(
        cachesim_path, trace_path, "lru", cache_size_ratio,
        max_requests=max_requests
    )

    # Run LRUForgiveEmbCache with params
    test_mr = run_cachesim(
        cachesim_path, trace_path, "LRUForgiveEmbCache", cache_size_ratio,
        eviction_params=eviction_params, max_requests=max_requests
    )

    return (trace_name, baseline_mr, test_mr)


def evaluate_params(
    cachesim_path: str,
    trace_paths: List[str],
    cache_size_ratio: float,
    lr: float,
    ctx_speed: float,
    threshold: float,
    emb_budget: float,
    max_requests: Optional[int] = None,
    n_workers: int = 4,
) -> Tuple[float, List[dict]]:
    """
    Evaluate parameters across all traces.
    Returns (avg_improvement_pct, per_trace_results).
    """
    eviction_params = f"lr={lr},ctx-speed={ctx_speed:.2e},threshold={threshold},max-emb-entries={int(emb_budget)}x"

    args_list = [
        (cachesim_path, tp, cache_size_ratio, eviction_params, max_requests)
        for tp in trace_paths
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_trace, args): args[1] for args in args_list}
        for future in as_completed(futures):
            trace_name, baseline_mr, test_mr = future.result()
            if baseline_mr is not None and test_mr is not None:
                improvement = (baseline_mr - test_mr) / baseline_mr * 100
                results.append({
                    'trace': trace_name,
                    'baseline_mr': baseline_mr,
                    'test_mr': test_mr,
                    'improvement_pct': improvement
                })

    if not results:
        return -100.0, []  # Penalize if all failed

    avg_improvement = sum(r['improvement_pct'] for r in results) / len(results)
    return avg_improvement, results


def create_objective(
    cachesim_path: str,
    trace_paths: List[str],
    cache_size_ratio: float,
    emb_budget: float,
    max_requests: Optional[int],
    n_workers: int,
):
    """Create Optuna objective function."""
    trial_count = [0]
    best_so_far = [float('-inf')]
    start_time = [time.time()]

    def objective(trial: optuna.Trial) -> float:
        trial_count[0] += 1

        # Sample hyperparameters
        lr = trial.suggest_float("lr", 0.01, 0.5, log=True)
        ctx_speed = trial.suggest_float("ctx_speed", 1e-7, 1e-3, log=True)
        threshold = trial.suggest_float("threshold", 0.3, 0.95)

        # Evaluate
        avg_improvement, results = evaluate_params(
            cachesim_path, trace_paths, cache_size_ratio,
            lr, ctx_speed, threshold, emb_budget,
            max_requests, n_workers
        )

        # Update best
        if avg_improvement > best_so_far[0]:
            best_so_far[0] = avg_improvement

        # Print progress
        elapsed = time.time() - start_time[0]
        print(f"[Trial {trial_count[0]:3d}] lr={lr:.4f} ctx={ctx_speed:.2e} th={threshold:.3f} "
              f"=> avg_imp={avg_improvement:+.3f}% (best={best_so_far[0]:+.3f}%) [{elapsed/60:.1f}m]")

        # Optuna minimizes, so negate improvement
        return -avg_improvement

    return objective


def main():
    parser = argparse.ArgumentParser(description="Multi-trace Optuna tuning for LRUForgiveEmbCache")
    parser.add_argument("--traces", type=str, nargs="+", required=True,
                        help="Paths to training traces")
    parser.add_argument("--cachesim", type=str, default="/mydata/cachesim/libCacheSim/_build/bin/cachesim",
                        help="Path to cachesim binary")
    parser.add_argument("--cache-size-ratio", type=float, default=0.01,
                        help="Cache size as ratio of working set")
    parser.add_argument("--emb-budget", type=float, default=5.0,
                        help="Embedding budget multiplier")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="Cap traces at this many requests")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of Optuna trials")
    parser.add_argument("--n-parallel-trials", type=int, default=10,
                        help="Number of parallel Optuna trials")
    parser.add_argument("--n-trace-workers", type=int, default=2,
                        help="Workers per trial for trace evaluation")
    parser.add_argument("--output", type=str, default="tuning_lru_results.json",
                        help="Output JSON file")
    parser.add_argument("--study-name", type=str, default="lru-embcache",
                        help="Optuna study name")

    args = parser.parse_args()

    # Validate traces exist
    trace_paths = []
    for tp in args.traces:
        if Path(tp).exists():
            trace_paths.append(tp)
        else:
            print(f"WARNING: Trace not found: {tp}")

    if not trace_paths:
        print("ERROR: No valid traces found")
        sys.exit(1)

    print(f"=" * 70)
    print(f"Multi-trace Optuna Tuning for LRUForgiveEmbCache")
    print(f"=" * 70)
    print(f"Training traces: {len(trace_paths)}")
    for tp in trace_paths:
        print(f"  - {Path(tp).name}")
    print(f"Cache size ratio: {args.cache_size_ratio}")
    print(f"Embedding budget: {args.emb_budget}x")
    print(f"Max requests: {args.max_requests or 'unlimited'}")
    print(f"Trials: {args.n_trials} ({args.n_parallel_trials} parallel)")
    print(f"Trace workers per trial: {args.n_trace_workers}")
    print(f"=" * 70)

    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",  # We negate improvement, so minimize
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Create objective
    objective = create_objective(
        args.cachesim, trace_paths, args.cache_size_ratio,
        args.emb_budget, args.max_requests, args.n_trace_workers
    )

    # Run optimization
    print(f"\nStarting optimization...")
    print("-" * 70)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_parallel_trials,
        show_progress_bar=False,
    )

    # Results
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best avg improvement: {-study.best_value:+.3f}%")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        if "ctx" in key:
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value:.4f}")

    # Final evaluation with best params
    print("\nFinal evaluation on training traces:")
    best = study.best_params
    avg_imp, per_trace = evaluate_params(
        args.cachesim, trace_paths, args.cache_size_ratio,
        best['lr'], best['ctx_speed'], best['threshold'],
        args.emb_budget, args.max_requests, args.n_trace_workers
    )

    for r in sorted(per_trace, key=lambda x: x['improvement_pct']):
        print(f"  {r['trace']}: {r['improvement_pct']:+.2f}% "
              f"(baseline={r['baseline_mr']:.4f}, test={r['test_mr']:.4f})")

    # Save results
    results = {
        'best_params': study.best_params,
        'best_avg_improvement': -study.best_value,
        'n_trials': args.n_trials,
        'n_traces': len(trace_paths),
        'traces': [str(tp) for tp in trace_paths],
        'per_trace_results': per_trace,
        'cache_size_ratio': args.cache_size_ratio,
        'emb_budget': args.emb_budget,
        'max_requests': args.max_requests,
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
