#!/usr/bin/env python3
"""
Evaluate LRU and LRUForgiveEmbCache on all traces for a workload.
Outputs results to CSV.
"""

import subprocess
import re
import argparse
import csv
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_cachesim(cachesim_path, trace_path, algo, cache_size_ratio, eviction_params="", max_requests=None):
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        output = result.stdout + result.stderr
        match = re.search(r'miss ratio\s+([0-9.]+)', output)
        if match:
            return float(match.group(1))
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {Path(trace_path).name}", file=sys.stderr)
    except Exception as e:
        print(f"  ERROR: {Path(trace_path).name}: {e}", file=sys.stderr)
    return None


def evaluate_trace(args):
    """Evaluate a single trace with both algorithms."""
    cachesim_path, trace_path, cache_size_ratio, lr, ctx_speed, threshold, emb_budget, max_requests = args
    trace_name = Path(trace_path).name

    # Run LRU baseline
    baseline_mr = run_cachesim(cachesim_path, trace_path, "lru", cache_size_ratio, max_requests=max_requests)

    # Run LRUForgiveEmbCache with optimal params
    eviction_params = f"lr={lr},ctx-speed={ctx_speed},threshold={threshold},max-emb-entries={int(emb_budget)}x"
    test_mr = run_cachesim(cachesim_path, trace_path, "LRUForgiveEmbCache", cache_size_ratio, eviction_params, max_requests=max_requests)

    return {
        'trace': trace_name,
        'lru_mr': baseline_mr,
        'embcache_mr': test_mr,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LRU vs LRUForgiveEmbCache on all traces")
    parser.add_argument("--traces", type=str, nargs="+", required=True, help="Trace files to evaluate")
    parser.add_argument("--cachesim", type=str, default="/mydata/cachesim/libCacheSim/_build/bin/cachesim")
    parser.add_argument("--cache-size-ratio", type=float, default=0.01)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--ctx-speed", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--emb-budget", type=float, default=5.0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--n-workers", type=int, default=10)
    parser.add_argument("--workload", type=str, required=True, help="Workload name for CSV")

    args = parser.parse_args()

    # Validate traces
    trace_paths = []
    for tp in args.traces:
        if Path(tp).exists():
            trace_paths.append(tp)
        else:
            print(f"WARNING: Trace not found: {tp}", file=sys.stderr)

    if not trace_paths:
        print("ERROR: No valid traces found", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(trace_paths)} traces for {args.workload}")
    print(f"Optimal params: lr={args.lr}, ctx-speed={args.ctx_speed}, threshold={args.threshold}")
    print(f"Embedding budget: {args.emb_budget}x")
    if args.max_requests:
        print(f"Max requests: {args.max_requests}")
    print("-" * 60)

    # Build args list
    args_list = [
        (args.cachesim, tp, args.cache_size_ratio, args.lr, args.ctx_speed, args.threshold, args.emb_budget, args.max_requests)
        for tp in trace_paths
    ]

    # Run evaluations in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(evaluate_trace, a): a[1] for a in args_list}
        for future in as_completed(futures):
            result = future.result()
            if result['lru_mr'] is not None and result['embcache_mr'] is not None:
                improvement = (result['lru_mr'] - result['embcache_mr']) / result['lru_mr'] * 100
                result['improvement_pct'] = improvement
                result['workload'] = args.workload
                results.append(result)
                print(f"  {result['trace']}: lru={result['lru_mr']:.4f}, embcache={result['embcache_mr']:.4f}, imp={improvement:+.2f}%")
            else:
                print(f"  {result['trace']}: FAILED")

    # Sort by trace name
    results.sort(key=lambda x: x['trace'])

    # Write CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['workload', 'trace', 'lru_mr', 'embcache_mr', 'improvement_pct'])
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    if results:
        avg_imp = sum(r['improvement_pct'] for r in results) / len(results)
        pos_count = sum(1 for r in results if r['improvement_pct'] > 0)
        print("-" * 60)
        print(f"Results: {len(results)} traces evaluated")
        print(f"Average improvement: {avg_imp:+.3f}%")
        print(f"Traces improved: {pos_count}/{len(results)} ({100*pos_count/len(results):.0f}%)")
        print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
