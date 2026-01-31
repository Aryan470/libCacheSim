#!/usr/bin/env python3
"""
Parameter sensitivity analysis for S3FIFOForgiveClean.
Runs experiments with perturbed parameters on Wikipedia traces.
"""

import csv
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
CACHESIM = Path("/home/aryan/ecache/libCacheSim/_build/bin/cachesim")
DATA_DIR = Path("/home/aryan/ecache/libCacheSim/data")
OUTPUT_DIR = Path("/home/aryan/sigcomm/parameter_tuning/wikipedia")

# Baseline parameters
BASELINE = {
    "lr": 0.10,
    "ctx-speed": 0.001,
    "threshold": 0.5,
    "window": 32,
    "min-access": 2,
    "max-forgives": 15,
}

# Perturbations: (decreased, increased)
PERTURBATIONS = {
    "lr": (0.05, 0.20),
    "ctx-speed": (0.0005, 0.002),
    "threshold": (0.3, 0.7),
    "window": (16, 64),
    "min-access": (1, 4),
    "max-forgives": (5, 30),
}

# Traces
TRACES = ["cache-t-00", "cache-t-01", "cache-t-02"]
CACHE_SIZE = "0.01"

def build_params_str(params):
    """Build parameter string for -e flag."""
    return ",".join(f"{k}={v}" for k, v in params.items())

def run_experiment(trace, params, param_name, direction):
    """Run a single experiment."""
    trace_path = DATA_DIR / trace
    params_str = build_params_str(params)

    cmd = [
        str(CACHESIM),
        str(trace_path),
        "csv",
        "s3fifoforgiveclean",
        CACHE_SIZE,
        "--ignore-obj-size=false",
        "--num-thread=1",
        "-t", "obj-id-col=2,obj-size-col=3,has-header=true",
        "-e", params_str,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=3600)
        output = result.stdout.decode('utf-8', errors='replace')

        # Parse miss ratio
        import re
        miss_match = re.search(r'miss ratio\s+([\d.]+)', output)
        byte_miss_match = re.search(r'byte miss ratio\s+([\d.]+)', output)

        miss_ratio = float(miss_match.group(1)) if miss_match else None
        byte_miss_ratio = float(byte_miss_match.group(1)) if byte_miss_match else None

        # For baseline, use "baseline" as value; otherwise use the param value
        value = "baseline" if param_name == "baseline" else params.get(param_name, "N/A")

        return {
            "trace": trace,
            "param": param_name,
            "direction": direction,
            "value": value,
            "miss_ratio": miss_ratio,
            "byte_miss_ratio": byte_miss_ratio,
        }
    except Exception as e:
        print(f"Error: {trace}/{param_name}/{direction}: {e}", file=sys.stderr)
        return None

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build experiment list
    experiments = []

    # Add baseline experiments
    for trace in TRACES:
        experiments.append((trace, BASELINE.copy(), "baseline", "baseline"))

    # Add perturbed experiments
    for param_name, (low, high) in PERTURBATIONS.items():
        for trace in TRACES:
            # Decreased
            params_low = BASELINE.copy()
            params_low[param_name] = low
            experiments.append((trace, params_low, param_name, "decreased"))

            # Increased
            params_high = BASELINE.copy()
            params_high[param_name] = high
            experiments.append((trace, params_high, param_name, "increased"))

    print(f"Running {len(experiments)} experiments...")

    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(run_experiment, trace, params, param_name, direction): (trace, param_name, direction)
            for trace, params, param_name, direction in experiments
        }

        for i, future in enumerate(as_completed(futures)):
            trace, param_name, direction = futures[future]
            result = future.result()
            if result:
                results.append(result)
                print(f"[{i+1}/{len(experiments)}] {trace}/{param_name}/{direction}: miss={result['miss_ratio']:.4f}")

    # Save raw results
    output_csv = OUTPUT_DIR / "param_sensitivity.csv"
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["trace", "param", "direction", "value", "miss_ratio", "byte_miss_ratio"])
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda x: (x["param"], x["trace"], x["direction"])))

    print(f"\nResults saved to {output_csv}")

    # Print summary table
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY ANALYSIS - S3FIFOForgiveClean on Wikipedia (1% cache)")
    print("="*80)

    # Get baseline results
    baselines = {r["trace"]: r["miss_ratio"] for r in results if r["param"] == "baseline"}

    # Group by parameter
    from collections import defaultdict
    param_results = defaultdict(lambda: {"decreased": [], "increased": []})

    for r in results:
        if r["param"] != "baseline":
            param_results[r["param"]][r["direction"]].append({
                "trace": r["trace"],
                "miss_ratio": r["miss_ratio"],
                "value": r["value"],
                "baseline": baselines.get(r["trace"], 0),
            })

    # Print table
    print(f"\n{'Parameter':<15} {'Direction':<10} {'Value':<10} {'Avg Miss Δ':<12} {'Effect':<20}")
    print("-"*70)

    for param in ["lr", "ctx-speed", "threshold", "window", "min-access", "max-forgives"]:
        for direction in ["decreased", "increased"]:
            data = param_results[param][direction]
            if data:
                avg_miss = sum(d["miss_ratio"] for d in data) / len(data)
                avg_baseline = sum(d["baseline"] for d in data) / len(data)
                delta = (avg_baseline - avg_miss) * 100  # Positive = improvement
                value = data[0]["value"]

                if delta > 0.1:
                    effect = "BETTER (+)"
                elif delta < -0.1:
                    effect = "WORSE (-)"
                else:
                    effect = "minimal"

                print(f"{param:<15} {direction:<10} {value:<10} {delta:+.2f}%       {effect}")

    print("="*80)
    print(f"Baseline avg miss ratio: {sum(baselines.values())/len(baselines):.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
