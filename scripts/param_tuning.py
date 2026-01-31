#!/usr/bin/env python3
"""
Parameter tuning for S3FIFOForgiveClean.
Runs 20 intuitive parameter combinations based on sensitivity analysis.
"""

import csv
import subprocess
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHESIM = Path("/home/aryan/ecache/libCacheSim/_build/bin/cachesim")
DATA_DIR = Path("/home/aryan/ecache/libCacheSim/data")
OUTPUT_DIR = Path("/home/aryan/sigcomm/parameter_tuning/wikipedia")

TRACES = ["cache-t-00", "cache-t-01", "cache-t-02"]
CACHE_SIZE = "0.01"

# 20 intuitive parameter combinations based on sensitivity insights:
# - lr: lower seems better (0.05 > 0.10 > 0.20)
# - threshold: higher seems better (0.7 > 0.5 > 0.3)
# - ctx-speed: higher seems better (0.002 > 0.001)
# - window: smaller seems better (16 > 32)
# - min-access, max-forgives: minimal impact

CONFIGS = [
    # Config 0: Current baseline
    {"name": "baseline", "lr": 0.10, "ctx-speed": 0.001, "threshold": 0.5, "window": 32, "min-access": 2, "max-forgives": 15},

    # Config 1-3: Best from sensitivity (low lr, high threshold, high ctx-speed, small window)
    {"name": "best_combo_v1", "lr": 0.05, "ctx-speed": 0.002, "threshold": 0.7, "window": 16, "min-access": 2, "max-forgives": 15},
    {"name": "best_combo_v2", "lr": 0.05, "ctx-speed": 0.002, "threshold": 0.7, "window": 8, "min-access": 2, "max-forgives": 15},
    {"name": "best_combo_v3", "lr": 0.03, "ctx-speed": 0.002, "threshold": 0.7, "window": 16, "min-access": 2, "max-forgives": 15},

    # Config 4-6: Very high threshold variations
    {"name": "high_thresh_0.8", "lr": 0.05, "ctx-speed": 0.002, "threshold": 0.8, "window": 16, "min-access": 2, "max-forgives": 15},
    {"name": "high_thresh_0.9", "lr": 0.05, "ctx-speed": 0.002, "threshold": 0.9, "window": 16, "min-access": 2, "max-forgives": 15},
    {"name": "high_thresh_0.6", "lr": 0.05, "ctx-speed": 0.002, "threshold": 0.6, "window": 16, "min-access": 2, "max-forgives": 15},

    # Config 7-9: Very low learning rate
    {"name": "very_low_lr_0.02", "lr": 0.02, "ctx-speed": 0.002, "threshold": 0.7, "window": 16, "min-access": 2, "max-forgives": 15},
    {"name": "very_low_lr_0.01", "lr": 0.01, "ctx-speed": 0.002, "threshold": 0.7, "window": 16, "min-access": 2, "max-forgives": 15},
    {"name": "medium_lr_0.08", "lr": 0.08, "ctx-speed": 0.002, "threshold": 0.7, "window": 16, "min-access": 2, "max-forgives": 15},

    # Config 10-12: Context speed variations
    {"name": "high_ctx_0.003", "lr": 0.05, "ctx-speed": 0.003, "threshold": 0.7, "window": 16, "min-access": 2, "max-forgives": 15},
    {"name": "high_ctx_0.005", "lr": 0.05, "ctx-speed": 0.005, "threshold": 0.7, "window": 16, "min-access": 2, "max-forgives": 15},
    {"name": "very_high_ctx_0.01", "lr": 0.05, "ctx-speed": 0.01, "threshold": 0.7, "window": 16, "min-access": 2, "max-forgives": 15},

    # Config 13-15: Window size variations
    {"name": "tiny_window_4", "lr": 0.05, "ctx-speed": 0.002, "threshold": 0.7, "window": 4, "min-access": 2, "max-forgives": 15},
    {"name": "small_window_12", "lr": 0.05, "ctx-speed": 0.002, "threshold": 0.7, "window": 12, "min-access": 2, "max-forgives": 15},
    {"name": "medium_window_24", "lr": 0.05, "ctx-speed": 0.002, "threshold": 0.7, "window": 24, "min-access": 2, "max-forgives": 15},

    # Config 16-17: Aggressive forgiveness
    {"name": "aggressive_forgive", "lr": 0.05, "ctx-speed": 0.002, "threshold": 0.5, "window": 16, "min-access": 1, "max-forgives": 30},
    {"name": "conservative_forgive", "lr": 0.05, "ctx-speed": 0.002, "threshold": 0.8, "window": 16, "min-access": 3, "max-forgives": 5},

    # Config 18-19: Combined extreme best
    {"name": "ultra_best_v1", "lr": 0.03, "ctx-speed": 0.003, "threshold": 0.75, "window": 12, "min-access": 2, "max-forgives": 10},
    {"name": "ultra_best_v2", "lr": 0.04, "ctx-speed": 0.004, "threshold": 0.7, "window": 8, "min-access": 2, "max-forgives": 12},
]

def build_params_str(config):
    return f"lr={config['lr']},ctx-speed={config['ctx-speed']},threshold={config['threshold']},window={config['window']},min-access={config['min-access']},max-forgives={config['max-forgives']}"

def run_experiment(trace, config):
    trace_path = DATA_DIR / trace
    params_str = build_params_str(config)

    cmd = [
        str(CACHESIM), str(trace_path), "csv", "s3fifoforgiveclean", CACHE_SIZE,
        "--ignore-obj-size=false", "--num-thread=1",
        "-t", "obj-id-col=2,obj-size-col=3,has-header=true",
        "-e", params_str,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=3600)
        output = result.stdout.decode('utf-8', errors='replace')

        miss_match = re.search(r'miss ratio\s+([\d.]+)', output)
        byte_miss_match = re.search(r'byte miss ratio\s+([\d.]+)', output)

        return {
            "config_name": config["name"],
            "trace": trace,
            "miss_ratio": float(miss_match.group(1)) if miss_match else None,
            "byte_miss_ratio": float(byte_miss_match.group(1)) if byte_miss_match else None,
            **{k: v for k, v in config.items() if k != "name"}
        }
    except Exception as e:
        print(f"Error: {trace}/{config['name']}: {e}")
        return None

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build experiments: 20 configs x 3 traces = 60 experiments
    experiments = [(trace, config) for config in CONFIGS for trace in TRACES]
    print(f"Running {len(experiments)} experiments...")

    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(run_experiment, t, c): (t, c["name"]) for t, c in experiments}
        for i, future in enumerate(as_completed(futures)):
            trace, name = futures[future]
            r = future.result()
            if r:
                results.append(r)
                print(f"[{i+1}/{len(experiments)}] {name}/{trace}: miss={r['miss_ratio']:.4f}")

    # Save raw results
    output_csv = OUTPUT_DIR / "param_tuning.csv"
    fieldnames = ["config_name", "trace", "miss_ratio", "byte_miss_ratio", "lr", "ctx-speed", "threshold", "window", "min-access", "max-forgives"]
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda x: (x["config_name"], x["trace"])))

    print(f"\nResults saved to {output_csv}")

    # Aggregate by config
    from collections import defaultdict
    config_stats = defaultdict(list)
    for r in results:
        config_stats[r["config_name"]].append(r)

    # Calculate averages and rank
    summary = []
    for name, runs in config_stats.items():
        avg_miss = sum(r["miss_ratio"] for r in runs) / len(runs)
        avg_byte_miss = sum(r["byte_miss_ratio"] for r in runs if r["byte_miss_ratio"]) / len(runs)
        config = runs[0]
        summary.append({
            "name": name,
            "avg_miss": avg_miss,
            "avg_byte_miss": avg_byte_miss,
            "lr": config["lr"],
            "ctx-speed": config["ctx-speed"],
            "threshold": config["threshold"],
            "window": config["window"],
            "min-access": config["min-access"],
            "max-forgives": config["max-forgives"],
        })

    summary.sort(key=lambda x: x["avg_miss"])

    # Print ranked table
    print("\n" + "="*120)
    print("PARAMETER TUNING RESULTS - S3FIFOForgiveClean on Wikipedia (1% cache)")
    print("="*120)
    print(f"{'Rank':<5} {'Config Name':<22} {'Avg Miss':<10} {'Δ vs Base':<10} {'lr':<6} {'ctx-sp':<8} {'thresh':<7} {'win':<5} {'min-ac':<7} {'max-fg':<7}")
    print("-"*120)

    baseline_miss = next((s["avg_miss"] for s in summary if s["name"] == "baseline"), 0.7343)

    for i, s in enumerate(summary):
        delta = (baseline_miss - s["avg_miss"]) * 100
        delta_str = f"+{delta:.2f}%" if delta > 0 else f"{delta:.2f}%"
        print(f"{i+1:<5} {s['name']:<22} {s['avg_miss']:.4f}    {delta_str:<10} {s['lr']:<6} {s['ctx-speed']:<8} {s['threshold']:<7} {s['window']:<5} {s['min-access']:<7} {s['max-forgives']:<7}")

    print("="*120)
    print(f"\nBEST CONFIG: {summary[0]['name']}")
    print(f"  Parameters: lr={summary[0]['lr']}, ctx-speed={summary[0]['ctx-speed']}, threshold={summary[0]['threshold']}, window={summary[0]['window']}, min-access={summary[0]['min-access']}, max-forgives={summary[0]['max-forgives']}")
    print(f"  Avg miss ratio: {summary[0]['avg_miss']:.4f} ({(baseline_miss - summary[0]['avg_miss'])*100:+.2f}% vs baseline)")
    print("="*120)

if __name__ == "__main__":
    main()
