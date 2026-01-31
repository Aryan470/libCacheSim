#!/usr/bin/env python3
"""
Generate bar plots comparing LRU, LRU+Emb, S3FIFO, S3FIFO+Emb for each workload.
One plot per workload type.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path

# Read data
results = defaultdict(lambda: defaultdict(dict))

with open("benchmark_emb_all_workloads.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        wl = row['workload']
        trace = row['trace']
        algo = row['algorithm']
        miss = float(row['miss_ratio'])
        results[wl][trace][algo] = miss

# Define colors
colors = {
    'lru': '#1f77b4',
    'lruforgiveclean': '#aec7e8',
    's3fifo': '#ff7f0e',
    's3fifoforgiveclean': '#ffbb78'
}

algo_labels = {
    'lru': 'LRU',
    'lruforgiveclean': 'LRU+Emb',
    's3fifo': 'S3FIFO',
    's3fifoforgiveclean': 'S3FIFO+Emb'
}

algos = ['lru', 'lruforgiveclean', 's3fifo', 's3fifoforgiveclean']

output_dir = Path("plots_workload_comparison")
output_dir.mkdir(exist_ok=True)

for workload in ['wikipedia', 'kvcache', 'metacdn', 'cloudphysics']:
    if workload not in results:
        continue

    traces = sorted(results[workload].keys())
    n_traces = len(traces)

    # Determine figure size based on number of traces
    if workload == 'cloudphysics':
        fig_width = max(20, n_traces * 0.4)
        fig_height = 8
    else:
        fig_width = max(12, n_traces * 1.5)
        fig_height = 6

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = np.arange(n_traces)
    width = 0.2

    for i, algo in enumerate(algos):
        values = []
        for trace in traces:
            val = results[workload][trace].get(algo, 0) * 100
            values.append(val)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=algo_labels[algo], color=colors[algo])

    # Formatting
    ax.set_ylabel('Miss Ratio (%)', fontsize=12)
    ax.set_xlabel('Trace', fontsize=12)
    ax.set_title(f'{workload.upper()} - Cache Policy Comparison', fontsize=14, fontweight='bold')

    # Set x-axis labels
    if workload == 'cloudphysics':
        # Shorten cloudphysics trace names
        short_names = [t.replace('cloudphysics-', '') for t in traces]
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    else:
        short_names = [t.replace(f'{workload}-', '').replace('wiki-', '').replace('kvcache-', 'kv').replace('metacdn-', '') for t in traces]
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)

    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / f"{workload}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

print("\nAll plots saved to plots_workload_comparison/")
