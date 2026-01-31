#!/usr/bin/env python3
"""
Generate comparison tables and plots for cache eviction benchmark results.

Creates markdown tables showing miss ratios for each algorithm across traces,
with a winner column highlighting when forgive algorithms win.

Also generates box-and-whisker plots showing hit rate improvement of forgive
algorithms over their base versions.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def generate_boxplot(csv_path: str, output_path: str = None):
    """Generate box-and-whisker plot of hit rate changes for forgive algorithms."""

    # Read data
    df = pd.read_csv(csv_path)

    # Clean trace names
    df['trace'] = df['trace'].str.replace('.oracleGeneral.bin', '')
    df['trace'] = df['trace'].str.replace('.oracleGeneral', '')

    # Convert miss ratio to hit rate
    df['hit_rate'] = 1 - df['miss_ratio']

    # Define base-forgive pairs
    pairs = [
        ('lru', 'lruforgive', 'LRU-F vs LRU'),
        ('sieve', 'sieveembforgive', 'SIEVE-F vs SIEVE'),
        ('s3fifo', 's3fifoforgive', 'S3FIFO-F vs S3FIFO'),
    ]

    # Calculate hit rate improvement for each pair
    improvements = {}

    for base_algo, forgive_algo, label in pairs:
        base_df = df[df['algorithm'] == base_algo][['trace', 'cache_size_mb', 'hit_rate']]
        forgive_df = df[df['algorithm'] == forgive_algo][['trace', 'cache_size_mb', 'hit_rate']]

        if base_df.empty or forgive_df.empty:
            continue

        # Merge on trace and cache_size
        merged = pd.merge(
            base_df, forgive_df,
            on=['trace', 'cache_size_mb'],
            suffixes=('_base', '_forgive')
        )

        # Calculate improvement (positive = forgive is better)
        # Change in hit rate: forgive_hit_rate - base_hit_rate
        merged['improvement'] = merged['hit_rate_forgive'] - merged['hit_rate_base']

        # Convert to percentage points
        merged['improvement_pct'] = merged['improvement'] * 100

        improvements[label] = merged['improvement_pct'].values

    if not improvements:
        print("No data to plot")
        return

    # Create box plot
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = list(improvements.keys())
    data = [improvements[label] for label in labels]

    # Create box plot with custom colors
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # blue, red, green
    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Labels and title
    ax.set_ylabel('Hit Rate Change (percentage points)', fontsize=12)
    ax.set_xlabel('Forgive Algorithm vs Base', fontsize=12)
    ax.set_title('Hit Rate Improvement of Forgive Algorithms\n(Positive = Forgive is Better)', fontsize=14)

    # Add statistics text
    stats_text = []
    for i, (label, values) in enumerate(improvements.items()):
        median = np.median(values)
        mean = np.mean(values)
        positive = np.sum(values > 0)
        total = len(values)
        stats_text.append(f"{label}:\n  Median: {median:+.2f}pp\n  Mean: {mean:+.2f}pp\n  Wins: {positive}/{total}")

    # Add text box with stats
    textstr = '\n'.join(stats_text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for stats text

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Box plot saved to {output_path}")
    else:
        plt.show()

    plt.close()

    return improvements


def generate_transposed_table(csv_path: str, size_idx: int = 0, output_path: str = None):
    """Generate a transposed table (algorithms as rows, traces as columns)."""

    # Read data
    df = pd.read_csv(csv_path)

    # Clean trace names
    df['trace'] = df['trace'].str.replace('.oracleGeneral.bin', '')
    df['trace'] = df['trace'].str.replace('.oracleGeneral', '')

    # Get cache size indices
    trace_sizes = df.groupby('trace')['cache_size_mb'].apply(
        lambda x: sorted(x.unique())
    ).to_dict()

    def get_size_index(row):
        sizes = trace_sizes[row['trace']]
        return sizes.index(row['cache_size_mb'])

    df['size_index'] = df.apply(get_size_index, axis=1)

    # Filter to requested size
    subset = df[df['size_index'] == size_idx]

    # Algorithm order and display names
    algo_order = ['lru', 'lruforgive', 'sieve', 'sieveembforgive', 's3fifo', 's3fifoforgive']
    algo_display = {
        'lru': 'LRU',
        'lruforgive': 'LRU-F',
        'sieve': 'SIEVE',
        'sieveembforgive': 'SIEVE-F',
        's3fifo': 'S3FIFO',
        's3fifoforgive': 'S3FIFO-F'
    }

    # Pivot: traces as columns, algorithms as rows
    pivot = subset.pivot(index='algorithm', columns='trace', values='miss_ratio')

    # Sort traces numerically
    def trace_sort_key(t):
        if t.startswith('w') and t[1:].isdigit():
            return (0, int(t[1:]))
        return (1, t)

    sorted_traces = sorted(pivot.columns, key=trace_sort_key)
    pivot = pivot[sorted_traces]

    # Reorder rows by algo_order
    pivot = pivot.reindex([a for a in algo_order if a in pivot.index])

    # Find winner for each trace (column)
    winners = {}
    for trace in pivot.columns:
        col = pivot[trace]
        winner = col.idxmin()
        winners[trace] = winner

    size_labels = {0: '1%', 1: '5%', 2: '10%'}

    output_lines = []
    output_lines.append(f"## Cache Size: {size_labels[size_idx]} of Working Set (Transposed)\n")

    # Build header (traces)
    header = "| Algorithm | " + " | ".join(sorted_traces) + " |"
    separator = "|-----------|" + "|".join(["-----:" for _ in sorted_traces]) + "|"

    output_lines.append(header)
    output_lines.append(separator)

    # Build rows (algorithms)
    forgive_algos = {'lruforgive', 'sieveembforgive', 's3fifoforgive'}

    for algo in pivot.index:
        row = pivot.loc[algo]
        algo_name = algo_display.get(algo, algo)

        cells = []
        for trace in sorted_traces:
            val = row[trace]
            cells.append(f"{val:.3f}")

        output_lines.append(f"| {algo_name} | " + " | ".join(cells) + " |")

    # Add winner row
    winner_cells = []
    for trace in sorted_traces:
        winner_algo = winners[trace]
        winner_name = algo_display.get(winner_algo, winner_algo)
        if winner_algo in forgive_algos:
            winner_cells.append(f"**{winner_name}**")
        else:
            winner_cells.append(winner_name)

    output_lines.append(f"| **Winner** | " + " | ".join(winner_cells) + " |")

    # Count forgive wins
    forgive_wins = sum(1 for w in winners.values() if w in forgive_algos)
    output_lines.append(f"\n**Forgive wins: {forgive_wins}/{len(winners)}**")

    output_text = "\n".join(output_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(output_text)
        print(f"Transposed table written to {output_path}")
    else:
        print(output_text)

    return output_text


def generate_tables(csv_path: str, output_path: str = None):
    """Generate comparison tables from benchmark CSV results."""

    # Read data
    df = pd.read_csv(csv_path)

    # Clean trace names
    df['trace'] = df['trace'].str.replace('.oracleGeneral.bin', '')
    df['trace'] = df['trace'].str.replace('.oracleGeneral', '')

    # Get unique traces and their cache sizes
    traces = df['trace'].unique()

    # For each trace, identify the 3 cache size indices (small, medium, large)
    trace_sizes = df.groupby('trace')['cache_size_mb'].apply(
        lambda x: sorted(x.unique())
    ).to_dict()

    # Add size_index (0=1%, 1=5%, 2=10%)
    def get_size_index(row):
        sizes = trace_sizes[row['trace']]
        return sizes.index(row['cache_size_mb'])

    df['size_index'] = df.apply(get_size_index, axis=1)

    # Algorithm display names and order
    algo_order = ['lru', 'lruforgive', 'sieve', 'sieveembforgive', 's3fifo', 's3fifoforgive']
    algo_display = {
        'lru': 'LRU',
        'lruforgive': 'LRU-F',
        'sieve': 'SIEVE',
        'sieveembforgive': 'SIEVE-F',
        's3fifo': 'S3FIFO',
        's3fifoforgive': 'S3FIFO-F'
    }

    # Filter to only algorithms present in the data
    present_algos = [a for a in algo_order if a in df['algorithm'].unique()]

    forgive_algos = {'lruforgive', 'sieveembforgive', 's3fifoforgive'}
    size_labels = {0: '1%', 1: '5%', 2: '10%'}

    output_lines = []
    output_lines.append("# Cache Eviction Algorithm Comparison\n")

    total_forgive_wins = 0
    total_entries = 0

    for size_idx in [0, 1, 2]:
        output_lines.append(f"\n## Cache Size: {size_labels[size_idx]} of Working Set\n")

        subset = df[df['size_index'] == size_idx]

        if subset.empty:
            output_lines.append("No data for this cache size.\n")
            continue

        # Pivot to get trace x algorithm
        pivot = subset.pivot(index='trace', columns='algorithm', values='miss_ratio')

        # Reorder columns to match algo_order (only present ones)
        pivot = pivot[[a for a in present_algos if a in pivot.columns]]

        # Find winner for each trace
        winners = []
        for trace in pivot.index:
            row = pivot.loc[trace]
            winner_algo = row.idxmin()
            winners.append(winner_algo)

        pivot['winner'] = winners

        # Build markdown table
        header_cols = [algo_display.get(a, a) for a in pivot.columns if a != 'winner']
        header = "| Trace | " + " | ".join(header_cols) + " | Winner |"
        separator = "|-------|" + "|".join(["------:" for _ in header_cols]) + "|--------|"

        output_lines.append(header)
        output_lines.append(separator)

        forgive_wins = 0

        # Sort traces numerically if they start with 'w'
        def trace_sort_key(t):
            if t.startswith('w') and t[1:].isdigit():
                return (0, int(t[1:]))
            return (1, t)

        for trace in sorted(pivot.index, key=trace_sort_key):
            row = pivot.loc[trace]
            winner = row['winner']

            # Format values
            cells = []
            for algo in pivot.columns:
                if algo == 'winner':
                    continue
                val = row[algo]
                if pd.isna(val):
                    cells.append("-")
                else:
                    cells.append(f"{val:.2f}")

            # Format winner name
            winner_display = algo_display.get(winner, winner)
            if winner in forgive_algos:
                winner_display = f"**{winner_display}**"
                forgive_wins += 1

            output_lines.append(f"| {trace} | " + " | ".join(cells) + f" | {winner_display} |")

        total_forgive_wins += forgive_wins
        total_entries += len(pivot)
        output_lines.append(f"\n**Forgive algorithm wins: {forgive_wins}/{len(pivot)}**\n")

    output_lines.append(f"\n---\n**Total forgive wins across all cache sizes: {total_forgive_wins}/{total_entries}**\n")

    output_text = "\n".join(output_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(output_text)
        print(f"Tables written to {output_path}")
    else:
        print(output_text)

    return output_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison tables and plots from benchmark results"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="benchmark_cloudphysics_full.csv",
        help="Input CSV file with benchmark results"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output markdown file for tables (default: print to stdout)"
    )
    parser.add_argument(
        "--boxplot", "-b",
        type=str,
        default=None,
        help="Output path for box-and-whisker plot (e.g., plot.png)"
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Only generate tables, skip plots"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots, skip tables"
    )
    parser.add_argument(
        "--transposed", "-t",
        type=str,
        default=None,
        help="Output path for transposed table (algorithms as rows)"
    )
    parser.add_argument(
        "--size-index",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Cache size index for transposed table (0=1%%, 1=5%%, 2=10%%)"
    )

    args = parser.parse_args()

    if not args.plot_only and not args.transposed:
        generate_tables(args.input, args.output)

    if args.transposed:
        generate_transposed_table(args.input, args.size_index, args.transposed)

    if not args.tables_only and args.boxplot:
        generate_boxplot(args.input, args.boxplot)


if __name__ == "__main__":
    main()
