#!/usr/bin/env python3
"""
Bar chart comparing LRU, LRU-Random, and LRU-Embedding hit rates
across 6 Wikipedia CDN traces.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from standalone simulator (miss ratios)
traces = ['cache-t-00', 'cache-t-01', 'cache-t-02', 'cache-t-03', 'cache-t-04', 'cache-t-05', 'cache-u-00\n(10M)']
lru_miss = [0.7741, 0.7642, 0.7671, 0.7737, 0.7763, 0.7746, 0.8048]
random_miss = [0.7686, 0.7596, 0.7626, 0.7689, 0.7701, 0.7681, 0.8013]
emb_miss = [0.7035, 0.6932, 0.6990, 0.7086, 0.7094, 0.7084, 0.7614]

# Convert to hit rates
lru_hit = [1 - m for m in lru_miss]
random_hit = [1 - m for m in random_miss]
emb_hit = [1 - m for m in emb_miss]

# Plot setup
x = np.arange(len(traces))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))

# Create bars
bars1 = ax.bar(x - width, lru_hit, width, label='LRU', color='#1f77b4', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, random_hit, width, label='LRU + Random Forgive', color='#ff7f0e', edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, emb_hit, width, label='LRU + Embedding Forgive', color='#2ca02c', edgecolor='black', linewidth=0.5)

# Labels and formatting
ax.set_ylabel('Hit Rate', fontsize=12)
ax.set_xlabel('Trace', fontsize=12)
ax.set_title('Cache Hit Rate Comparison: LRU vs Random vs Embedding Forgiveness\n(Wikipedia CDN Traces, 1% Cache Size)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(traces, fontsize=10)
ax.legend(loc='upper left', fontsize=10)

# Set y-axis to start from reasonable value
ax.set_ylim(0.18, 0.35)

# Add grid
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=90)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Add summary stats as text
avg_lru = np.mean(lru_hit)
avg_random = np.mean(random_hit)
avg_emb = np.mean(emb_hit)
emb_vs_lru = (avg_emb - avg_lru) * 100  # in percentage points
emb_vs_random = (avg_emb - avg_random) * 100

summary_text = f'Average improvement:\nEmb vs LRU: +{emb_vs_lru:.2f} pp\nEmb vs Random: +{emb_vs_random:.2f} pp'
ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/aryan/ecache/libCacheSim/plots/lru_forgive_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('/home/aryan/ecache/libCacheSim/plots/lru_forgive_comparison.pdf', bbox_inches='tight')
print(f"Saved to /home/aryan/ecache/libCacheSim/plots/lru_forgive_comparison.png")

# Also print table
print("\n" + "="*70)
print("Summary Table")
print("="*70)
print(f"{'Trace':<12} {'LRU Hit':<12} {'Random Hit':<12} {'Emb Hit':<12} {'Emb-LRU':<10}")
print("-"*70)
for i, trace in enumerate(traces):
    diff = (emb_hit[i] - lru_hit[i]) * 100
    print(f"{trace:<12} {lru_hit[i]:<12.4f} {random_hit[i]:<12.4f} {emb_hit[i]:<12.4f} +{diff:.2f} pp")
print("-"*70)
print(f"{'Average':<12} {avg_lru:<12.4f} {avg_random:<12.4f} {avg_emb:<12.4f} +{emb_vs_lru:.2f} pp")
print("="*70)
