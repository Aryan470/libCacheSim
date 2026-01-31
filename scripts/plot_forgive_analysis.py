#!/usr/bin/env python3
"""Scatter plots analyzing SieveEmbForgive results."""

import matplotlib.pyplot as plt
import numpy as np

# Data from threshold=0.3 experiments
# Format: (trace, cache_pct, sieve_mr, forgive_mr, forgive_rate, trace_type)
data = [
    # CloudPhysics traces (wXX)
    ("w30", "1%", 0.8841, 0.8706, 0.03, "CloudPhysics"),
    ("w30", "5%", 0.7227, 0.5662, 0.40, "CloudPhysics"),
    ("w30", "10%", 0.5305, 0.3485, 0.90, "CloudPhysics"),
    ("w31", "1%", 0.8749, 0.8749, 0.00, "CloudPhysics"),
    ("w31", "5%", 0.8344, 0.8344, 0.00, "CloudPhysics"),
    ("w31", "10%", 0.8189, 0.8189, 0.00, "CloudPhysics"),
    ("w32", "1%", 0.7200, 0.7189, 0.89, "CloudPhysics"),
    ("w32", "5%", 0.6745, 0.6700, 1.04, "CloudPhysics"),
    ("w32", "10%", 0.6302, 0.5502, 2.25, "CloudPhysics"),
    ("w33", "1%", 0.1993, 0.1993, 3.82, "CloudPhysics"),
    ("w33", "5%", 0.1816, 0.1816, 3.28, "CloudPhysics"),
    ("w33", "10%", 0.1568, 0.1570, 2.28, "CloudPhysics"),
    ("w34", "1%", 0.7317, 0.7272, 0.40, "CloudPhysics"),
    ("w34", "5%", 0.5347, 0.5087, 0.40, "CloudPhysics"),
    ("w34", "10%", 0.4579, 0.4579, 0.05, "CloudPhysics"),
    ("w35", "1%", 0.1461, 0.1442, 1.03, "CloudPhysics"),
    ("w35", "5%", 0.0849, 0.0851, 1.10, "CloudPhysics"),
    ("w35", "10%", 0.0750, 0.0751, 1.37, "CloudPhysics"),
    ("w36", "1%", 0.1883, 0.1885, 2.06, "CloudPhysics"),
    ("w36", "5%", 0.1711, 0.1710, 3.64, "CloudPhysics"),
    ("w36", "10%", 0.1458, 0.1460, 2.98, "CloudPhysics"),
    ("w37", "1%", 0.7246, 0.7201, 1.66, "CloudPhysics"),
    ("w37", "5%", 0.5183, 0.5183, 0.00, "CloudPhysics"),
    ("w37", "10%", 0.4247, 0.4247, 0.00, "CloudPhysics"),
    ("w38", "1%", 0.6976, 0.6928, 1.12, "CloudPhysics"),
    ("w38", "5%", 0.6460, 0.6446, 0.86, "CloudPhysics"),
    ("w38", "10%", 0.5860, 0.5851, 0.08, "CloudPhysics"),
    ("w39", "1%", 0.1308, 0.1308, 1.87, "CloudPhysics"),
    ("w39", "5%", 0.1167, 0.1203, 2.14, "CloudPhysics"),
    ("w39", "10%", 0.0843, 0.0853, 3.41, "CloudPhysics"),
    ("w40", "1%", 0.1591, 0.1591, 1.51, "CloudPhysics"),
    ("w40", "5%", 0.1410, 0.1435, 2.18, "CloudPhysics"),
    ("w40", "10%", 0.1170, 0.1173, 2.93, "CloudPhysics"),
    ("w41", "1%", 0.1375, 0.1376, 1.71, "CloudPhysics"),
    ("w41", "5%", 0.0934, 0.0943, 0.91, "CloudPhysics"),
    ("w41", "10%", 0.0699, 0.0701, 1.62, "CloudPhysics"),
    ("w42", "1%", 0.7866, 0.7852, 0.52, "CloudPhysics"),
    ("w42", "5%", 0.7222, 0.7225, 0.11, "CloudPhysics"),
    ("w42", "10%", 0.6530, 0.6535, 0.10, "CloudPhysics"),
    # Meta CDN
    ("meta_rprn", "1%", 0.5301, 0.5299, 1.08, "Meta CDN"),
    ("meta_rprn", "5%", 0.4750, 0.4757, 1.06, "Meta CDN"),
    ("meta_rprn", "10%", 0.4447, 0.4454, 1.03, "Meta CDN"),
    # Meta KV
    ("meta_kvcache", "1%", 0.1565, 0.1569, 3.64, "Meta KV"),
    ("meta_kvcache", "5%", 0.0943, 0.0946, 2.81, "Meta KV"),
    ("meta_kvcache", "10%", 0.0765, 0.0769, 2.36, "Meta KV"),
    # Wiki CDN
    ("wiki_2019t", "1%", 0.4935, 0.4994, 4.40, "Wiki CDN"),
    ("wiki_2019t", "5%", 0.2633, 0.2638, 1.98, "Wiki CDN"),
    ("wiki_2019t", "10%", 0.1897, 0.1895, 0.91, "Wiki CDN"),
]

# Parse data
traces = [d[0] for d in data]
sieve_mr = np.array([d[2] for d in data])
forgive_mr = np.array([d[3] for d in data])
forgive_rate = np.array([d[4] for d in data])
trace_types = [d[5] for d in data]

# Calculate improvement (negative = better)
improvement = (forgive_mr - sieve_mr) / sieve_mr * 100

# Color mapping
colors = {
    "CloudPhysics": "blue",
    "Meta CDN": "red",
    "Meta KV": "green",
    "Wiki CDN": "orange"
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Base miss ratio vs New miss ratio
ax1 = axes[0]
for ttype in colors:
    mask = [t == ttype for t in trace_types]
    if any(mask):
        x = sieve_mr[mask]
        y = forgive_mr[mask]
        ax1.scatter(x, y, c=colors[ttype], label=ttype, alpha=0.7, s=60)

# Add diagonal line (y=x, no change)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No change')
ax1.set_xlabel('SIEVE Miss Ratio', fontsize=12)
ax1.set_ylabel('SieveEmbForgive Miss Ratio', fontsize=12)
ax1.set_title('Miss Ratio: SIEVE vs SieveEmbForgive (threshold=0.3)', fontsize=12)
ax1.legend()
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# Plot 2: Base miss ratio vs Forgive rate, colored by improvement
ax2 = axes[1]
for ttype in colors:
    mask = [t == ttype for t in trace_types]
    if any(mask):
        x = sieve_mr[mask]
        y = forgive_rate[mask]
        ax2.scatter(x, y, c=colors[ttype], label=ttype, alpha=0.7, s=60)

ax2.set_xlabel('SIEVE Miss Ratio', fontsize=12)
ax2.set_ylabel('Forgive Rate (%)', fontsize=12)
ax2.set_title('Base Miss Ratio vs Forgive Rate', fontsize=12)
ax2.legend()
ax2.set_xlim(0, 1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ak_docs/forgive_analysis_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# Also create a plot showing improvement vs base miss ratio
fig, ax = plt.subplots(figsize=(10, 6))

for ttype in colors:
    mask = [t == ttype for t in trace_types]
    if any(mask):
        x = sieve_mr[mask]
        y = improvement[mask]
        ax.scatter(x, y, c=colors[ttype], label=ttype, alpha=0.7, s=60)

ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='No change')
ax.set_xlabel('SIEVE Miss Ratio', fontsize=12)
ax.set_ylabel('Miss Ratio Change (%)', fontsize=12)
ax.set_title('Improvement vs Base Miss Ratio (negative = better)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ak_docs/forgive_improvement_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

print("Plots saved to ak_docs/forgive_analysis_scatter.png and ak_docs/forgive_improvement_scatter.png")

# Print summary stats
print("\n=== Summary by Trace Type ===")
for ttype in colors:
    mask = [t == ttype for t in trace_types]
    if any(mask):
        imp = improvement[mask]
        print(f"{ttype}: avg improvement = {imp.mean():.2f}%, min = {imp.min():.2f}%, max = {imp.max():.2f}%")
