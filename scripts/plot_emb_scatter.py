#!/usr/bin/env python3
"""
Scatter plot of embedding similarity vs reuse distance
"""

import numpy as np
import matplotlib.pyplot as plt

# Read the data
data = np.genfromtxt('/tmp/emb_diag_samples.csv', delimiter=',', skip_header=1)
similarity = data[:, 0]
reuse_distance = data[:, 1]
frequency = data[:, 2]
recency = data[:, 3]

print(f"Loaded {len(similarity)} samples")
print(f"Similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")
print(f"Reuse distance range: [{reuse_distance.min():.0f}, {reuse_distance.max():.0f}]")

# Create figure with multiple plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Scatter plot (subsample for visibility)
ax1 = axes[0, 0]
sample_size = min(10000, len(similarity))
np.random.seed(42)
idx = np.random.choice(len(similarity), sample_size, replace=False)
ax1.scatter(similarity[idx], reuse_distance[idx], alpha=0.3, s=5)
ax1.set_xlabel('Embedding Similarity')
ax1.set_ylabel('Reuse Distance')
ax1.set_title(f'Similarity vs Reuse Distance (n={sample_size} sampled)')

# Add correlation line
z = np.polyfit(similarity, reuse_distance, 1)
p = np.poly1d(z)
x_line = np.linspace(similarity.min(), similarity.max(), 100)
ax1.plot(x_line, p(x_line), "r--", alpha=0.8)
corr = np.corrcoef(similarity, reuse_distance)[0, 1]
ax1.legend([f'r = {corr:.4f}'])

# 2. Hexbin density plot
ax2 = axes[0, 1]
hb = ax2.hexbin(similarity, reuse_distance, gridsize=50, cmap='Blues', mincnt=1)
ax2.set_xlabel('Embedding Similarity')
ax2.set_ylabel('Reuse Distance')
ax2.set_title('Density Plot (all samples)')
plt.colorbar(hb, ax=ax2, label='Count')

# 3. Binned means with error bars
ax3 = axes[1, 0]
n_bins = 20
bin_edges = np.linspace(similarity.min(), similarity.max(), n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_means = []
bin_stds = []
bin_counts = []

for i in range(n_bins):
    mask = (similarity >= bin_edges[i]) & (similarity < bin_edges[i+1])
    if i == n_bins - 1:  # Include right edge for last bin
        mask = (similarity >= bin_edges[i]) & (similarity <= bin_edges[i+1])
    if mask.sum() >= 10:
        bin_means.append(reuse_distance[mask].mean())
        bin_stds.append(reuse_distance[mask].std())
        bin_counts.append(mask.sum())
    else:
        bin_means.append(np.nan)
        bin_stds.append(np.nan)
        bin_counts.append(0)

bin_means = np.array(bin_means)
bin_stds = np.array(bin_stds)
bin_counts = np.array(bin_counts)

valid = ~np.isnan(bin_means)
ax3.errorbar(bin_centers[valid], bin_means[valid],
             yerr=bin_stds[valid]/np.sqrt(bin_counts[valid]),
             fmt='o-', capsize=3, markersize=5)
ax3.set_xlabel('Embedding Similarity (binned)')
ax3.set_ylabel('Mean Reuse Distance')
ax3.set_title('Binned Mean Reuse Distance (±SE)')
ax3.axhline(y=reuse_distance.mean(), color='r', linestyle='--', alpha=0.5, label='Overall mean')
ax3.legend()

# 4. Distribution of reuse distance by similarity quartile
ax4 = axes[1, 1]
quartiles = np.percentile(similarity, [25, 50, 75])
q_labels = ['Q1\n(low sim)', 'Q2', 'Q3', 'Q4\n(high sim)']
q_data = [
    reuse_distance[similarity <= quartiles[0]],
    reuse_distance[(similarity > quartiles[0]) & (similarity <= quartiles[1])],
    reuse_distance[(similarity > quartiles[1]) & (similarity <= quartiles[2])],
    reuse_distance[similarity > quartiles[2]]
]

# Cap at 99th percentile for visibility
cap = np.percentile(reuse_distance, 99)
q_data_capped = [np.clip(d, 0, cap) for d in q_data]

bp = ax4.boxplot(q_data_capped, labels=q_labels)
ax4.set_xlabel('Similarity Quartile')
ax4.set_ylabel('Reuse Distance (capped at 99th pct)')
ax4.set_title('Reuse Distance Distribution by Similarity Quartile')

# Add mean markers
means = [np.mean(d) for d in q_data]
ax4.scatter(range(1, 5), means, color='red', marker='D', s=50, zorder=3, label='Mean')
ax4.legend()

plt.tight_layout()
plt.savefig('/home/aryan/ecache/libCacheSim/plots/emb_vs_reuse_scatter.png', dpi=150)
print(f"\nSaved plot to /home/aryan/ecache/libCacheSim/plots/emb_vs_reuse_scatter.png")

# Print summary stats
print("\n=== Summary Statistics ===")
print(f"Overall correlation: {corr:.4f}")
print("\nBy quartile:")
for i, label in enumerate(['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']):
    print(f"  {label}: mean_reuse={np.mean(q_data[i]):.1f}, median={np.median(q_data[i]):.1f}, n={len(q_data[i])}")
