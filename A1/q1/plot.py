import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# AI generated code detailed comparison plot

# Data
support = [5, 10, 25, 50, 90]
apriori_time = [250210.5670, 634.3822, 37.5633, 34.4578, 32.6978]
fp_growth_time = [175.6084, 116.7836, 35.5155, 33.5101, 32.6103]

fig = plt.figure(figsize=(10, 10))

# Create a 2x2 grid to place all the graphs
graph = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: Plot with log scale
ax1 = fig.add_subplot(graph[0, :])
ax1.plot(support, apriori_time, marker='o', linewidth=1.5, markersize=4, label='Apriori', color='blue')
ax1.plot(support, fp_growth_time, marker='o', linewidth=1.5, markersize=4, label='FP-Growth', color='orange')
ax1.set_yscale('log')
ax1.set_xlabel('Support Threshold', fontsize=11, fontweight='bold')
ax1.set_ylabel('Time (seconds) - Log Scale', fontsize=11, fontweight='bold')
ax1.set_title('Full Performance Comparison (Logarithmic Scale)', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(support)

# Add annotations for extreme values
ax1.annotate(f'{apriori_time[0]:.1f}s', 
             xy=(support[0], apriori_time[0]), 
             xytext=(10, 20), textcoords='offset points',
             fontsize=9, color='blue', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='blue', alpha=0.8))
ax1.annotate(f'{fp_growth_time[0]:.1f}s', 
             xy=(support[0], fp_growth_time[0]), 
             xytext=(10, -25), textcoords='offset points',
             fontsize=9, color='orange', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='orange', alpha=0.8))

# Plot 2: Detailed view plot
ax2 = fig.add_subplot(graph[1, :])
support_zoom = support[2:]  # Support 25, 50, 90
apriori_zoom = apriori_time[2:]
fp_growth_zoom = fp_growth_time[2:]

ax2.plot(support_zoom, apriori_zoom, marker='o', linewidth=1.5, markersize=4, label='Apriori', color='blue')
ax2.plot(support_zoom, fp_growth_zoom, marker='o', linewidth=1.5, markersize=4, label='FP-Growth', color='orange')
ax2.set_xlabel('Support Threshold', fontsize=11, fontweight='bold')
ax2.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax2.set_title('Detailed View: Support ≥ 25\n(Linear Scale)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.4, linestyle='--')
ax2.set_xticks(support_zoom)

# Add value labels
for i, (s, a, f) in enumerate(zip(support_zoom, apriori_zoom, fp_growth_zoom)):
    ax2.text(s, a, f'{a:.2f}s', ha='right', va='top', fontsize=9, color='blue', fontweight='bold')
    ax2.text(s, f, f'{f:.2f}s', ha='right', va='top', fontsize=9, color='orange', fontweight='bold')

# Set y limits with some padding to show differences
y_min = min(min(apriori_zoom), min(fp_growth_zoom)) - 1
y_max = max(max(apriori_zoom), max(fp_growth_zoom)) + 1
ax2.set_ylim(y_min, y_max)



plt.savefig('plot.png', dpi=300, bbox_inches='tight')
print("Detailed comparison plot saved!")