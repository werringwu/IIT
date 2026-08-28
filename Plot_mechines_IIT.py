# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# 填入你精心挑选的那一组数据的 json 路径
JSON_PATH = 'path/to/extreme_cases/Top1_stats.json'
OUTPUT_IMG = 'figB_mechanism_dissection.pdf'

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 11

def plot_case_mechanism(json_path, output_path):
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    v_m = data['v_m']
    weights = data['weights']
    gamma = data['gamma']
    P_before = data['P_before']
    P_after = data['P_after']
    
    base_weights = [2.0, 1.0, 0.5, 0.25]
    M = len(weights)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    # 1. Scale-wise Projection (v_m)
    ax1 = axes[0]
    scales = [f'$L_{i}$' if i < M-1 else 'Anchor' for i in range(M)]
    colors = ['#e63946' if v < 0 else '#457b9d' for v in v_m]
    
    ax1.bar(scales, v_m, color=colors, alpha=0.8, edgecolor='black')
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_title('Scale-wise Projection ($v_m$)')
    ax1.set_ylabel(r'$\langle \mathbf{g}_m, \mathbf{g}_{\mathrm{anchor}} \rangle$')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 2. Weight Adjustment (a_m^0 -> a_m)
    ax2 = axes[1]
    x = np.arange(M)
    width = 0.35
    
    ax2.bar(x - width/2, base_weights, width, label='Before (Base)', color='#a8dadc', edgecolor='black')
    ax2.bar(x + width/2, weights, width, label='After SCA', color='#1d3557', edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(scales)
    ax2.set_title(f'SCA Weight Surgery ($\gamma={gamma:.3f}$)')
    ax2.set_ylabel('Effective Task Weight')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 3. Aggregate Projection
    ax3 = axes[2]
    # 如果 P_after 勉强被救回到 0，给个稍微不同的颜色显示它的“保底成功”
    p_colors = ['#e63946', '#2a9d8f' if P_after >= -1e-6 else '#f4a261']
    ax3.bar(['$P_{\mathrm{before}}$', '$P_{\mathrm{after}}$'], [P_before, P_after], 
            color=p_colors, width=0.5, edgecolor='black')
    ax3.axhline(0, color='black', linewidth=1)
    ax3.set_title('Aggregate Projection ($P$)')
    ax3.set_ylabel(r'$\mathbf{g}_{\mathrm{anchor}}^{\top} \sum a_m \mathbf{g}_m$')
    
    ax3.text(0, P_before, f'{P_before:.2e}', ha='center', va='bottom' if P_before>0 else 'top', fontsize=10)
    ax3.text(1, P_after, f'{P_after:.2e}', ha='center', va='bottom' if P_after>0 else 'top', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Mechanism plot saved to {output_path}")

if __name__ == "__main__":
    plot_case_mechanism(JSON_PATH, OUTPUT_IMG)