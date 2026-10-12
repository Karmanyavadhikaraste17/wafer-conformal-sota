import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import cfg

sns.set_theme(style="whitegrid", font_scale=1.1)

def plot_selective_classification(triage_df, save_path=None):
    """
    Plots the Conformal Fab Triage trade-off:
    Automation Rate vs Automated Accuracy & Human Workload Reduction.
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "selective_classification_curve.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    color_auto = '#1f77b4'
    color_acc = '#2ca02c'

    # Plot Automation Rate
    ax1.set_xlabel("Significance Level $\\alpha$ (Error Tolerance)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Automation Rate (%) [Zero-Human Flow]", color=color_auto, fontsize=12, fontweight='bold')
    line1 = ax1.plot(triage_df['alpha'], triage_df['automation_rate'] * 100.0, marker='o', color=color_auto, linewidth=2.5, label='Automation Rate (%)')
    ax1.tick_params(axis='y', labelcolor=color_auto)
    ax1.set_ylim(0, 100)

    # Secondary axis: Accuracy on automated subset
    ax2 = ax1.twinx()
    ax2.set_ylabel("Precision on Automated Subset (%)", color=color_acc, fontsize=12, fontweight='bold')
    line2 = ax2.plot(triage_df['alpha'], triage_df['automated_accuracy'] * 100.0, marker='s', color=color_acc, linewidth=2.5, linestyle='--', label='Automated Precision (%)')
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(95.0, 100.5)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', framealpha=0.9)

    plt.title("Conformal Selective Classification: Automated Throughput vs Precision", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_ood_rejection(ood_results, ood_confs, save_path=None):
    """
    Visualizes OOD detection: Baseline Softmax Hallucination vs Conformal Empty-Set Rejection.
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "ood_rejection_vs_hallucination.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Baseline Hallucination Confidence Histogram
    sns.histplot(ood_confs, bins=25, kde=True, ax=ax1, color='#d9534f')
    ax1.axvline(np.mean(ood_confs), color='darkred', linestyle='--', linewidth=2, label=f'Mean Conf: {np.mean(ood_confs)*100:.1f}%')
    ax1.set_xlabel("Baseline Softmax Confidence on OOD Wafers", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Number of OOD Samples", fontsize=11, fontweight='bold')
    ax1.set_title("Standard Baselines (Hallucinates on OOD)", fontsize=12, fontweight='bold')
    ax1.legend()

    # Right: Comparison Bar Chart
    categories = ['Standard Argmax\n(Published Papers)', 'Conformal Prediction\n(Our Approach)']
    hallucination_pct = [100.0, (1.0 - ood_results['ood_conformal_rejection_rate']) * 100.0]
    rejection_pct = [0.0, ood_results['ood_conformal_rejection_rate'] * 100.0]

    bar_width = 0.4
    x = np.arange(len(categories))

    ax2.bar(x - bar_width/2, hallucination_pct, width=bar_width, label='Silent Hallucination (%)', color='#d9534f')
    ax2.bar(x + bar_width/2, rejection_pct, width=bar_width, label='Empty-Set Rejection ($C(X)=\\emptyset$)', color='#5cb85c')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.set_ylabel("Percentage (%)", fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 115)
    ax2.set_title("OOD / Novel Defect Safety Comparison", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')

    for i in range(len(x)):
        ax2.text(x[i] - bar_width/2, hallucination_pct[i] + 2, f"{hallucination_pct[i]:.1f}%", ha='center', fontweight='bold')
        ax2.text(x[i] + bar_width/2, rejection_pct[i] + 2, f"{rejection_pct[i]:.1f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_mondrian_parity(mondrian_df, save_path=None):
    """
    Plots Coverage Parity: Group-Conditional Mondrian vs Marginal Conformal across Mix-Depths.
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "mondrian_vs_marginal_coverage.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(mondrian_df))
    width = 0.35

    ax.bar(x - width/2, mondrian_df['marginal_coverage'] * 100.0, width, label='Marginal RAPS Coverage (%)', color='#4a90e2')
    ax.bar(x + width/2, mondrian_df['mondrian_coverage'] * 100.0, width, label='Mondrian Group Coverage (%)', color='#50e3c2')

    ax.axhline(90.0, color='red', linestyle='--', linewidth=2, label='Target Coverage Guarantee (90%)')

    ax.set_xlabel("Wafer Defect Mix Depth (0: Normal to 4: Quadruple)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Empirical Coverage (%)", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Depth {int(d)}" for d in mondrian_df['mix_depth']], fontsize=11)
    ax.set_ylim(85.0, 102.0)
    ax.set_title("Mondrian vs Marginal Conformal Coverage Parity by Defect Depth", fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_cost_sensitivity(sensitivity_df, save_path=None):
    """
    Plots Financial Loss Reduction across Escape Cost Ratios ($C_{escape} / C_{FA}$).
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "fab_economic_risk_sensitivity.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(9, 6))

    plt.plot(sensitivity_df['cost_ratio'], sensitivity_df['baseline_loss'] / 1000.0, marker='o', color='#d9534f', linewidth=2.5, label='Standard Argmax Policy ($1000s Loss)')
    plt.plot(sensitivity_df['cost_ratio'], sensitivity_df['conformal_loss'] / 1000.0, marker='^', color='#5cb85c', linewidth=2.5, label='Conformal Risk-Controlled Policy ($1000s Loss)')

    plt.xlabel("Escape Cost Severity Ratio ($C_{\\text{escape}} / C_{\\text{FA}}$)", fontsize=12, fontweight='bold')
    plt.ylabel("Total Fab Financial Loss ($k)", fontsize=12, fontweight='bold')
    plt.title("Semiconductor Fab Operational Loss: Argmax vs Conformal Risk Policy", fontsize=13, fontweight='bold', pad=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')

    # Add text annotation of reduction
    mean_red = sensitivity_df['loss_reduction_pct'].mean()
    plt.annotate(
        f"Average Loss Reduction: {mean_red:.1f}%",
        xy=(sensitivity_df['cost_ratio'].iloc[-2], sensitivity_df['conformal_loss'].iloc[-2] / 1000.0),
        xytext=(sensitivity_df['cost_ratio'].iloc[-4], sensitivity_df['baseline_loss'].iloc[-3] / 1000.0),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontweight='bold', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
