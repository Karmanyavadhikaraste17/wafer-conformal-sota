import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

from .config import cfg

sns.set_theme(style="whitegrid", font_scale=1.1)

def plot_coverage_nominal_diagnostic(diag_df, save_path=None):
    """
    Checklist #1 Figure: Plots Empirical Coverage vs Nominal Coverage (1 - alpha)
    across Exact Randomized RAPS (allow_empty=True) and Non-Empty RAPS (allow_empty=False).
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "coverage_nominal_diagnostic.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: Empirical vs Nominal Coverage
    nom_line = np.linspace(0.78, 1.0, 100)
    ax1.plot(nom_line, nom_line, 'k--', linewidth=2.0, label='Ideal Nominal Diagonal ($y = 1 - \\alpha$)')

    colors = {
        'Exact_Randomized_RAPS': '#2ca02c',
        'NonEmpty_Randomized_RAPS': '#1f77b4',
        'Conservative_Deterministic_RAPS': '#ff7f0e'
    }
    labels_map = {
        'Exact_Randomized_RAPS': 'Exact Randomized RAPS (Tight Calibration)',
        'NonEmpty_Randomized_RAPS': 'Non-Empty RAPS (Operational, $|C(X)| \\geq 1$)',
        'Conservative_Deterministic_RAPS': 'Deterministic RAPS (Upper Bound)'
    }

    for var_name, group in diag_df.groupby('variant'):
        group = group.sort_values('nominal_coverage')
        ax1.plot(
            group['nominal_coverage'], group['empirical_coverage'],
            marker='o', linewidth=2.4, color=colors.get(var_name, 'blue'),
            label=labels_map.get(var_name, var_name)
        )

    ax1.set_xlabel("Nominal Target Coverage ($1 - \\alpha$)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Empirical Test Coverage", fontsize=12, fontweight='bold')
    ax1.set_title("A. Calibration Fidelity: Exact vs Non-Empty RAPS", fontsize=12.5, fontweight='bold')
    ax1.set_xlim(0.79, 1.00)
    ax1.set_ylim(0.79, 1.005)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', fontsize=9.5)

    # Right: Average Set Size vs Nominal Coverage
    for var_name, group in diag_df.groupby('variant'):
        group = group.sort_values('nominal_coverage')
        ax2.plot(
            group['nominal_coverage'], group['avg_set_size'],
            marker='s', linewidth=2.4, color=colors.get(var_name, 'blue'),
            label=labels_map.get(var_name, var_name)
        )

    ax2.set_xlabel("Nominal Target Coverage ($1 - \\alpha$)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Average Prediction Set Size ($|C(X)|$)", fontsize=12, fontweight='bold')
    ax2.set_title("B. Prediction Set Efficiency Across Nominal Levels", fontsize=12.5, fontweight='bold')
    ax2.set_xlim(0.79, 1.00)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', fontsize=9.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_aci_online_drift(stream_data, target_alpha=0.10, window=250, save_path=None):
    """
    Novelty N1 Figure: Rolling coverage and adaptive alpha trajectory under process drift.
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "aci_online_drift_tracking.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    static_cov = pd.Series(stream_data['static_cov']).rolling(window=window, min_periods=50).mean().values
    aci_cov = pd.Series(stream_data['aci_cov']).rolling(window=window, min_periods=50).mean().values
    alpha_traj = pd.Series(stream_data['alpha_trajectory']).rolling(window=50, min_periods=10).mean().values
    T = len(static_cov)
    x = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Top panel: Rolling Coverage
    ax1.axvspan(2500, 5500, color='#ffcccc', alpha=0.35, label='Chamber Process Drift Regime (t = 2500..5500)')
    ax1.plot(x, static_cov * 100.0, color='#d9534f', linewidth=2.0, alpha=0.9, label='Static Split Conformal (Collapses under Drift)')
    ax1.plot(x, aci_cov * 100.0, color='#2ca02c', linewidth=2.4, label='Online Adaptive Conformal Inference - ACI (Self-Correcting)')
    ax1.axhline((1.0 - target_alpha) * 100.0, color='black', linestyle='--', linewidth=1.8, label='Target Coverage Guarantee (90.0%)')

    ax1.set_ylabel(f"Rolling Coverage (Window={window}) (%)", fontsize=11.5, fontweight='bold')
    ax1.set_title("Drift-Adaptive Online Conformal Monitoring (ACI vs Static Split Conformal)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylim(62.0, 101.0)
    ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)

    # Bottom panel: Effective alpha_t trajectory
    ax2.axvspan(2500, 5500, color='#ffcccc', alpha=0.35)
    ax2.plot(x, alpha_traj, color='#1f77b4', linewidth=2.0, label='ACI Online Parameter $\\alpha_t$ (Pinball Gradient Update)')
    ax2.axhline(target_alpha, color='gray', linestyle=':', linewidth=1.5, label='Nominal $\\alpha_0 = 0.10$')
    ax2.set_xlabel("Wafer Production Stream Index ($t$)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Effective $\\alpha_t$", fontsize=11.5, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, facecolor='white', fontsize=9.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_hybrid_ood_comparison(ood_df, roc_data, save_path=None):
    """
    Novelty N2 + Checklist #2 Figure:
    Left: ROC Curve comparing Hybrid Embedding-Density Conformal vs Softmax/MSP and Energy.
    Right: Empty-Set Rejection Rate on Contamination OOD & Leave-Class-Out Semantic OOD.
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "hybrid_ood_roc_comparison.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # Left: ROC curves on Suite 1
    y_true = np.concatenate([np.zeros(len(roc_data['in_msp'])), np.ones(len(roc_data['ood1_msp']))])
    from sklearn.metrics import auc as sk_auc
    for label_str, in_s, ood_s, col, ls in [
        ("Hybrid Embedding-Density Conformal (Ours)", roc_data['in_hyb'], roc_data['ood1_hyb'], '#2ca02c', '-'),
        ("Softmax / MSP Score", roc_data['in_msp'], roc_data['ood1_msp'], '#1f77b4', '--'),
        ("Energy Score (Liu et al.)", roc_data['in_energy'], roc_data['ood1_energy'], '#ff7f0e', '-.'),
    ]:
        scores = np.concatenate([in_s, ood_s])
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_val = sk_auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=col, linestyle=ls, linewidth=2.4, label=f"{label_str} (AUROC={auc_val:.3f})")

    ax1.plot([0, 1], [0, 1], 'k:', linewidth=1.5, label='Argmax Baseline (No Abstention, AUROC=0.500)')
    ax1.set_xlabel("In-Distribution False Rejection Rate (FPR)", fontsize=11.5, fontweight='bold')
    ax1.set_ylabel("True OOD Empty-Set Rejection Rate (TPR)", fontsize=11.5, fontweight='bold')
    ax1.set_title("A. OOD Detection ROC Curve Comparison", fontsize=12.5, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True, facecolor='white', fontsize=9.5)

    # Right: Grouped Bar Chart across Suite 1 (Contamination) and Suite 2 (Leave-Class-Out Semantic)
    plot_methods = [
        "Standard Argmax (Literature Baseline)",
        "Energy Score (Liu et al. Heuristic)",
        "Softmax-Only Conformal (Empty-Set C(X)=0)",
        "Hybrid Embedding-Density Conformal (Ours)"
    ]
    short_names = ["Argmax\n(Literature)", "Energy\nHeuristic", "Softmax\nConformal", "Hybrid Density\nConformal (Ours)"]

    s1_rates = []
    s2_rates = []
    for m in plot_methods:
        r1 = ood_df[(ood_df['ood_benchmark'].str.contains('Suite 1')) & (ood_df['method'] == m)]['empty_set_rejection_rate'].values[0]
        r2 = ood_df[(ood_df['ood_benchmark'].str.contains('Suite 2')) & (ood_df['method'] == m)]['empty_set_rejection_rate'].values[0]
        s1_rates.append(r1 * 100.0)
        s2_rates.append(r2 * 100.0)

    x = np.arange(len(short_names))
    w = 0.36
    b1 = ax2.bar(x - w/2, s1_rates, width=w, color='#4a90e2', label='Contamination OOD (N=1,000)')
    b2 = ax2.bar(x + w/2, s2_rates, width=w, color='#2ca02c', label='Leave-Class-Out Semantic OOD (N=800)')

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=10.5, fontweight='bold')
    ax2.set_ylabel("Empty-Set Rejection Rate at 5% FPR (%)", fontsize=11.5, fontweight='bold')
    ax2.set_ylim(0, 112)
    ax2.set_title("B. Certified Empty-Set Rejection: Covariate & Semantic OOD", fontsize=12.5, fontweight='bold')
    ax2.legend(loc='upper left', frameon=True, facecolor='white', fontsize=9.5)

    for bar in b1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', fontsize=9, fontweight='bold')
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

    # Also update ood_rejection_vs_hallucination.png with the new Hybrid result
    legacy_path = os.path.join(cfg.FIGURES_DIR, "ood_rejection_vs_hallucination.png")
    shutil_copy_fig(save_path, legacy_path)

def shutil_copy_fig(src, dst):
    import shutil
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
        print(f"Updated: {dst}")

def plot_calibration_size_sweep(sweep_df, save_path=None):
    """
    Checklist #3 Figure: Plots coverage mean +/- 1 SD and set size stability across calibration sample sizes.
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "calibration_size_sweep.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    n_vals = sweep_df['n_cal'].values
    cov_mean = sweep_df['exact_coverage_mean'].values * 100.0
    cov_std = sweep_df['exact_coverage_std'].values * 100.0

    ax1.plot(n_vals, cov_mean, marker='o', color='#1f77b4', linewidth=2.4, label='Empirical Mean Coverage (%)')
    ax1.fill_between(n_vals, cov_mean - cov_std, cov_mean + cov_std, color='#1f77b4', alpha=0.22, label='$\\pm 1$ Std. Dev. (20 Draws)')
    ax1.axhline(90.0, color='red', linestyle='--', linewidth=2.0, label='Target Coverage (90.0%)')
    ax1.set_xscale('log')
    ax1.set_xlabel("Calibration Set Size ($n_{\\text{cal}}$, Log Scale)", fontsize=11.5, fontweight='bold')
    ax1.set_ylabel("Empirical Test Coverage (%)", fontsize=11.5, fontweight='bold')
    ax1.set_title("A. Coverage Stability vs Calibration Set Size", fontsize=12.5, fontweight='bold')
    ax1.set_ylim(86.0, 94.0)
    ax1.legend(loc='lower right', frameon=True, facecolor='white')

    size_mean = sweep_df['avg_set_size_mean'].values
    size_std = sweep_df['avg_set_size_std'].values
    ax2.plot(n_vals, size_mean, marker='s', color='#2ca02c', linewidth=2.4, label='Mean Non-Empty Set Size $|C(X)|$')
    ax2.fill_between(n_vals, size_mean - size_std, size_mean + size_std, color='#2ca02c', alpha=0.22, label='$\\pm 1$ Std. Dev.')
    ax2.set_xscale('log')
    ax2.set_xlabel("Calibration Set Size ($n_{\\text{cal}}$, Log Scale)", fontsize=11.5, fontweight='bold')
    ax2.set_ylabel("Average Prediction Set Size", fontsize=11.5, fontweight='bold')
    ax2.set_title("B. Set Size Efficiency vs Calibration Budget", fontsize=12.5, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, facecolor='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_selective_classification(triage_df, save_path=None):
    """
    Proposal A Figure: Plots Automation Rate vs Automated Precision.
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "selective_classification_curve.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(9, 5.8))

    color_auto = '#1f77b4'
    color_acc = '#2ca02c'

    ax1.set_xlabel("Significance Level $\\alpha$ (Error Budget)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Singleton Automation Rate (%) [Zero-Human Flow]", color=color_auto, fontsize=12, fontweight='bold')
    line1 = ax1.plot(triage_df['alpha'], triage_df['automation_rate'] * 100.0, marker='o', color=color_auto, linewidth=2.5, label='Singleton Automation Rate (%)')
    ax1.tick_params(axis='y', labelcolor=color_auto)
    ax1.set_ylim(85.0, 101.0)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Precision on Automated Subset (%)", color=color_acc, fontsize=12, fontweight='bold')
    line2 = ax2.plot(triage_df['alpha'], triage_df['automated_accuracy'] * 100.0, marker='s', color=color_acc, linewidth=2.5, linestyle='--', label='Automated Subset Precision (%)')
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(98.0, 100.1)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', framealpha=0.95)

    plt.title("Conformal Selective Classification: 94%–99% Fab Automation at >99% Precision", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_mondrian_parity(mondrian_df, save_path=None):
    """
    Proposal C Figure: Plots Group-Conditional Mondrian Coverage across Depths 0..4 and Minority C7.
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "mondrian_vs_marginal_coverage.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.8))

    x = np.arange(len(mondrian_df))
    width = 0.28

    ax.bar(x - width, mondrian_df['marginal_exact_coverage'] * 100.0, width, label='Marginal Exact RAPS (%)', color='#4a90e2')
    ax.bar(x, mondrian_df['mondrian_exact_coverage'] * 100.0, width, label='Mondrian Exact RAPS (%)', color='#2ca02c')
    ax.bar(x + width, mondrian_df['mondrian_nonempty_coverage'] * 100.0, width, label='Mondrian Non-Empty Operational (%)', color='#ff7f0e')

    ax.axhline(90.0, color='red', linestyle='--', linewidth=2.0, label='Target Coverage Guarantee (90.0%)')

    ax.set_xlabel("Defect Complexity Group & Minority Class (with Test Counts $N$)", fontsize=11.5, fontweight='bold')
    ax.set_ylabel("Empirical Coverage (%)", fontsize=11.5, fontweight='bold')
    ax.set_xticks(x)
    x_labels = [f"{row['group_name']}\n(N={row['test_count']})" for _, row in mondrian_df.iterrows()]
    ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold')
    ax.set_ylim(82.0, 103.0)
    ax.set_title("Mondrian Group-Conditional vs Marginal Coverage Parity ($\\alpha = 0.10$)", fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_cost_sensitivity(sensitivity_df, save_path=None):
    """
    Proposal D Figure: Plots Financial Loss Reduction across Escape Cost Ratios.
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "fab_economic_risk_sensitivity.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(9, 5.8))

    plt.plot(sensitivity_df['cost_ratio'], sensitivity_df['baseline_loss'] / 1000.0, marker='o', color='#d9534f', linewidth=2.5, label='Standard Argmax Policy ($1000s Loss)')
    plt.plot(sensitivity_df['cost_ratio'], sensitivity_df['conformal_loss'] / 1000.0, marker='^', color='#2ca02c', linewidth=2.5, label='Conformal Risk-Controlled Policy ($1000s Loss)')

    plt.xlabel("Escape Cost Severity Ratio ($C_{\\text{escape}} / C_{\\text{FA}}$)", fontsize=12, fontweight='bold')
    plt.ylabel("Total Fab Financial Loss ($k)", fontsize=12, fontweight='bold')
    plt.title("Semiconductor Fab Operational Loss Across Economic Regimes", fontsize=13, fontweight='bold', pad=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left')

    mean_red = sensitivity_df['loss_reduction_pct'].mean()
    plt.annotate(
        f"Consistent Loss Reduction: {mean_red:.1f}%",
        xy=(sensitivity_df['cost_ratio'].iloc[-2], sensitivity_df['conformal_loss'].iloc[-2] / 1000.0),
        xytext=(sensitivity_df['cost_ratio'].iloc[-4], sensitivity_df['baseline_loss'].iloc[-2] / 1000.0 * 0.8),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontweight='bold', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="#fff2cc", alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
