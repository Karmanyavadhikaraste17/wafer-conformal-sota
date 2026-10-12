import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import cfg

sns.set_theme(style="whitegrid", font_scale=1.1)

def plot_coverage_vs_alpha(conformal_df, save_path=None):
    """
    Plots empirical coverage vs nominal coverage (1 - alpha)
    for all models and conformal methods (APS vs RAPS).
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "coverage_vs_alpha.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    
    # Nominal reference line
    nominal_vals = np.array([0.90, 0.95, 0.99])
    plt.plot(nominal_vals, nominal_vals, 'k--', label='Nominal Coverage (Ideal)', linewidth=2.0)

    # Plot each model / method curve
    palette = sns.color_palette("tab10")
    color_idx = 0

    for (model, method), group in conformal_df.groupby(['model', 'method']):
        # Map alpha to 1 - alpha
        group = group.sort_values(by='alpha', ascending=False)
        nom = 1.0 - group['alpha'].values
        cov = group['marginal_coverage'].values

        plt.plot(
            nom, cov, marker='o', linewidth=2.2,
            label=f"{model.upper()} ({method.upper()})",
            color=palette[color_idx % len(palette)]
        )
        color_idx += 1

    plt.xlabel("Nominal Coverage Level (1 - $\\alpha$)", fontsize=13, fontweight='bold')
    plt.ylabel("Empirical Marginal Coverage", fontsize=13, fontweight='bold')
    plt.title("Conformal Prediction: Empirical vs Nominal Coverage on MixedWM38", fontsize=14, fontweight='bold', pad=12)
    plt.xlim(0.88, 1.00)
    plt.ylim(0.88, 1.01)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_coverage_by_depth(depth_df, save_path=None):
    """
    Grouped bar chart showing empirical coverage broken down by mix-depth
    (Depth 0: Normal, 1: Single, 2: Double, 3: Triple, 4: Quadruple) at alpha=0.10.
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "coverage_by_depth.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(
        data=depth_df,
        x='mix_depth',
        y='coverage',
        hue='model',
        palette="crest"
    )

    # 90% target threshold
    plt.axhline(0.90, color='red', linestyle='--', linewidth=2.2, label='Target Coverage (90%)')

    plt.xlabel("Wafer Defect Mix Depth (0: Normal, 1: Single, 2: Double, 3: Triple, 4: Quad)", fontsize=12, fontweight='bold')
    plt.ylabel("Empirical Coverage", fontsize=12, fontweight='bold')
    plt.title("Conformal Coverage by Defect Mixture Complexity ($\\alpha$ = 0.10)", fontsize=14, fontweight='bold', pad=12)
    plt.ylim(0.80, 1.02)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_setsize_by_depth(depth_df, save_path=None):
    """
    Bar chart showing average prediction set size by mix depth.
    Demonstrates the expansion of prediction sets for complex mixtures.
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "setsize_by_depth.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(
        data=depth_df,
        x='mix_depth',
        y='avg_set_size',
        hue='model',
        palette="viridis"
    )

    plt.xlabel("Wafer Defect Mix Depth (0: Normal, 1: Single, 2: Double, 3: Triple, 4: Quad)", fontsize=12, fontweight='bold')
    plt.ylabel("Average Conformal Set Size (|C(X)|)", fontsize=12, fontweight='bold')
    plt.title("Prediction Set Efficiency vs Defect Mixture Complexity ($\\alpha$ = 0.10)", fontsize=14, fontweight='bold', pad=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_noise_study(noise_df, save_path=None):
    """
    Plots the robustness finding:
    Accuracy vs Noise Rate vs Uncalibrated/Noisy Coverage vs Clean-Recalibrated Coverage.
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "noise_accuracy_vs_coverage.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(9, 6))
    noise_pct = noise_df['noise_rate'] * 100

    plt.plot(noise_pct, noise_df['test_accuracy'] * 100, marker='s', color='#2b5c8f', linewidth=2.5, label='Test Accuracy (%)')
    plt.plot(noise_pct, noise_df['noisy_cal_coverage'] * 100, marker='o', color='#d9534f', linewidth=2.5, label='Noisy Calibration Coverage (%) [Collapses]')
    plt.plot(noise_pct, noise_df['clean_cal_coverage'] * 100, marker='^', color='#5cb85c', linewidth=2.5, label='Clean Recalibrated Coverage (%) [Restored]')
    plt.axhline(90.0, color='gray', linestyle=':', linewidth=1.8, label='90% Target Coverage Guarantee')

    plt.xlabel("Symmetric Label Noise Rate (%)", fontsize=12, fontweight='bold')
    plt.ylabel("Percentage (%)", fontsize=12, fontweight='bold')
    plt.title("Label Noise Robustness: Accuracy Hides Failure, Conformal Recalibration Repairs It", fontsize=13, fontweight='bold', pad=12)
    plt.ylim(65.0, 102.0)
    plt.legend(frameon=True, facecolor='white', framealpha=0.95, loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_set_size_examples(test_dataset, class_df, pred_sets, example_indices, save_path=None):
    """
    Visualizes 5 example wafer maps with their prediction sets rendered as captions.
    """
    if save_path is None:
        save_path = os.path.join(cfg.FIGURES_DIR, "set_size_examples.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))

    for idx, sample_idx in enumerate(example_indices):
        item = test_dataset[sample_idx]
        img_np = item['image'].numpy()
        true_label = item['label'].item()
        true_depth = item['depth'].item()
        pred_set = pred_sets[sample_idx]

        display_map = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.float32)
        display_map[img_np[0] == 1] = [0.12, 0.12, 0.12]
        display_map[img_np[1] == 1] = [0.20, 0.70, 0.90]
        display_map[img_np[2] == 1] = [0.95, 0.20, 0.20]

        true_name = class_df.loc[class_df['class_idx'] == true_label, 'class_name'].values[0]
        set_ids = [class_df.loc[class_df['class_idx'] == c, 'class_id'].values[0] for c in pred_set]

        ax = axes[idx]
        ax.imshow(display_map)
        ax.set_title(
            f"True: {true_name} (D{true_depth})\n"
            f"Set ({len(pred_set)}): {set_ids}",
            fontsize=9.5, pad=6
        )
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
