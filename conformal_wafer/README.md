# Conformal Wafer: Guaranteed-Coverage Prediction Sets, Online Drift Monitoring & Hybrid OOD Rejection for Mixed-Type Wafer Defect Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end ML research benchmark providing the **first split conformal, online drift-adaptive (ACI), and hybrid embedding-density OOD-rejecting framework** on the **MixedWM38** semiconductor wafer defect benchmark (38 defect classes, 38,015 wafer maps).

---

## 🔬 Key Research Highlights

1. **Tri-Architecture + Conformal Ensemble Benchmark:** Reimplements **DAP-Net** (*IEEE Access* 2025, **98.54%** test accuracy with only 875K parameters) alongside **ResNet-18** (**98.65%**), **ViT-Tiny** (**91.38%**), and a **Cross-Architecture Conformal Ensemble** (**98.71%**).
2. **Exact Randomized vs. Operational Non-Empty Conformal Sets (Checklist #1):**
   - **Exact Randomized RAPS (`allow_empty=True`):** Tracks nominal coverage $1-\alpha$ tightly across $\alpha \in [0.01, 0.20]$ (**99.17%, 95.15%, 90.41%** at nominal 99%, 95%, 90%).
   - **Operational Non-Empty RAPS (`allow_empty=False`):** Guarantees at least one candidate per wafer, achieving **94.50%–99.09% singleton automation** with **1.009–1.056 average set size**.
3. **Novelty N1 — Drift-Adaptive Online Conformal Monitoring (ACI, Gibbs & Candès 2021):**
   - Under mid-stream chamber process drift ($t \in [2501, 5500]$), Static Split Conformal coverage collapses to **79.93%**, while **Online ACI** self-corrects via pinball loss updates to **89.97%** during drift and **90.02%** across all 7,603 wafers without retraining.
4. **Novelty N2 — Hybrid Embedding-Density Conformal OOD & Leave-Class-Out Semantic OOD:**
   - Combines 512-dim penultimate feature cosine distance to class centroids with RAPS nonconformity scores.
   - Boosts certified empty-set ($C(X) = \emptyset$) rejection on **Contamination OOD ($N=1,000$)** from **77.90% $\to$ 100.00%** (**0.9935 AUROC**, **0.9365 AUPR**) at **4.18% false rejection**, and achieves **99.25% rejection (0.9977 AUROC)** on **Leave-Class-Out Unseen 4-Mixed Defect Modes (C35–C38, $N=800$)**.
5. **Proposals A, C, D — Fab Triage, Mondrian Subgroup Parity & Metrology Cost-Loss Optimization:**
   - **Selective Fab Triage:** Automates **94.50% (7,185/7,603)** of wafers at **99.28% precision (7,133/7,185)**.
   - **Mondrian Conformal:** Guarantees group-conditional coverage across Depths 0–4 and **100.0% (30/30) coverage** on extreme minority class **C7 (Near-Full)**.
   - **Fab Economic Risk Control:** Cuts expected fab financial loss by **44.24% ($\$460,700$ saved across 7,603 test wafers)** at $C_{\text{escape}} = \$10,000$.

---

## 📁 Repository Structure

```
conformal_wafer/
├── checkpoints/                 # Trained PyTorch model weights (*.pt)
│   ├── dapnet_best.pt           # DAP-Net (IEEE Access 2025, 98.54% Acc)
│   ├── resnet18_best.pt         # ResNet-18 baseline (98.65% Acc)
│   ├── vit_tiny_best.pt         # ViT-Tiny baseline (91.38% Acc)
│   ├── dapnet_noise_*.pt        # Label noise checkpoints (5%, 10%, 20%)
│   └── dapnet_ls01.pt           # Label smoothing (0.1) ablation checkpoint
├── data/                        # Dataset storage (38,015 x 52 x 52 maps)
│   ├── Wafer_Map_Datasets.npz
│   └── Description.pdf
├── figures/                     # 13 High-resolution publication plots (300 DPI)
│   ├── coverage_vs_alpha.png
│   ├── coverage_by_depth.png
│   ├── setsize_by_depth.png
│   ├── coverage_nominal_diagnostic.png
│   ├── aci_online_drift_tracking.png
│   ├── hybrid_ood_roc_comparison.png
│   ├── calibration_size_sweep.png
│   ├── selective_classification_curve.png
│   ├── mondrian_vs_marginal_coverage.png
│   ├── fab_economic_risk_sensitivity.png
│   ├── noise_accuracy_vs_coverage.png
│   ├── set_size_examples.png
│   └── gradcam_examples.png
├── reports/                     # Complete presentation pack (all CSVs + PNGs)
├── results/                     # 15 Verified experimental CSV tables
│   ├── baseline_table.csv
│   ├── conformal_table.csv
│   ├── coverage_nominal_diagnostic.csv
│   ├── aci_online_drift.csv
│   ├── hybrid_ood_benchmark.csv
│   ├── triage_table.csv
│   ├── mondrian_table.csv
│   ├── calibration_size_sweep.csv
│   ├── kreg_ablation.csv
│   ├── ensemble_conformal.csv
│   ├── economic_risk_table.csv
│   ├── noise_table.csv
│   └── ablation.csv
├── src/                         # Modular source code
│   ├── config.py                # Global hyperparameters & paths
│   ├── dataset.py               # Categorical 3-channel encoding & leakage-free splits
│   ├── models.py                # DAP-Net, ResNet-18, ViT-Tiny architectures
│   ├── train.py                 # Mixed-precision training & evaluation
│   ├── conformal.py             # Split conformal engine (Exact & Non-Empty APS/RAPS)
│   ├── advanced_conformal.py    # N1 ACI, N2 Hybrid OOD, Triage, Mondrian, Cost Matrix
│   ├── advanced_plots.py        # Publication plotting suite for all extensions
│   ├── explainability.py        # Grad-CAM visualization
│   ├── plots.py                 # Core conformal plotting utilities
│   ├── run_pipeline.py          # Full training + core conformal driver
│   └── run_novelty_extensions.py# Post-hoc novelty ladder & diagnostic runner (~50s)
├── requirements.txt
├── RESULTS.md                   # Complete experimental report & tables
└── README.md
```

---

## ⚡ Quickstart & Execution

### 1. Run Post-Hoc Novelty Ladder & Diagnostic Suite (~50 seconds on GPU)
Using the pre-trained checkpoints in `conformal_wafer/checkpoints/`:
```bash
python -u -m conformal_wafer.src.run_novelty_extensions
```

### 2. Re-Train All Checkpoints from Scratch (~50 minutes on RTX 4050 GPU)
```bash
python -u -m conformal_wafer.src.run_pipeline
```

---

## 📚 Key References

1. **Base SOTA Reference:** J. Liu and H. Wang, *"Wafer composite defect recognition framework based on residual dynamic perception network with asymmetric multi-label loss (RDP-Net),"* **ISA Transactions**, vol. 167, pp. 1811–1827, 2025.
2. **Reimplemented Baseline:** *"Semiconductor Wafer Defect Recognition Based on Improved Coordinate Attention Mechanism (DAP-Net),"* **IEEE Access**, 2025.
3. **MixedWM38 Dataset:** J. Wang et al., *"Deformable Convolutional Networks for Efficient Mixed-Type Wafer Defect Pattern Recognition,"* **IEEE Transactions on Semiconductor Manufacturing**, vol. 33, no. 4, pp. 587–596, 2020.
4. **Split & Adaptive Conformal Prediction:**
   - Y. Romano, M. Sesia, and E. J. Candès, *"Classification with Valid and Adaptive Prediction Sets (APS),"* **NeurIPS**, 2020.
   - A. N. Angelopoulos, S. Bates, M. Jordan, and J. Malik, *"Uncertainty Sets for Image Classifiers using Conformal Prediction (RAPS),"* **ICLR**, 2021.
   - I. Gibbs and E. J. Candès, *"Adaptive Conformal Inference Under Distribution Shift,"* **NeurIPS**, 2021.
