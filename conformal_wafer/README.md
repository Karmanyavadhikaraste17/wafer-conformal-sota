# Grammar-Constrained Polar Networks with Conformal Novelty Detection for Unseen Mixed-Defect Wafer Maps

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📄 Abstract
Fabrication lines need wafer-map classifiers that know when they do not know, yet closed-set softmax recognizers return confident labels for defect compositions never seen in training. We study this on MixedWM38 by training on 34 known patterns (up to three co-occurring defects) and treating the four genuine quadruple mixtures as unseen. A flat 34-class softmax with split conformal calibration at $\alpha=0.05$ rejects only 65.13% (521/800) of them (AUROC 0.9168). We introduce **CompositionalPolarNet**, which combines equal-area polar resampling with azimuthally circular convolutions and a factorized eight-primitive attribute head constrained by a physical defect grammar. Its Library Novelty Score, the negative log probability mass placed on the known vocabulary, is conformally thresholded on a disjoint calibration set. We prove it grows linearly in the number of unseen primitives (exponentially in residual mass) whenever constituent logits exceed a margin. At $\alpha=0.05$ the method rejects **97.50% (780/800)** of unseen mixtures (AUROC 0.9941, AUPR 0.9792) while falsely rejecting **4.78% (325/6,803)** of known-pattern wafers, with **98.63%** known-class accuracy. Online adaptive conformal inference restores coverage under simulated chamber drift (89.97% vs 79.93% static), Mondrian calibration gives balanced coverage across mixture depths including the rare Near-Full class, and selective triage automates 94.50% of wafers at 99.28% precision. A cost-sensitivity analysis over 30 scenarios indicates yield-loss reductions of 44–51% under assumed costs. Encoding manufacturing grammar in the posterior, rather than only calibrating a flat softmax, is the key to trustworthy open-world defect recognition.

---

## 🔬 Key Research Highlights

1. **Tri-Architecture + Conformal Ensemble Benchmark:** Reimplements **DAP-Net** (*IEEE Access* 2025, **98.54%** test accuracy with only 875K parameters) alongside **ResNet-18** (**98.65%**), **ViT-Tiny** (**91.38%**), and a **Cross-Architecture Conformal Ensemble** (**98.71%**).
2. **Exact Randomized vs. Operational Non-Empty Conformal Sets (Checklist #1):**
   - **Exact Randomized RAPS (`allow_empty=True`):** Tracks nominal coverage $1-\alpha$ tightly across $\alpha \in [0.01, 0.20]$ (**99.17%, 95.15%, 90.41%** at nominal 99%, 95%, 90%).
   - **Operational Non-Empty RAPS (`allow_empty=False`):** Guarantees at least one candidate per wafer, achieving **94.50%–99.09% singleton automation** with **1.009–1.056 average set size**.
3. **Novelty N1 — Drift-Adaptive Online Conformal Monitoring (ACI, Gibbs & Candès 2021):**
   - Under mid-stream chamber process drift ($t \in [2501, 5500]$), Static Split Conformal coverage collapses to **79.93%**, while **Online ACI** self-corrects via pinball loss updates to **89.97%** during drift and **90.02%** across all 7,603 wafers without retraining.
4. **Novelty N2 & Architectural Breakthrough — True Leave-Class-Out Semantic OOD & Polar Neuro-Symbolic $S_{\text{lib}}(x)$:**
   - **Baseline 34-Class Model ($N_{\text{id}}=6,803, N_{\text{ood}}=800$):** Using a fixed softmax nonconformity score, our conformal detector achieves **65.13% empty-set rejection ($C(X) = \emptyset$, 521/800)** with **0.9168 AUROC** and **0.6607 AUPR** (base prevalence $\pi=0.1052$) at **4.32% realized test ID false rejection (294/6,803)**, matching MSP ranking while establishing finite-sample ID coverage control.
   - **Compositional Polar Neuro-Symbolic Network (`CompositionalPolarNet`):** Combines continuous polar resampling (`PolarConv`) with a formal defect grammar factorizing MixedWM38 into 8 primitive attributes. The resulting **Library Novelty Score $S_{\text{lib}}(x)$** drives unseen quad-mixture rejection to **97.50% (780/800)** with **0.9941 AUROC** and **0.9792 AUPR** at **4.78% ID false rejection** ($\le 5\%$), decisively conquering the open-world compound defect bottleneck!
   - **Contamination Diagnostic ($N=1,000$):** Hybrid feature-softmax conformal achieves **100.00% empty-set rejection** (**0.9935 AUROC**, **0.9365 AUPR**) at **4.18% in-distribution false rejection**.
5. **Proposals A, C, D — 4-Way Fab Triage, Mondrian Calibration & Metrology Cost-Loss Control:**
   - **Selective Fab Triage:** Automates **94.50% (7,185/7,603)** of wafers at **99.28% precision (7,133/7,185)**, deferring only **5.50% (418/7,603)** to human FA review.
   - **Mondrian Conformal:** Group-conditional calibration across Depths 0–4 with **100.0% (30/30) observed coverage** on extreme minority class **C7 (Near-Full, Wilson 95% CI $[88.65\%, 100.0\%]$)**.
   - **Fab Economic Risk Control:** Cuts expected fab financial loss by **44.24%–51.11% ($\$382,300–\$500,500$ saved across 7,603 test wafers)** across 30 distinct industrial cost-tiering and human FA miss probability scenarios ($C_{\text{within}} \in [\$200, \$10,000], p_{\text{miss}} \in [0.0, 1.0]$).

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
