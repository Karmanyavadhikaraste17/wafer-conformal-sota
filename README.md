# Conformal Wafer: Guaranteed-Coverage Prediction Sets for Mixed-Type Wafer Defect Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end ML research codebase providing the **first split conformal prediction sets with finite-sample coverage guarantees** on the **MixedWM38** benchmark (38 wafer defect classes, 38,015 maps).

---

## 🔬 Scientific Key Findings

1. **First Conformal Sets on MixedWM38:** While published models compete strictly on accuracy (saturated between 98.0% and 99.13%), this project provides the first calibrated prediction sets. At nominal 90% coverage ($\alpha = 0.10$), our RAPS method achieves **99.78% empirical coverage** with an average prediction set size of **1.92 classes**.
2. **Coverage Guarantees Hold Across Complexity:** Conformal coverage holds strictly across defect mix depths (100.0% on single defects, 99.81% on double mixtures, 99.65% on triple mixtures, and 99.62% on quadruple mixtures).
3. **Accuracy Hides Label Noise Breakdown; Conformal Exposes and Repairs It:**
   - Under 20% symmetric label noise, point test accuracy drops only 1.39% (from 98.54% down to 97.15%), deceivingly masking the corruption.
   - However, uncalibrated conformal set sizes explode from 1.92 to **21.11 classes** (over 55% of all classes).
   - Conformal recalibration on a small clean verification set restores prediction set size to **2.20 classes** while maintaining **99.75% coverage**.

---

## 📁 Repository Structure

```
conformal_wafer/
├── checkpoints/             # Trained PyTorch model weights (*.pt)
│   ├── dapnet_best.pt       # DAP-Net (IEEE Access 2025)
│   ├── resnet18_best.pt     # ResNet-18 baseline
│   ├── vit_tiny_best.pt     # ViT-Tiny baseline
│   ├── dapnet_noise_*.pt    # Noise study checkpoints (5%, 10%, 20%)
│   └── dapnet_ls01.pt       # Label smoothing ablation checkpoint
├── data/                    # Dataset storage
│   ├── Wafer_Map_Datasets.npz
│   └── Description.pdf      # MixedWM38 dataset specification
├── figures/                 # High-resolution publication plots (300 DPI)
│   ├── coverage_vs_alpha.png
│   ├── coverage_by_depth.png
│   ├── setsize_by_depth.png
│   ├── set_size_examples.png
│   ├── noise_accuracy_vs_coverage.png
│   └── gradcam_examples.png
├── reports/                 # Presentation-ready pack (all figures + tables)
├── results/                 # Verified experimental CSV outputs
│   ├── class_map.csv
│   ├── baseline_table.csv
│   ├── baseline_by_depth.csv
│   ├── conformal_table.csv
│   ├── conformal_by_depth.csv
│   ├── noise_table.csv
│   └── ablation.csv
├── src/                     # Modular source code
│   ├── config.py            # Global hyperparameters & paths
│   ├── dataset.py           # Categorical 3-channel encoding & leakage-free splits
│   ├── models.py            # DAP-Net, ResNet-18, ViT-Tiny architectures
│   ├── train.py             # Mixed-precision training & evaluation
│   ├── conformal.py         # Split conformal prediction engine (APS & RAPS)
│   ├── explainability.py   # Grad-CAM visualization
│   ├── plots.py             # Publication plotting utilities
│   └── run_pipeline.py      # End-to-end pipeline driver
├── requirements.txt         # Python dependencies
├── RESULTS.md               # Detailed experimental report
└── README.md
```

---

## ⚡ Quickstart & Replication

### 1. Environment Setup
```bash
# Clone and enter directory
cd conformal_wafer

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Acquisition
The dataset is downloaded automatically via `kagglehub`:
```python
import kagglehub
kagglehub.dataset_download("co1d7era/mixedtype-wafer-defect-datasets")
```
It carries the corrected C7/C9 labels verified by Dr. Uzma Batool.

### 3. Run Entire End-to-End Pipeline
```bash
python -u -m conformal_wafer.src.run_pipeline
```
*Expected Runtime on NVIDIA RTX 4050 Laptop GPU:*
- Full pipeline (3 baselines + conformal + noise study + ablation + Grad-CAM): **~50 minutes**.

---

## 📊 Summary Results Table

| Architecture | Parameters | Test Accuracy | Test Macro-F1 | 90% Conformal Coverage | 90% Avg Set Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DAP-Net (Ours)** | **875,814** | **98.54%** | **98.49%** | **99.78%** | **1.92** |
| **ResNet-18** | 11,188,326 | **98.65%** | **98.64%** | **99.80%** | **1.92** |
| **ViT-Tiny** | 5,388,326 | **91.38%** | **91.32%** | **99.09%** | **2.08** |

---

## 📚 Citations & Academic References

1. **Base SOTA Paper:**  
   J. Liu and H. Wang, *"Wafer composite defect recognition framework based on residual dynamic perception network with asymmetric multi-label loss (RDP-Net),"* **ISA Transactions**, vol. 167, pp. 1811–1827, 2025. DOI: [10.1016/j.isatra.2025.09.005](https://doi.org/10.1016/j.isatra.2025.09.005).

2. **Reimplemented Baseline:**  
   *"Semiconductor Wafer Defect Recognition Based on Improved Coordinate Attention Mechanism (DAP-Net),"* **IEEE Access**, 2025. DOI: [10.1109/ACCESS.2025.3546781](https://doi.org/10.1109/ACCESS.2025.3546781).

3. **Benchmark Dataset:**  
   J. Wang, C. Xu, Z. Yang, J. Zhang, and X. Li, *"Deformable Convolutional Networks for Efficient Mixed-Type Wafer Defect Pattern Recognition,"* **IEEE Transactions on Semiconductor Manufacturing**, vol. 33, no. 4, pp. 587–596, 2020. DOI: [10.1109/TSM.2020.3020985](https://doi.org/10.1109/TSM.2020.3020985).

4. **Conformal Prediction:**  
   - Y. Romano, M. Sesia, and E. J. Candès, *"Classification with Valid and Adaptive Prediction Sets (APS),"* **NeurIPS**, 2020.  
   - A. N. Angelopoulos, S. Bates, M. Jordan, and J. Malik, *"Uncertainty Sets for Image Classifiers using Conformal Prediction (RAPS),"* **ICLR**, 2021.
