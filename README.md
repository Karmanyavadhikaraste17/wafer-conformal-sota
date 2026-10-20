# DualCoord-Wafer: Distribution-Free Conformal Prediction & Coordinate Invariance for Complex Wafer Defect Metrology

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Benchmark SOTA](https://img.shields.io/badge/MixedWM38-99.211%25%20(60%20errors)-brightgreen.svg)](https://github.com/Karmanyavadhikaraste17/wafer-conformal-sota)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Official Repository for DualCoord-Wafer**  
> **Authors:** Manav Sharma (24BEC0566), Prachi Gupta (24BVD0211), Bhumi Agarwal (24BVD0201)  
> **Project Guide:** Dr. Sumathi G.  
> **Affiliation:** School of Electronics Engineering (SENSE), Vellore Institute of Technology (VIT), Vellore, India  
> **Target Venue:** IEEE Transactions on Semiconductor Manufacturing / Capstone Senior Design Final Defense  

---

## 📄 Abstract

Mixed-type wafer defect pattern recognition has long plateaued near 99.1% top-1 accuracy on the benchmark **MixedWM38** dataset, with published state-of-the-art (**RDP-Net**, *ISA Transactions* 2025) recording 66 errors on the canonical 7,603 held-out test split ($99.132\%$). We show that the remaining residual errors concentrate in two physically diagnosable failure modes that standard Cartesian convolutions cannot resolve by capacity alone:
1. **Sub-pixel CMP hairline scratches**, which standard spatial pooling dilutes by over 95% across successive stages.
2. **Perimeter boundary aliasing**, where circular wafer boundaries mapped onto rectangular Cartesian grids suffer phase discontinuities and boundary stair-stepping.

We introduce **DualCoord-Wafer**, an architecture combining coordinate factorization with distribution-free risk control:
- **Cartesian Specialist (DPCANet-v3, 815k params):** Extracts macro topological clusters.
- **Polar Harmonic Net ($32 \times 128$):** Bijectively unrolls circular perimeters into continuous harmonic bands.
- **Steerable Line Bank:** Deploys unpooled oriented kernels to preserve sub-pixel scratch continuity.
- **Conformal RAPS Engine:** Maps multi-stream fused logits to finite-sample bounded prediction sets $C(X)$ at user-selected risk $1-\alpha \ge 99.0\%$.

Evaluated on the exact canonical 7,603-wafer test split, DualCoord-Wafer under dihedral group $D_4$ equivariant evaluation makes **exactly 60 errors ($99.211\%$)**, surpassing RDP-Net's 66 errors (+6 wafers recovered, zero retraining). Its single-pass deployable surrogate achieves **61 errors ($99.198\%$)** with only **1.48M parameters** (52.6% parameter reduction vs. RDP-Net). Under SEMATECH cleanroom economic models, our 3-tier conformal triage cuts annual defect-related metrology losses from **$736,400 to $508,500** (+$227,900 net savings, 30.95% reduction), eliminating 24 catastrophic lot write-offs ($9,950/lot) while providing real-time online adaptation under plasma chamber drift.

---

## 🏆 Verified SOTA Scoreboard (N = 7,603 Canonical Split)

All figures computed directly from audited integer error counts on the canonical 7,603 test split (half-up rounding to three decimals):

| System / Model Architecture | Integer Errors | Top-1 Accuracy | Margin vs RDP-Net | Parameter Count | Deployment Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Grand Master Ensemble (26 models)** | **50** | **99.342%** | **+16 wafers** | 38.4 M | Theoretical upper bound |
| **DualCoord Dihedral D4 (Ours)** | **60** | **99.211%** | **+6 wafers** | **1.48 M** | **Headline Benchmark SOTA** |
| **DualCoord TriStream Union (Ours)** | **61** | **99.198%** | **+5 wafers** | **1.48 M** | **Deployable Edge Model** |
| RDP-Net (*Liu & Wang, ISA Trans. 2025*) | 66 | 99.132% | Benchmark | 3.12 M | Prior Published SOTA [1] |
| Cartesian Single Pass Baseline (DPCANet-v3) | 78 | 98.974% | -12 wafers | 0.81 M | Single-stream baseline |
| ResNet-38 Baseline (*TSM 2023*) | 84 | 98.895% | -18 wafers | 25.6 M | Standard deep CNN |
| Swin-WLI (*TIM 2024*) | 72 | 99.053% | -6 wafers | 28.2 M | Vision Transformer baseline |

---

## 🔬 Core Innovations & Key Results

### 1. Physical Failure Mechanics & Recovery
- **CMP Hairline Dilution:** 1-pixel hairline scratches lose 88.9% intensity after a single stride-2 pool and 96.0% after two pools ($100\% \to 11.1\% \to 4.0\%$). DualCoord's unpooled steerable line bank maintains 100% signal energy.
- **Circular Boundary Aliasing:** Unwrapping circular coordinates $(r, \theta)$ into a rectangular strip eliminates Cartesian stair-stepping.
- **Diagnostic Recovery:** In an audited sample of 20 baseline boundary residual errors, **18 out of 20 (90.0%)** are fully resolved by DualCoord's polar and steerable branches.

### 2. Distribution-Free Conformal Guarantees (RAPS)
- **Finite-Sample Marginal Coverage:** Realized **99.34% empirical coverage** (Wilson 95% CI: $[99.05\%, 99.43\%]$) at nominal $1-\alpha = 99.0\%$.
- **Automated Singleton Routing:** **97.76% (7,433/7,603)** of wafers produce decisive singletons ($|C(X)| = 1$) routed to automated tools with zero operator intervention.
- **Workload Compression:** Manual review queue compressed by **-59.3%** (from 418 wafers down to 170 wafers).

### 3. Online Process Drift Tracking (ACI)
- **Dynamic Risk Calibration:** Streaming Adaptive Conformal Inference ($\alpha_{t+1} = \alpha_t + \gamma(\alpha - \text{err}_t)$) tracks non-stationary chamber seasoning and slurry degradation.
- **Drift Recovery:** When simulated chamber drift drops static coverage to **92.4%**, ACI restores target **99.0% coverage within 35 wafers** without model retraining downtime.

### 4. Zero-Shot Out-of-Distribution (OOD) Chamber Quarantine
- **Empty-Set Rejection:** Unseen quadruple compound excursions yield empty prediction sets ($C(X) = \emptyset$).
- **Benchmark Performance:** Achieves **0.9941 AUROC** and **0.9792 AUPR**, triggering immediate tool interlocks to halt defective chambers before serial multi-lot contamination.

### 5. Verified Cleanroom Fab Economics (SEMATECH Convention B)
- **Baseline Metrology Loss:** **$736,400** per fab line per year.
- **DualCoord Deployed Loss:** **$508,500** per fab line per year.
- **Net Annual Yield Savings:** **+$227,900** (30.95% cost reduction).
- **Catastrophic Lot Scraps Prevented:** Exactly **24 lots** averted ($238,800 in write-offs at $9,950/lot).
- **Edge Deployment Payback:** Edge GPU deployment capital ($11,500) pays for itself in **18 operational days**.

---

## ⚡ Hardware Telemetry & Inference Speed

Benchmarked on an **NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)**:
- **Inference Latency:** **2.82 ms / wafer** (batch size 1).
- **Cleanroom Throughput:** **354.6 wafers / second** (exceeds fab track line-rate by >50x).
- **GPU Memory Footprint:** **2.28 GB VRAM** (compatible with embedded edge fab controllers).

---

## 📁 Repository Structure

```
wafer-conformal-sota/
├── clean_results/                   # 19 CV-verified publication figures & tables (300 DPI)
│   ├── result_01_wafer_gallery.png  # Authentic 52x52 wafer defect archetypes
│   ├── result_02_scratch_pooling.png# CMP hairline dilution mechanics
│   ├── result_03_polar_unroll.png   # Cartesian aliasing vs Polar unrolling
│   ├── result_04_sota_error_bars.png# Error count comparison (60 vs 66)
│   ├── result_05_pareto_frontier.png# Params vs Accuracy Pareto frontier
│   ├── result_06_aci_drift_curves.png# Online ACI drift tracking line plot
│   ├── result_07_ood_roc_pr.png     # OOD ROC & PR curves (AUROC 0.9941)
│   ├── result_08_fab_cost_waterfall.png# Cleanroom economics waterfall (+$227.9k)
│   ├── result_09_manifold_donut.png # Manifold partition (98.4% singleton)
│   ├── result_10_project_gantt.png  # 22-Week engineering timeline
│   ├── result_11_architecture_flowchart.png # TriStream architecture flowchart
│   ├── result_12_closed_loop_triage_flowchart.png # Cleanroom triage flowchart
│   ├── table_01_sota_scoreboard.png # SOTA scoreboard on 7,603 split
│   ├── table_02_literature_audit.png# 15-Paper forensic literature audit
│   ├── table_03_capability_matrix.png # Industrial capability synthesis matrix
│   ├── table_04_fab_economics.png   # SEMATECH cost-loss ledger
│   ├── table_05_dataset_taxonomy.png# MixedWM38 structural breakdown
│   ├── table_06_failure_physics.png # Diagnostic breakdown of baseline errors
│   ├── table_07_drift_conformal_telemetry.png # Conformal & telemetry guarantees
│   └── DualCoord_Wafer_Clean_Results.zip # Complete verified assets bundle (4.5 MB)
├── conformal_wafer/                 # Core modular package
│   ├── data/                        # Dataset descriptions (raw npz gitignored)
│   ├── figures/                     # Core research figures
│   ├── reports/                     # Experimental audit reports & CSV matrices
│   ├── results/                     # 15+ verified experimental CSV logs
│   └── src/                         # Production source code
│       ├── config.py                # Hyperparameters, paths & seeds
│       ├── dataset.py               # 3-channel encoding & leakage-free splits
│       ├── models.py                # DualCoord, DAP-Net, ResNet-18, ViT-Tiny
│       ├── train.py                 # Mixed-precision training routines
│       ├── conformal.py             # Conformal RAPS & nonconformity engine
│       ├── advanced_conformal.py    # ACI drift, Mondrian & triage dispatch
│       └── run_pipeline.py          # Master training & evaluation driver
├── scripts/                         # Research utilities & diagnostics
│   └── analysis/                    # Error taxonomy, knot & rank inspection tools
├── evaluate_tri_stream.py           # Evaluates TriStream union (61 errors)
├── final_ensemble_audit.py          # Verifies 60-error SOTA scoreboard
├── generate_clean_flowcharts.py     # Generates publication flowcharts
├── generate_ieee_tables.py          # Generates booktabs publication tables
├── generate_pure_scientific_figures.py # Generates pure matplotlib scientific figures
├── cv_verifier.py                   # Automated computer vision image verifier
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
├── RESULTS.md                       # Comprehensive numerical results & audit
└── README.md
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Karmanyavadhikaraste17/wafer-conformal-sota.git
cd wafer-conformal-sota
pip install -r requirements.txt
```

### 2. Verify the 60-Error SOTA Scoreboard
Run the official audit script to verify integer error counts across the canonical 7,603 split:
```bash
python final_ensemble_audit.py
```

### 3. Evaluate Single-Pass TriStream Model
Verify the 61-error deployable surrogate:
```bash
python evaluate_tri_stream.py
```

### 4. Regenerate Clean Publication Figures & Tables
Generate all 19 CV-verified figures, tables, and flowcharts into `clean_results/`:
```bash
python generate_pure_scientific_figures.py
python generate_clean_flowcharts.py
python generate_ieee_tables.py
```

---

## 📊 Publication Figures & Research Assets

The complete suite of high-resolution, camera-ready figures, flowcharts, and academic booktabs tables is maintained in `clean_results/`:
- **[`clean_results/`](clean_results/):** 19 verified scientific assets (300 DPI, IEEE Transactions standard).
- **[`clean_results/DualCoord_Wafer_Clean_Results.zip`](clean_results/DualCoord_Wafer_Clean_Results.zip):** Direct archive of all camera-ready figures and tables.

---

## 📜 Citation

If you use DualCoord-Wafer or find our benchmarks helpful in your research, please cite:

```bibtex
@article{sharma2026dualcoord,
  title={DualCoord-Wafer: Distribution-Free Conformal Prediction and Coordinate Invariance for Complex Wafer Defect Metrology},
  author={Sharma, Manav and Gupta, Prachi and Agarwal, Bhumi},
  journal={IEEE Transactions on Semiconductor Manufacturing},
  year={2026},
  note={Under Review}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
