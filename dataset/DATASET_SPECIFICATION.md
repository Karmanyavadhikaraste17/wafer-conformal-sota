# MixedWM38 Benchmark Dataset Specification

## 1. Overview & Dataset Provenance
- **Dataset Title:** MixedWM38 (Mixed-Type Semiconductor Wafer Map Defect Pattern Recognition Benchmark)
- **Primary Citation:** J. Wang et al., *"Deformable Convolutional Networks for Efficient Mixed-Type Wafer Defect Pattern Recognition"*, IEEE Transactions on Semiconductor Manufacturing, vol. 33, no. 4, pp. 587–596, 2020.
- **Total Physical Sample Count:** **38,015 real semiconductor wafer bin maps**.
- **Spatial Resolution:** **52 × 52 grid matrix** (representing 2,704 silicon die positions per wafer).
- **Physical Defect Modality:** Multi-label composite defect patterns resulting from photolithography focus excursions, chemical mechanical planarization (CMP) scratches, plasma etch RF irregularities, and physical contamination.

---

## 2. Die Grid Encoding
Each pixel in a wafer map matrix represents a physical die on the silicon disc:
- **`0` (Exterior / Unused Silicon):** Die positions lying outside the circular wafer disc perimeter.
- **`1` (Normal / Passing Die):** Die that successfully passed inline electrical parameter and optical testing.
- **`2` (Defective / Failing Die):** Die exhibiting circuit failure, leakage, or optical particle contamination.

---

## 3. Defect Taxonomy Across 5 Mixture Depths

The 38 defect patterns are rigorously categorized across 5 physical mixture depths based on the number of co-occurring defect modes:

| Depth Level | Defect Category | Number of Classes | Total Wafers | Description & Physical Genesis |
| :---: | :--- | :---: | :---: | :--- |
| **Depth 0** | Normal Silicon | 1 Class (C1) | 1,000 | Clean wafer without systematic spatial signatures. |
| **Depth 1** | Single Atomic Defects | 8 Classes (C2–C9) | 7,015 | Fundamental modes: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-Full, Scratch, Random. |
| **Depth 2** | Double Mixed Defects | 13 Classes (C10–C22) | 13,000 | Two simultaneous defect signatures (e.g., C+ER, EL+S, L+S). |
| **Depth 3** | Triple Mixed Defects | 12 Classes (C23–C34) | 12,000 | Three interacting defect mechanisms (e.g., C+EL+S, D+ER+L). |
| **Depth 4** | Quadruple Mixed Defects | 4 Classes (C35–C38) | 4,000 | Severe chamber excursions with four co-occurring signatures (C+L+EL+S, etc.). |
| **TOTAL** | **Full Benchmark** | **38 Classes** | **38,015 Wafers** | **Complete High-Volume Manufacturing Benchmark** |

---

## 4. Benchmark Partitioning & Canonical Split Protocol

To evaluate generalization and ensure reproducible benchmark integrity, MixedWM38 uses a stratified split protocol:
- **Training Set (60.0%):** 22,809 wafers used for supervised feature learning.
- **Calibration Set (20.0%):** 7,603 wafers used exclusively for Conformal Prediction risk calibration (nonconformity score quantiles).
- **Canonical Test Split (20.0%):** Exactly **7,603 held-out benchmark wafers** used for final SOTA accuracy, coverage verification, and SEMATECH economic loss auditing.

---

## 5. Visual Artifacts Included in this Folder
1. **`MixedWM38_38_Class_Visual_Gallery.png`**: High-resolution poster showing representative wafer photos across all 38 classes.
2. **`wafer_photos_by_depth/`**: Individual high-resolution wafer images categorized into subfolders by mixture depth.
3. **`class_distribution_manifest.csv`**: Full tabular manifest with exact sample counts and class indices.
4. **`MixedWM38_Sample_Subset_500.npz`**: Lightweight 500-wafer NumPy array ready for instant live Python loading and testing.
