import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.makedirs("clean_results", exist_ok=True)

# -------------------------------------------------------------
# 1. System Architecture Flowchart
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

# Define box styles
box_kw = dict(boxstyle='round,pad=0.6', facecolor='#F8FAFC', edgecolor='#1E293B', lw=1.5)
blue_box_kw = dict(boxstyle='round,pad=0.6', facecolor='#EFF6FF', edgecolor='#2563EB', lw=1.5)
green_box_kw = dict(boxstyle='round,pad=0.6', facecolor='#F0FDF4', edgecolor='#16A34A', lw=1.5)
amber_box_kw = dict(boxstyle='round,pad=0.6', facecolor='#FEFCE8', edgecolor='#D97706', lw=1.5)
red_box_kw = dict(boxstyle='round,pad=0.6', facecolor='#FEF2F2', edgecolor='#DC2626', lw=1.5)
arrow_kw = dict(arrowstyle='->', lw=1.8, color='#334155', shrinkA=6, shrinkB=6)

# Coordinates
# Input: x=0.08, y=0.5
ax.text(0.08, 0.5, "Input Wafer Map\n$X \\in \\{0,1\\}^{52 \\times 52}$\n(Binary Defect Mask)", 
        ha='center', va='center', fontsize=10, weight='bold', bbox=box_kw)

# Coordinate Decomposition (3 Branches)
# Branch 1: Cartesian Specialist
ax.text(0.35, 0.82, "Branch 1: Cartesian Backbone\n(DPCANet-v3, 815k params)\nCaptures macro topological clusters", 
        ha='center', va='center', fontsize=9.5, bbox=blue_box_kw)

# Branch 2: Polar Harmonic Specialist
ax.text(0.35, 0.50, "Branch 2: Polar Harmonic Net\n(Bilinear Warping $r,\\theta$, 32x128)\nResolves circular perimeter aliasing", 
        ha='center', va='center', fontsize=9.5, bbox=blue_box_kw)

# Branch 3: Steerable Filter Bank
ax.text(0.35, 0.18, "Branch 3: Steerable Line Bank\n(8 Oriented Kernels, Unpooled)\nPreserves 1-pixel hairline scratches", 
        ha='center', va='center', fontsize=9.5, bbox=blue_box_kw)

# Fusion & Conformal Engine
ax.text(0.65, 0.50, "Logit Fusion & Conformal RAPS\n$l_{\\mathrm{fused}} = \\sum w_i l_i$\n$C(X) = \\{y : E(X,y) \\leq \\hat{\\tau}_{\\mathrm{cal}}\\}$\nMarginal Coverage $\\geq 99.0\\%$", 
        ha='center', va='center', fontsize=10, weight='bold', bbox=amber_box_kw)

# 3 Triage Outputs
ax.text(0.92, 0.82, "Singleton Set: $|C(X)| = 1$\n(97.76% of Wafers)\n$\\to$ Automated Fab Routing", 
        ha='center', va='center', fontsize=9.5, weight='bold', color='#16A34A', bbox=green_box_kw)

ax.text(0.92, 0.50, "Review Set: $|C(X)| > 1$\n(2.24% Ambiguous)\n$\\to$ Human Expert Triage", 
        ha='center', va='center', fontsize=9.5, weight='bold', color='#D97706', bbox=amber_box_kw)

ax.text(0.92, 0.18, "Empty Set: $|C(X)| = 0$\n(OOD / Chamber Drift)\n$\\to$ Chamber Quarantine", 
        ha='center', va='center', fontsize=9.5, weight='bold', color='#DC2626', bbox=red_box_kw)

# Connecting Arrows
# Input to Branches
ax.annotate('', xy=(0.20, 0.82), xytext=(0.14, 0.56), arrowprops=arrow_kw)
ax.annotate('', xy=(0.20, 0.50), xytext=(0.15, 0.50), arrowprops=arrow_kw)
ax.annotate('', xy=(0.20, 0.18), xytext=(0.14, 0.44), arrowprops=arrow_kw)

# Branches to Fusion
ax.annotate('', xy=(0.53, 0.56), xytext=(0.49, 0.80), arrowprops=arrow_kw)
ax.annotate('', xy=(0.53, 0.50), xytext=(0.49, 0.50), arrowprops=arrow_kw)
ax.annotate('', xy=(0.53, 0.44), xytext=(0.49, 0.20), arrowprops=arrow_kw)

# Fusion to Triage
ax.annotate('', xy=(0.80, 0.80), xytext=(0.77, 0.56), arrowprops=arrow_kw)
ax.annotate('', xy=(0.80, 0.50), xytext=(0.77, 0.50), arrowprops=arrow_kw)
ax.annotate('', xy=(0.80, 0.20), xytext=(0.77, 0.44), arrowprops=arrow_kw)

ax.set_xlim(0, 1.02)
ax.set_ylim(0.05, 0.95)
ax.axis('off')

plt.title("DualCoord-Wafer TriStream Architecture & Conformal Prediction Flowchart", 
          fontsize=13, weight='bold', pad=15, color='#0F172A')
plt.tight_layout()
plt.savefig("clean_results/result_11_architecture_flowchart.png", dpi=300, facecolor='#FFFFFF')
plt.close()
print("result_11 generated.")

# -------------------------------------------------------------
# 2. Closed-Loop Fab Triage & Online Adaptation Flowchart
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

# Blocks
ax.text(0.08, 0.65, "In-Line Metrology\nOptical Scanning\n(52x52 Wafer Map)", 
        ha='center', va='center', fontsize=9.5, weight='bold', bbox=box_kw)

ax.text(0.25, 0.65, "DualCoord Model\nInference (2.82 ms)\nRAPS Score Computation", 
        ha='center', va='center', fontsize=9.5, weight='bold', bbox=blue_box_kw)

# Decision Diamond / Gate
ax.text(0.46, 0.65, "Conformal Set Size\n$|C(X)|$ Evaluation\n(Nominal $1-\\alpha = 99.0\\%$)", 
        ha='center', va='center', fontsize=9.5, weight='bold', bbox=amber_box_kw)

# 3 Branches
ax.text(0.69, 0.88, "|C(X)| = 1 (Singleton)\nDecisive Classification\n(Top-1 Confident)", 
        ha='center', va='center', fontsize=9, bbox=green_box_kw)
ax.text(0.92, 0.88, "Automated Lot Dispatch\n$\\to$ Etch / Litho / CMP Tools\n(Zero Operator Overhead)", 
        ha='center', va='center', fontsize=9, weight='bold', color='#16A34A', bbox=green_box_kw)

ax.text(0.69, 0.65, "|C(X)| > 1 (Ambiguous)\nMulti-Class Uncertainty\n(2 to 4 Candidate Classes)", 
        ha='center', va='center', fontsize=9, bbox=amber_box_kw)
ax.text(0.92, 0.65, "Defect Metrology Review\nOperator inspects candidates\n(-59.3% Review Queue Burden)", 
        ha='center', va='center', fontsize=9, weight='bold', color='#D97706', bbox=amber_box_kw)

ax.text(0.69, 0.42, "|C(X)| = 0 (Empty Set)\nNonconformity > Threshold\nOOD Novel Defect Detected", 
        ha='center', va='center', fontsize=9, bbox=red_box_kw)
ax.text(0.92, 0.42, "Chamber Quarantine\nImmediate Tool Halt\n(Prevents $9,950 Lot Scraps)", 
        ha='center', va='center', fontsize=9, weight='bold', color='#DC2626', bbox=red_box_kw)

# Bottom Feedback Loop: ACI Online Drift Tracking
ax.text(0.50, 0.12, "Online Feedback Loop: Adaptive Conformal Inference (ACI)\n$\\alpha_{t+1} = \\alpha_t + \\gamma (\\alpha - \\mathrm{err}_t)$ dynamically tracks chamber drift in real time.\nNovel OOD wafers buffered for automated fine-tuning retraining pipeline.", 
        ha='center', va='center', fontsize=9.5, style='italic', color='#1E293B', 
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#F1F5F9', edgecolor='#64748B', lw=1.2))

# Arrows
ax.annotate('', xy=(0.17, 0.65), xytext=(0.14, 0.65), arrowprops=arrow_kw)
ax.annotate('', xy=(0.37, 0.65), xytext=(0.33, 0.65), arrowprops=arrow_kw)

# Decision to outcomes
ax.annotate('', xy=(0.59, 0.88), xytext=(0.54, 0.70), arrowprops=arrow_kw)
ax.annotate('', xy=(0.59, 0.65), xytext=(0.54, 0.65), arrowprops=arrow_kw)
ax.annotate('', xy=(0.59, 0.42), xytext=(0.54, 0.60), arrowprops=arrow_kw)

# Outcome to actions
ax.annotate('', xy=(0.82, 0.88), xytext=(0.79, 0.88), arrowprops=arrow_kw)
ax.annotate('', xy=(0.82, 0.65), xytext=(0.79, 0.65), arrowprops=arrow_kw)
ax.annotate('', xy=(0.82, 0.42), xytext=(0.79, 0.42), arrowprops=arrow_kw)

# Feedback loop arrows
ax.annotate('', xy=(0.92, 0.32), xytext=(0.92, 0.35), arrowprops=dict(arrowstyle='-', lw=1.2, color='#64748B'))
ax.annotate('', xy=(0.78, 0.18), xytext=(0.92, 0.32), arrowprops=dict(arrowstyle='->', lw=1.2, color='#64748B'))
ax.annotate('', xy=(0.25, 0.55), xytext=(0.25, 0.20), arrowprops=dict(arrowstyle='->', lw=1.2, color='#64748B'))

ax.set_xlim(0, 1.02)
ax.set_ylim(0.02, 0.98)
ax.axis('off')

plt.title("Closed-Loop Cleanroom Metrology & Online Adaptive Triage Flowchart", 
          fontsize=13, weight='bold', pad=15, color='#0F172A')
plt.tight_layout()
plt.savefig("clean_results/result_12_closed_loop_triage_flowchart.png", dpi=300, facecolor='#FFFFFF')
plt.close()
print("result_12 generated.")
