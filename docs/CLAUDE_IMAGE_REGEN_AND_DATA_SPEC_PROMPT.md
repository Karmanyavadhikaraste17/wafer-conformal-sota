# COMPLEMENTARY DIRECTIVE FOR CLAUDE: ASSET REVIEW, DATA SPECIFICATIONS & AUTONOMOUS RE-GENERATION ENGINE

> **Instructions for User:**  
> Copy and paste this prompt to Claude along with `DualCoord_Wafer_Clean_Results.zip`.  
> This empowers Claude to either use the attached verified assets OR write Python/Matplotlib code to generate fresh, custom-styled multi-image layouts if it feels any visual can be enhanced.

***

```markdown
# MISSION UPGRADE: VISUAL AUDIT, MULTI-IMAGE SLIDE COMPOSITION & AUTONOMOUS RE-GENERATION ENGINE
**Project Title:** DualCoord-Wafer: Distribution-Free Conformal Prediction & Coordinate Invariance for Complex Wafer Defect Metrology  
**Target:** IEEE Transactions on Semiconductor Manufacturing / Capstone Senior Design Defense  
**Authors:** Manav Sharma, Prachi Gupta, Bhumi Agarwal (Department of SENSE, VIT Vellore)  

---

### YOUR DUAL CAPABILITY MANDATE (REVIEW OR REGENERATE):
1. **ASSET REVIEW & AUDIT:**
   - I have provided `DualCoord_Wafer_Clean_Results.zip` containing 19 pre-rendered, verified scientific figures (`result_01` to `result_12`) and academic booktabs tables (`table_01` to `table_07`).
   - **Examine all images.** If you find that the provided images fit the layout perfectly, embed them directly.
   - **Autonomous Re-Generation Freedom:** If you feel any visual is not up to the highest aesthetic mark, or if you want custom styling, seamless integration, or higher visual impact, **YOU HAVE FULL PERMISSION AND COMPLETE DATA TO REGENERATE NEW IMAGES FROM SCRATCH** (via Python code execution, Matplotlib/Seaborn, SVG rendering, or Canva connectors).

2. **MULTI-IMAGE SLIDE COMPOSITION (MANDATORY):**
   - Do NOT limit slides to a single isolated chart. Where appropriate, **compose 2 to 3 complementary visuals per slide** (e.g., Table + Accompanying Bar Chart; Wafer Map + Zoom Inset + Dilution Plot; ROC Curve + PR Curve; Architecture Flowchart + Output Dispatch Tree).
   - This creates an authentic, high-impact defense slide deck that speaks for itself visually.

3. **MINIMAL SLIDE TEXT (STRICT RULE):**
   - Maximum **3 to 4 crisp, high-impact bullet points** per slide (1-2 lines each, bolding key numbers).
   - Professors and committee members will NOT read dense text.
   - All deep technical explanations, statistical derivations, and mathematical proofs go into the **SPEAKER NOTES** section of each slide.

4. **DESIGN LANGUAGE (NO AI CLICHÉS):**
   - STRICTLY AVOID dark blue / purple gradient "AI generated" templates.
   - Use a **Cleanroom Minimalist Editorial Aesthetic**:
     - Background: Pure White (`#FFFFFF`) or Crisp Off-White (`#F8FAFC`).
     - Typography: Clean sans-serif (Inter, SF Pro, Helvetica Neue, or Roboto). Primary text in Deep Slate (`#0F172A`).
     - Accents: Silicon Blue (`#2563EB`), Yield Emerald (`#16A34A`), Attention Amber (`#D97706`), Scrap Red (`#DC2626`).
     - Booktabs table styling: Thick top and bottom rules (`#0F172A`), thin header divider, no vertical grid lines.

---

# COMPREHENSIVE DATA SPECIFICATIONS FOR ALL 16 SLIDES
(Use these exact mathematical datasets, integer counts, formulas, and visual specs if regenerating any image)

---

### SLIDE 01: Title & Executive Hook
- **Slide Focus:** Executive introduction, 99.13% plateau context, headline SOTA surpass.
- **Images on Slide (2 Visuals):**
  1. `Visual 1A [Wafer Gallery Grid]`: 2x4 grid of authentic $52 \times 52$ wafer maps showing 8 defect classes:
     - Classes: (a) Center, (b) Donut, (c) Edge-Loc, (d) Edge-Ring, (e) Loc, (f) Near-Full, (g) Scratch, (h) Random.
     - Encoding: Circular wafer boundary at $R=26$, defect dies in red/dark slate, normal dies in light gray, outer background pure white.
  2. `Visual 1B [KPI Metric Badges]`: 4 clean executive stat cards:
     - **Top-1 Accuracy:** 99.211% (New Benchmark SOTA)
     - **Integer Errors:** 60 / 7,603 wafers (-6 errors vs RDP-Net)
     - **Parameter Count:** 1.48M (52.6% parameter reduction vs SOTA)
     - **Cleanroom ROI:** +$227,900 net annual savings (30.95% loss reduction)
- **Slide Text (3 Bullets):**
  * **The Benchmark Plateau:** MixedWM38 held-out accuracy stalled at 66 errors (RDP-Net, ISA Trans. 2025).
  * **Verified SOTA Surpass:** Achieved **60 errors (99.211%)** on the fixed 7,603 canonical split (+6 wafers over SOTA).
  * **Dual Coordinate Invariance:** Cartesian clustering + Polar harmonic unrolling + Conformal triage.
- **Speaker Notes:** Full context introducing the project, authors, venue, and the core claim verified on integer error counts.

---

### SLIDE 02: The Cleanroom Crisis & Industrial Economics
- **Slide Focus:** Real-world cost of defect escapes in modern 300mm fabs.
- **Images on Slide (2 Visuals):**
  1. `Visual 2A [SEMATECH Cost Waterfall Bar]`:
     - Baseline Metrology Loss: **$736,400**
     - Avoided Catastrophic Lot Scraps (24 lots × $9,950): **-$238,800**
     - Avoided Downstream Die Yield Scraps: **-$19,200**
     - Metrology Review Labor Addition (170 wafers × $177): **+$30,100**
     - Final Deployed Loss: **$508,500**
     - Net Annual Yield Recovery: **+$227,900** (30.95% net reduction)
  2. `Visual 2B [Catastrophic Lot Multiplier Comparison]`:
     - Bar 1: Single Wafer Scrap Cost = **$35 / wafer**
     - Bar 2: Catastrophic 25-Wafer Lot Write-Off = **$9,950 / lot** (284x cost multiplier)
     - Callout: "24 Catastrophic Escapes Prevented Annually"
- **Slide Text (3 Bullets):**
  * **Severe Defect Multiplier:** An undetected cluster ruins an entire 25-wafer lot ($9,950 per occurrence).
  * **Operator Saturation:** Manual inspection overwhelmed with 418 daily ambiguous edge cases.
  * **Proven Financial ROI:** Net annual cleanroom loss slashed by **30.95% (+$227,900 net savings)**.
- **Speaker Notes:** Explain SEMATECH Convention B economic ledger, catastrophic lot penalties, and operator cost.

---

### SLIDE 03: Forensic Literature Audit (15 Papers)
- **Slide Focus:** Systematic analysis of prior art proving the 99.1% ceiling.
- **Images on Slide (2 Visuals):**
  1. `Visual 3A [Academic Audit Table (Booktabs)]`:
     - Columns: Model | Year | Venue | Params | Test Errors (N=7,603) | Accuracy | Coordinate Grid | Conformal Bounds
     - Row 1: VGG-16 | 2019 | TII | 138M | 148 err | 98.05% | Cartesian | None
     - Row 2: ResNet-38 | 2023 | TSM | 25.6M | 84 err | 98.89% | Cartesian | None
     - Row 3: Swin-WLI | 2024 | TIM | 28.2M | 72 err | 99.05% | Cartesian | None
     - Row 4: RDP-Net (SOTA) | 2025 | ISA Trans | 3.12M | 66 err | 99.13% | Cartesian | None
     - Row 5: **DualCoord D4 (Ours)** | **2026** | **Target: TSM** | **1.48M** | **60 err** | **99.21%** | **Dual (Cart+Polar)** | **Guaranteed (99%)**
  2. `Visual 3B [Coordinate Limitation Callout]`:
     - Stat Donut / Callout: **100% of Prior Literature** strictly uses Cartesian grids; **0%** incorporates Conformal/Drift safety.
- **Slide Text (3 Bullets):**
  * **Ceiling Stagnation:** Published models cluster at 98.9% - 99.13% despite 10x parameter expansion.
  * **Structural Blindspot:** 100% of audited models rely purely on Cartesian grids, ignoring circular wafer geometry.
  * **Zero Operational Guarantees:** No prior paper provides finite-sample risk bounds or real-time drift adaptation.
- **Speaker Notes:** Walk through the 15-paper audit matrix, emphasizing why capacity scaling stopped yielding progress.

---

### SLIDE 04: Industrial Cleanroom Capability Matrix
- **Slide Focus:** Comparing models across 8 industrial fab operational criteria.
- **Images on Slide (1 Large Visual or 2 Complementary):**
  1. `Visual 4A [Multi-Criteria Synthesis Matrix]`:
     - Matrix comparing: Criteria | ResNet-38 | Swin-WLI | RDP-Net | **DualCoord-Wafer**
     - 1. Top-1 SOTA (>99.13%): No | No | Benchmark (66) | **Yes (60 errors)**
     - 2. Parameter Efficiency (<2M): No (25.6M) | No (28.2M) | No (3.12M) | **Yes (1.48M)**
     - 3. Sub-3ms Edge Latency: No (14ms) | No (22ms) | Partial (4.1ms) | **Yes (2.82ms)**
     - 4. Rotational / Dihedral Invariance: No | No | No | **Yes (D4 Group Equivariance)**
     - 5. Unpooled Scratch Preservation: No | No | No | **Yes (Steerable Filters)**
     - 6. Distribution-Free Conformal Guarantees: No | No | No | **Yes (RAPS Coverage $\ge 99\%$)**
     - 7. Online Process Drift Tracking: No | No | No | **Yes (Streaming ACI)**
     - 8. Zero-Shot OOD Chamber Quarantine: No | No | No | **Yes (Empty Set Rejection)**
- **Slide Text (3 Bullets):**
  * **The Industrial Gap:** Current deep learning excels in accuracy but fails cleanroom safety and drift criteria.
  * **Only DualCoord Satisfies All 8:** Unifies sub-pixel invariance, statistical guarantees, and sub-3ms edge latency.
  * **Holistic Synthesis:** From raw wafer scanning to automated lot dispatch in 2.82 milliseconds.
- **Speaker Notes:** Explain each industrial criterion and why conventional models cannot be safely integrated into autonomous fabs.

---

### SLIDE 05: MixedWM38 Benchmark Taxonomy & Rigor
- **Slide Focus:** Dataset structure, multi-defect complexity, and canonical evaluation split.
- **Images on Slide (2 Visuals):**
  1. `Visual 5A [Dataset Taxonomy & Split Breakdown Table]`:
     - Total Dataset Size: **38,015 wafers** (52×52 resolution)
     - Training Partition: **26,610 wafers (70.0%)**
     - Calibration Partition: **3,802 wafers (10.0%)** (Strictly held-out for Conformal calibration)
     - Canonical Test Split: **7,603 wafers (20.0%)** (Identical canonical split to RDP-Net 2025)
  2. `Visual 5B [Defect Cardinality Breakdown Bar Chart]`:
     - Normal: 1 class (1,000 wafers)
     - Single-defect: 8 classes (8,000 wafers)
     - Double compound mixes: 13 classes (13,000 wafers)
     - Triple compound mixes: 13 classes (13,000 wafers)
     - Quadruple compound mixes: 4 classes (3,015 wafers, used as OOD challenge)
- **Slide Text (3 Bullets):**
  * **Multi-Defect Complexity:** 38 classes capturing overlapping scratch, ring, cluster, and normal signatures.
  * **Canonical 7,603 Test Split:** Evaluated on the exact frozen partition of RDP-Net to eliminate split bias.
  * **Dedicated Conformal Calibration:** 3,802 held-out wafers reserved solely for distribution-free quantile calibration.
- **Speaker Notes:** Detail the dataset composition, proving zero data leakage between train, calibration, and test splits.

---

### SLIDE 06: Physical Failure Mechanics (The 2 Unresolved Modes)
- **Slide Focus:** The physical root causes why Cartesian architectures hit the 99.1% wall.
- **Images on Slide (2 Visuals Side-by-Side):**
  1. `Visual 6A [1-Pixel CMP Hairline Scratch Dilution Plot]`:
     - Panel 1: Real wafer map showing single-pixel CMP scratch.
     - Panel 2: Microscopic pixel zoom of the 1-pixel wide line defect.
     - Panel 3: Spatial energy bar chart through standard $2\times2$ stride-2 pooling:
       * Native resolution ($52\times52$): **100% Signal Intensity**
       * First Stride-2 Pooling ($26\times26$): **11.1% Relative Intensity** (-88.9% dilution)
       * Second Stride-2 Pooling ($13\times13$): **4.0% Relative Intensity** (-96.0% dilution $\to$ defect disappears)
  2. `Visual 6B [Perimeter Boundary Aliasing vs Polar Unrolling]`:
     - Top: Cartesian coordinate representation showing circular boundary chopped by square pixels (stair-stepping artifacts).
     - Bottom: Polar coordinate transformation ($r, \theta$) unwrapping circular boundary into a continuous horizontal ribbon ($32 \times 128$).
- **Slide Text (3 Bullets):**
  * **Failure Mode 1 (CMP Scratches):** 1-pixel hairline defects lose 96% of spatial energy across two pooling layers.
  * **Failure Mode 2 (Boundary Aliasing):** Circular wafer perimeters suffer Cartesian discretization phase errors.
  * **The Physical Invariant:** Unpooled steerable filters preserve scratches; polar transformation preserves boundaries.
- **Speaker Notes:** Explain the physical mechanism of CMP mechanical polishing and why Cartesian pooling fundamentally destroys thin defects.

---

### SLIDE 07: Forensic Diagnostic Analysis of Baseline Errors
- **Slide Focus:** Empirical proof that baseline residual errors concentrate in these two failure modes.
- **Images on Slide (2 Visuals):**
  1. `Visual 7A [Baseline Failure Physics Audit Table]`:
     - Table classifying 20 inspected baseline residual errors:
       * Mode A (Edge-Ring Aliasing): 11 errors $\to$ 10 recovered via Polar Unrolling (90.9% recovery)
       * Mode B (CMP Hairline Pooling): 7 errors $\to$ 6 recovered via Steerable Filter Bank (85.7% recovery)
       * Mode C (Extreme Noise / Normal Die Scatter): 2 errors $\to$ Unresolved (Irreducible noise)
       * Total Recoverable: **18 out of 20 surveyed errors (90.0% recovery rate)**.
  2. `Visual 7B [Error Concentration Donut / Bar]`:
     - 85% of all baseline errors concentrate in compound patterns containing either Edge-Ring or Scratch.
- **Slide Text (3 Bullets):**
  * **Error Concentration:** 85% of baseline misclassifications stem directly from boundary and hairline defects.
  * **18 of 20 Errors Recovered:** DualCoord's coordinate-adapted branches successfully resolve 90% of investigated errors.
  * **Targeted Innovation:** Rather than blindly adding network depth, we solve the specific physical failure modes.
- **Speaker Notes:** Detail the empirical failure audit of the baseline model, showing how polar and steerable kernels resolve errors.

---

### SLIDE 08: Proposed DualCoord-Wafer TriStream Architecture
- **Slide Focus:** Complete architectural design and coordinate factorization pipeline.
- **Images on Slide (1 Comprehensive Flowchart Visual):**
  1. `Visual 8A [TriStream System Architecture Flowchart]`:
     - **Input:** Wafer Map $X \in \{0,1\}^{52 \times 52}$
     - **Branch 1 [Cartesian Backbone]:** DPCANet-v3 (815k params) $\to$ Extracts macro topological clusters $\to$ Logit vector $l_{\text{cart}} \in \mathbb{R}^{38}$
     - **Branch 2 [Polar Harmonic Net]:** Bilinear Warping $(r, \theta) \to 32 \times 128$ cylindrical strip $\to$ Resolves perimeter boundary harmonics $\to$ Logit vector $l_{\text{polar}} \in \mathbb{R}^{38}$
     - **Branch 3 [Steerable Line Bank]:** 8 unpooled oriented kernels $(0^\circ, 22.5^\circ, \dots, 157.5^\circ)$ $\to$ Resonates with 1-pixel scratches $\to$ Logit vector $l_{\text{scratch}} \in \mathbb{R}^{38}$
     - **Fusion & Conformal Engine:** Logit fusion $l_{\text{fused}} = \sum w_i l_i \to$ Conformal RAPS Calibration $\to$ Bounded Set $C(X)$.
     - **3-Way Dispatch:** $|C(X)|=1$ (Automated Fab Routing) | $|C(X)|>1$ (Human Review) | $|C(X)|=\emptyset$ (Chamber Quarantine).
- **Slide Text (3 Bullets):**
  * **TriStream Factorization:** Decouples macro clusters, circular boundaries, and 1-pixel scratches into specialized streams.
  * **Zero Information Loss:** Bypasses spatial pooling for thin defects and eliminates Cartesian boundary aliasing.
  * **Integrated Conformal Engine:** Outputs calibrated prediction sets with finite-sample statistical guarantees.
- **Speaker Notes:** Guide the audience through the three parallel streams, explaining the mathematical role of each coordinate branch.

---

### SLIDE 09: Benchmark Scoreboard: Surpassing 99.13%
- **Slide Focus:** Official integer-count verification proving the new benchmark state-of-the-art.
- **Images on Slide (2 Visuals Side-by-Side):**
  1. `Visual 9A [Academic Scoreboard Table (Booktabs)]`:
     - System | Integer Errors (N=7,603) | Top-1 Accuracy | Margin vs SOTA | Deployment Status
     - Grand Master Ensemble (26 models) | 50 errors | 99.342% | +16 wafers | Upper bound reference
     - **DualCoord Dihedral D4 (Ours)** | **60 errors** | **99.211%** | **+6 wafers** | **Headline SOTA Surpass**
     - **TriStream Single Pass (Ours)** | **61 errors** | **99.198%** | **+5 wafers** | **Deployable Edge Model**
     - RDP-Net (Liu & Wang 2025) | 66 errors | 99.132% | Benchmark | Prior Published SOTA
     - Cartesian Single Pass Baseline | 78 errors | 98.974% | -12 wafers | Efficiency baseline
     - ResNet-38 Baseline | 84 errors | 98.895% | -18 wafers | Standard deep CNN
  2. `Visual 9B [Integer Error Count Bar Chart]`:
     - Horizontal or vertical bar chart clearly plotting error counts: 60 vs 61 vs 66 vs 78 vs 84 vs 104 vs 111.
     - Distinct dashed benchmark reference line at $y=66$ (RDP-Net SOTA).
     - Green highlight on 60 errors (-6 wafers).
- **Slide Text (3 Bullets):**
  * **The New SOTA Champion:** DualCoord D4 achieves **60 errors (99.211%)**, beating RDP-Net's 66 errors (99.132%).
  * **Deployable Single Pass:** TriStream Union makes **61 errors (99.198%)** without any test-time augmentation.
  * **Audited Integer Precision:** Verified on the identical 7,603 split without selective rounding or subset filtering.
- **Speaker Notes:** Highlight that the claim is verified in exact integer wafer counts on the canonical split, cementing credibility.

---

### SLIDE 10: Parameter Efficiency & Pareto Frontier
- **Slide Focus:** Model complexity, edge inference latency, and hardware benchmarking.
- **Images on Slide (2 Visuals):**
  1. `Visual 10A [Parameter vs Error Pareto Frontier Scatter Plot]`:
     - X-axis: Number of Parameters (Log scale, 500k to 150M)
     - Y-axis: Canonical Test Errors (50 to 150 errors)
     - Points: VGG-16 (138M, 148 err), ResNet-38 (25.6M, 84 err), Swin-WLI (28.2M, 72 err), RDP-Net (3.12M, 66 err), DualCoord TriStream (1.48M, 61 err), DualCoord D4 (1.48M, 60 err).
     - Highlight the Pareto Frontier curve showing DualCoord's complete dominance.
  2. `Visual 10B [Cleanroom Edge Hardware Telemetry Table]`:
     - Device: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
     - Edge Inference Latency: **2.82 ms / wafer**
     - Cleanroom Throughput: **354.6 wafers / second** (Exceeds line rate by 50x)
     - Memory Footprint: **2.28 GB VRAM** (Easily deploys on embedded edge controllers)
- **Slide Text (3 Bullets):**
  * **52.6% Fewer Parameters:** 1.48M parameters vs RDP-Net's 3.12M and ResNet's 25.6M.
  * **Sub-3ms Edge Inference:** 2.82 ms per wafer enables real-time 354 wafer/sec inline inspection.
  * **Pareto Dominance:** Highest accuracy at the lowest compute footprint across all evaluated models.
- **Speaker Notes:** Explain cleanroom edge hardware constraints and demonstrate that DualCoord is optimized for real-time edge fab deployment.

---

### SLIDE 11: Statistical Calibration: Conformal RAPS & Mondrian Guarantees
- **Slide Focus:** Distribution-free prediction sets, finite-sample guarantees, and operator workload compression.
- **Images on Slide (2 Visuals Side-by-Side):**
  1. `Visual 11A [RAPS Empirical Coverage Calibration Curve]`:
     - X-axis: Nominal Coverage Level $(1 - \alpha)$, ranging from 0.90 to 0.999.
     - Y-axis: Empirical Realized Coverage on 7,603 held-out test wafers.
     - Target line $y = x$.
     - DualCoord RAPS Curve: Tracks line perfectly, achieving **99.34% realized coverage** at nominal 99.0% target (Wilson 95% CI: [99.05%, 99.43%]).
  2. `Visual 11B [Triage Dispatch Manifold Donut Chart]`:
     - **Automated Singleton Set ($|C(X)| = 1$):** 97.76% of wafers (7,433 wafers) $\to$ Automated tool routing.
     - **Ambiguous Review Set ($|C(X)| > 1$):** 2.24% of wafers (170 wafers) $\to$ Operator metrology review.
     - **Human Review Compression:** **-59.3% reduction** in manual review queue (down from 418 wafers in DAP-Net baseline).
- **Slide Text (3 Bullets):**
  * **Guaranteed Coverage:** Realized **99.34% coverage** with mathematically provable finite-sample bounds.
  * **Automated Dispatch:** 97.76% of wafers are decisive singletons routed directly without human intervention.
  * **59.3% Labor Reduction:** Ambiguous review volume compressed from 418 wafers down to only 170 wafers.
- **Speaker Notes:** Explain how Conformal Prediction transforms dangerous softmax probabilities into reliable set-valued decisions.

---

### SLIDE 12: Online Process Drift Adaptation (ACI)
- **Slide Focus:** Dynamic risk calibration during cleanroom plasma chamber drift and slurry aging.
- **Images on Slide (2 Visuals Stacked or Side-by-Side):**
  1. `Visual 12A [Streaming Coverage Drift & Recovery Curve]`:
     - X-axis: Streaming Wafer Steps ($t = 0$ to $500$).
     - At $t=150$, introduce severe simulated plasma etch drift (feature distribution shifts).
     - Static Model (No Adaptation): Coverage plunges to **92.4%** and stays broken.
     - DualCoord ACI ($\gamma = 0.005$): Drops momentarily, then dynamically recovers to **99.0% within 35 wafers**.
  2. `Visual 12B [Dynamic Significance Parameter $\alpha_t$ Adaptation]`:
     - Plots $\alpha_t$ dynamically adjusting via $\alpha_{t+1} = \alpha_t + \gamma(\alpha - \text{err}_t)$ to compensate for drift without retraining.
- **Slide Text (3 Bullets):**
  * **The Drift Hazard:** Tool aging causes static classifiers to fail silently (coverage drops to 92.4%).
  * **Online ACI Engine:** Real-time quantile modulation restores 99% coverage within 35 wafers.
  * **Zero Retraining Downtime:** Preserves cleanroom safety online without halting production lines.
- **Speaker Notes:** Contrast static deep learning with streaming adaptive conformal inference in dynamic cleanroom environments.

---

### SLIDE 13: Out-of-Distribution Defect Detection & Chamber Quarantine
- **Slide Focus:** Rejecting unseen defect chemistries and triggering equipment interlocks.
- **Images on Slide (2 Visuals Side-by-Side):**
  1. `Visual 13A [OOD ROC & Precision-Recall Curves]`:
     - ROC Curve: **AUROC = 0.9941** (False Positive Rate vs True Positive Rate)
     - PR Curve: **AUPR = 0.9792** (Recall vs Precision)
     - Evaluated on challenging quadruple compound excursions (unseen during training).
  2. `Visual 13B [Conformal Empty-Set Gate Schematic]`:
     - Flow diagram: Nonconformity Score $E(X, y) > \hat{\tau}_{\text{cal}} \implies C(X) = \emptyset$.
     - Immediate trigger: **Chamber Quarantine Signal** sent to fab SCADA system, halting tool before serial lot scrap.
- **Slide Text (3 Bullets):**
  * **The Unseen Risk:** Standard networks hallucinate incorrect defect labels on novel process excursion chemistries.
  * **SOTA OOD Rejection:** Achieves **0.9941 AUROC** using conformal empty prediction sets.
  * **Instant Tool Interlock:** Quarantines failing chambers instantly, preventing serial multi-lot contamination.
- **Speaker Notes:** Explain how the empty set property of conformal prediction functions as a mathematically rigorous OOD detector.

---

### SLIDE 14: Closed-Loop Cleanroom Metrology & Automated Dispatch
- **Slide Focus:** End-to-end operational integration of DualCoord in an autonomous semiconductor fab.
- **Images on Slide (1 Comprehensive Decision Flowchart Visual):**
  1. `Visual 14A [Closed-Loop Metrology & Triage Decision Tree]`:
     - Block 1: In-Line Optical Wafer Inspection ($52 \times 52$ binary defect map)
     - Block 2: DualCoord Edge Inference (2.82 ms GPU latency) $\to$ Generates RAPS prediction set $C(X)$
     - Decision Diamond: Evaluate Conformal Set Cardinality $|C(X)|$
       * **Branch 1 [$|C(X)| = 1$ (97.76%)]:** Decisive Singleton $\to$ Automated Tool Routing to Litho/Etch tracks (Zero labor).
       * **Branch 2 [$|C(X)| > 1$ (2.24%)]:** Ambiguous Mix $\to$ Forward to Metrology Engineer with candidate classes (-59.3% queue).
       * **Branch 3 [$|C(X)| = \emptyset$ (OOD)]:** Novel Excursion $\to$ Immediate Chamber Quarantine & Retraining Buffer.
     - Feedback Loop: ACI parameter updates streaming continuously; quarantined wafers feed automated fine-tuning pipeline.
- **Slide Text (3 Bullets):**
  * **Three-Tier Intelligent Triage:** Singletons automated, ambiguous reviewed, novel quarantined.
  * **Real-Time Integration:** 2.82 ms total decision cycle fits seamlessly into high-speed track metrology.
  * **Closed-Loop Learning:** Quarantined excursion wafers automatically update calibration and fine-tuning buffers.
- **Speaker Notes:** Walk the committee step-by-step through the industrial metrology decision tree.

---

### SLIDE 15: Cleanroom Economics & Yield Recovery Ledger
- **Slide Focus:** Comprehensive financial audit under SEMATECH standard cost modeling.
- **Images on Slide (2 Visuals Side-by-Side):**
  1. `Visual 15A [SEMATECH Cost-Loss Ledger Table (Booktabs)]`:
     - Line Item | Baseline Metrology | DualCoord Deployed | Net Impact
     - Undetected Escapes Cost | $687,200 | $429,200 | **+$258,000 saved**
     - Catastrophic Lot Scraps Avoided | 0 lots | 24 lots | **+$238,800 saved** ($9,950/lot)
     - Metrology Review Labor Cost | $49,200 | $79,300 | -$30,100 (triage cost)
     - Total Annual Cleanroom Loss | **$736,400** | **$508,500** | **+$227,900 net savings (-30.95%)**
  2. `Visual 15B [Cumulative Savings & Payback Curve]`:
     - Edge GPU Hardware Capital Cost: $11,500
     - Cumulative daily net yield recovery curve crossing breakeven at **Day 18**.
- **Slide Text (3 Bullets):**
  * **30.95% Net Cost Reduction:** Annual cleanroom loss reduced from $736,400 to $508,500.
  * **24 Catastrophic Lots Saved:** Prevents $238,800 in total lot write-offs by catching subtle compound defects.
  * **Rapid 18-Day Payback:** Edge GPU deployment costs fully recovered in under three weeks of fab operation.
- **Speaker Notes:** Review the financial figures with precision, explaining the distinction between gross scrap prevention and net savings.

---

### SLIDE 16: Research Engineering Roadmap & Scientific Verification
- **Slide Focus:** 22-week engineering timeline, empirical verification milestones, and publication plan.
- **Images on Slide (1 Comprehensive Gantt Timeline Visual):**
  1. `Visual 16A [22-Week Engineering Roadmap Gantt Chart]`:
     - **Phase 1 [Weeks 1-4]:** Baseline Auditing & 15-Paper Forensic Literature Survey (Completed)
     - **Phase 2 [Weeks 5-10]:** 44-Trial Architecture Campaign & Physical Failure Diagnosis (Completed)
     - **Phase 3 [Weeks 11-14]:** Conformal RAPS Calibration & Distribution-Free Risk Proofs (Completed)
     - **Phase 4 [Weeks 15-18]:** Online ACI Drift Streaming & SEMATECH Economic Ledger (Completed)
     - **Phase 5 [Weeks 19-22]:** Cleanroom Edge Hardware Pilot & IEEE Transactions Manuscript Submission (Active)
- **Slide Text (3 Bullets):**
  * **Systematic 22-Week Campaign:** Executed 44 rigorous empirical trials to discover and solve coordinate failure modes.
  * **Open Science & Reproducibility:** Fixed seeds, audited integer counts, and frozen canonical splits ensure 100% verifiability.
  * **Final Manuscript Submission:** Preparing submission to IEEE Transactions on Semiconductor Manufacturing.
- **Speaker Notes:** Conclude the presentation, summarize core contributions, and invite committee questions.

---

## EXECUTION GUIDELINES FOR CLAUDE:
1. **Audit the Provided Assets:** Review the 19 files in `DualCoord_Wafer_Clean_Results.zip`.
2. **Re-Generate Any Visuals Where Needed:** If you want a more unified color palette, composite multi-image slides, or refined charts, use the exact numerical datasets provided above to render new visual assets.
3. **Assemble the Presentation:** Combine clean layouts, generous whitespace, large titles, 3-4 minimal bullets per slide, and multi-visual placements. Embed full speaker notes into each slide's metadata.
```
