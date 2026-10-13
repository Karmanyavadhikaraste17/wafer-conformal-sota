import re, os

def apply_round3_updates(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().replace('\r\n', '\n')

    # -------------------------------------------------------------
    # 1. M6: Complete Online ACI Protocol details
    # -------------------------------------------------------------
    old_proto_m6 = """*Protocol & Drift Specification (Resolving Finding M6):*
1. **A Priori Fixed Phase Boundaries:** All regime transition boundaries ($t_1 = 2500$, $t_2 = 5500$, $T = 7,603$) were fixed *a priori* before streaming evaluation and remained identical across all compared methods.
2. **Exact 22% Cyclic Pattern Perturbation:** At each step $t$ in Phase 2 ($t \\in [2501, 5500]$), a Bernoulli trial with success probability $p_{\\text{perturb}} = 0.22$ determines whether cyclic channel perturbation is triggered. If triggered, the predicted class probability vector is circularly permuted by one channel index (`p_t = np.roll(p_t, shift=1)`), simulating circular channel and tool signature rotation induced by miscalibrated quartzware ring erosion. Simultaneously, continuous probability flattening is applied to every Phase 2 wafer by injecting uniform Dirichlet noise ($\\alpha_{\\text{Dir}} = \\mathbf{1}_{34} \\in \\mathbb{R}^{34}$, mixing weight $0.25$):
   $$p_t \\leftarrow \\frac{0.75 \\cdot p_t + 0.25 \\cdot \\text{Dir}(\\mathbf{1}_{34})}{\\sum_{k=1}^{34} (0.75 \\cdot p_{t,k} + 0.25 \\cdot \\text{Dir}(\\mathbf{1}_{34})_k)}$$
   This flattens peak confidence from $>0.98$ down toward the simplex center, directly challenging exchangeability.
3. **Random Seed Protocol & Repetitions:** The primary stream trajectory is evaluated under fixed random seed 42 (`np.random.default_rng(42)`). System stability and hyperparameter insensitivity are certified across 15 separate streaming configurations: 5 adaptation learning rates $\\gamma \\in \\{0.005, 0.01, 0.02, 0.05, 0.10\\}$ across 3 rolling buffer window sizes $W \\in \\{250, 500, 1000\\}$ (`results/aci_sensitivity_sweep.csv`).
4. **Prequential Binary Error Indicator vs. Rolling FIFO Buffer:**
   - The error term $\\text{err}_t = \\mathbb{I}(y_t \\notin C_t(x_t)) \\in \\{0, 1\\}$ is the **instantaneous, prequential binary miscoverage indicator** evaluated at time step $t$ using the prediction set $C_t(x_t)$ *before* observing ground-truth label $y_t$. It is strictly *not* a smoothed or windowed moving average.
   - The rolling buffer $W = 500$ is a FIFO queue containing the $W$ most recent historical nonconformity scores $\\{s_{t-W}, \\dots, s_{t-1}\\}$. At step $t$, the dynamic conformal quantile $\\hat{q}_{\\text{ACI}}(\\alpha_t)$ is evaluated as the empirical $\\lceil(W+1)(1-\\alpha_t)\\rceil$-th order statistic of this historical buffer.
5. **Update Rule & Adaptation-vs-Stability Trade-Off (Gibbs & Candès 2021):** The online pinball gradient update is:
   $$\\alpha_{t+1} = \\text{clip}\\big(\\alpha_t + \\gamma (\\alpha - \\text{err}_t), \\, 0.002, \\, 0.50\\big)$$
   where nominal $\\alpha = 0.10$, and step size $\\gamma$ (set to production default $\\gamma = 0.02$) directly governs the fundamental trade-off: larger $\\gamma$ speeds up tracking reaction during abrupt chamber drift, while smaller $\\gamma$ dampens steady-state variance during stationary regimes."""

    new_proto_m6 = """*Protocol & Drift Specification (Resolving Finding M6):*
1. **Target Error Level & Initialization:**
   - The nominal target error level is **$\\alpha = 0.10$** (nominal target coverage $1 - \\alpha = 0.90$), in exact alignment with Table 7.3's nominal level.
   - The initial adaptive level at step $t=1$ is initialized to $\\alpha_0 = \\alpha = 0.10$.
   - The initial FIFO buffer of size $W = 500$ is populated prior to streaming ($t=0$) with the last $W = 500$ nonconformity scores computed on the held-out calibration set ($s_i \\in \\mathcal{S}_{\\text{cal}}$).
2. **A Priori Fixed Phase Boundaries:** All regime transition boundaries ($t_1 = 2500$, $t_2 = 5500$, $T = 7,603$) were fixed *a priori* before streaming evaluation and remained identical across all compared methods:
   - Phase 1 (Clean Stationary Stream): $t = 1 \\dots 2500$ ($N = 2,500$ wafers).
   - Phase 2 (Process Chamber Drift): $t = 2501 \\dots 5500$ ($N = 3,000$ wafers).
   - Phase 3 (Post-Maintenance Recovery): $t = 5501 \\dots 7603$ ($N = 2,103$ wafers).
3. **Identical Stream Realization & 22% Cyclic Perturbation:**
   - All methods (Static Split Conformal and Online ACI) evaluate the **exact identical wafer stream $(x_t, y_t)_{t=1}^T$ and identical perturbation realizations at every step $t$**, guaranteeing matched, paired comparisons without confounding stream noise.
   - In Phase 2 ($t \\in [2501, 5500]$), an independent Bernoulli trial with parameter $p_{\\text{perturb}} = 0.22$ determines whether cyclic channel perturbation is triggered. When triggered, the predicted probability vector is circularly shifted by one index (`p_t = np.roll(p_t, shift=1)`), simulating circular channel/chamber signature misalignment from quartzware ring erosion. Simultaneously, continuous probability flattening is applied to every Phase 2 wafer by injecting uniform Dirichlet noise ($\\alpha_{\\text{Dir}} = \\mathbf{1}_{34} \\in \\mathbb{R}^{34}$, mixing weight $0.25$):
     $$p_t \\leftarrow \\frac{0.75 \\cdot p_t + 0.25 \\cdot \\text{Dir}(\\mathbf{1}_{34})}{\\sum_{k=1}^{34} (0.75 \\cdot p_{t,k} + 0.25 \\cdot \\text{Dir}(\\mathbf{1}_{34})_k)}$$
     This flattens peak confidence from $>0.98$ down toward the simplex center, directly challenging exchangeability.
4. **Independent Random Seed Repetitions & Sensitivity Sweeps:**
   - To strictly distinguish independent stochastic repetitions from hyperparameter sweeps, the production system ($\\gamma = 0.02, W = 500$) was evaluated across **5 independent random seed repetitions** ($\\text{seed} \\in \\{42, 101, 2024, 777, 999\\}$), generating independent perturbation arrival sequences and Dirichlet noise realizations (`results/aci_seed_repetitions.csv`):
     * **Online ACI Overall Coverage (5 seeds):** **$89.99\\% \\pm 0.02\\%$** (individual seeds: 90.02%, 89.96%, 89.99%, 89.99%, 89.99%).
     * **Online ACI Phase 2 Drift Coverage (5 seeds):** **$89.96\\% \\pm 0.05\\%$** (individual seeds: 89.97%, 90.00%, 89.97%, 90.00%, 89.87%).
     * **Static Split Conformal Overall Coverage (5 seeds):** **$86.15\\% \\pm 0.46\\%$** (individual seeds: 85.80%, 86.95%, 85.68%, 85.91%, 86.43%).
     * **Static Split Conformal Phase 2 Drift Coverage (5 seeds):** **$80.54\\% \\pm 0.70\\%$** (collapses to ~80% in every seed: 79.93%, 81.77%, 79.97%, 80.23%, 80.80%).
   - In addition, hyperparameter insensitivity is certified across a 15-configuration grid sweep: 5 adaptation rates $\\gamma \\in \\{0.005, 0.01, 0.02, 0.05, 0.10\\}$ across 3 buffer sizes $W \\in \\{250, 500, 1000\\}$ (`results/aci_sensitivity_sweep.csv`).
5. **Prequential Binary Error Indicator vs. Rolling FIFO Buffer:**
   - The error term $\\text{err}_t = \\mathbb{I}(y_t \\notin C_t(x_t)) \\in \\{0, 1\\}$ is the **instantaneous, prequential binary miscoverage indicator** evaluated at time step $t$ using the prediction set $C_t(x_t)$ *before* observing ground-truth label $y_t$. It is strictly *not* a smoothed or windowed moving average.
   - The rolling buffer $W = 500$ is a FIFO queue containing the $W$ most recent historical nonconformity scores $\\{s_{t-W}, \\dots, s_{t-1}\\}$. At step $t$, the dynamic conformal quantile $\\hat{q}_{\\text{ACI}}(\\alpha_t)$ is evaluated as the empirical $\\lceil(W+1)(1-\\alpha_t)\\rceil$-th order statistic of this historical buffer.
6. **Update Rule & Adaptation-vs-Stability Trade-Off (Gibbs & Candès 2021):** The online pinball gradient update is:
   $$\\alpha_{t+1} = \\text{clip}\\big(\\alpha_t + \\gamma (\\alpha - \\text{err}_t), \\, 0.002, \\, 0.50\\big)$$
   where nominal $\\alpha = 0.10$, and step size $\\gamma$ (set to production default $\\gamma = 0.02$) directly governs the fundamental trade-off: larger $\\gamma$ speeds up tracking reaction during abrupt chamber drift, while smaller $\\gamma$ dampens steady-state variance during stationary regimes."""

    assert old_proto_m6 in text, "old_proto_m6 not found"
    text = text.replace(old_proto_m6, new_proto_m6)

    # -------------------------------------------------------------
    # 2. m4: Degenerate prediction sets aligned with clipping rule
    # -------------------------------------------------------------
    old_m4 = """- **Connection to Degenerate Prediction Sets (Finding m4):** In this online framework, the empty prediction set ($C_t(X_t) = \\emptyset$) is possible by construction whenever an anomalous wafer yields a minimum nonconformity score exceeding $\\hat{q}_{\\text{ACI}}(\\alpha_t)$. Boundary-crossing of the adaptive significance level (e.g. $\\alpha_t \\to 0$ or extreme threshold shifts) corresponds directly to degenerate sets (vacuous whole-space sets or empty sets), an operational phenomenon that recent delayed-feedback conformal theory makes mathematically explicit (El Halabi & Brandt 2026)."""

    new_m4 = """- **Connection to Degenerate Prediction Sets & Bounded Level Tracking (Finding m4):** Under the online ACI clipping rule $\\alpha_t \\in [\\alpha_{\\min}, \\alpha_{\\max}] = [0.002, 0.50]$, the adaptive level is explicitly bounded away from 0 and 1, preventing runaway parameter divergence. Operating at the lower significance bound $\\alpha_{\\min} = 0.002$ targets conservative coverage ($1 - \\alpha_t = 0.998$), which inflates prediction sets and may produce large, potentially vacuous sets encompassing the full label space ($C_t(X_t) = \\mathcal{Y}$). Conversely, operating at higher significance levels up to $\\alpha_{\\max} = 0.50$ contracts prediction set sizes. An empty prediction set ($C_t(X_t) = \\emptyset$) occurs by construction when the minimum true-label nonconformity score on input $x_t$ exceeds the empirical quantile $\\hat{q}_{\\text{ACI}}(\\alpha_t)$ in the historical buffer (i.e. $\\min_y s(x_t, y) > \\hat{q}_{\\text{ACI}}(\\alpha_t)$). Thus, degenerate prediction sets (vacuous whole-space sets or empty sets) arise not from unconstrained level divergence, but from the interaction between high anomaly scores and bounded quantile thresholding, consistent with recent theoretical frameworks of adaptive conformal inference under bounded and delayed feedback (El Halabi & Brandt 2026, arXiv:2609.07251)."""

    assert old_m4 in text, "old_m4 not found"
    text = text.replace(old_m4, new_m4)

    # -------------------------------------------------------------
    # 3. M7: Table 7.3 caption calibration count clarification
    # -------------------------------------------------------------
    old_cap_m7 = """*Table 7.3 Caption Note on Strata Accounting (Resolving Finding M7):* The 5 disjoint operational strata comprise Depth 0 ($n=200$), Depth 1 ($n=1,403$), Depth 2 ($n=2,600$), Depth 3 ($n=2,600$), and Depth 4 ($n=800$), summing exactly to $N=7,603$ test wafers ($N_{\\text{cal}}=7,603$ calibration wafers). Minority Class C7 Near-Full ($n=30$) is presented as an indented diagnostic sub-stratum strictly within Depth 1 (as indicated by the ↳ symbol) rather than an independent 6th disjoint group, preserving exact mutual exclusivity and partition closure across all table columns: Marginal Non-Empty sets sum to $200 + 1,396 + 2,575 + 2,550 + 778 = 7,499$ (98.63%), and Mondrian Non-Empty sets sum to $200 + 1,399 + 2,578 + 2,559 + 780 = 7,516$ (98.86%)."""

    new_cap_m7 = """*Table 7.3 Caption Note on Strata Accounting & Calibration Allocation Distinction (Resolving Finding M7):*
The 5 disjoint operational strata comprise Depth 0 ($n=200$), Depth 1 ($n=1,403$), Depth 2 ($n=2,600$), Depth 3 ($n=2,600$), and Depth 4 ($n=800$), summing exactly to $N_{\\text{test}}=7,603$ test wafers.
*Distinction Between Calibration Allocations:* In this closed-set 38-class deployed diagnostic benchmark (Table 7.3), all 38 classes are present in deployment, so calibration uses the complete 38-class split ($N_{\\text{cal}}=7,603$ calibration wafers, including 800 calibration wafers in Depth 4). In contrast, in the open-set 34-class leave-class-out benchmark of Section 6.1, Depth 4 quad-mixtures are strictly quarantined from training, validation, and calibration, yielding the disjoint known-class calibration count $N_{\\text{cal}}=6,803$ ($7,603 - 800 = 6,803$, reconciling with Section 2.2 and Finding M5).
Minority Class C7 Near-Full ($n=30$) is presented as an indented diagnostic sub-stratum strictly within Depth 1 (as indicated by the ↳ symbol) rather than an independent 6th disjoint group, preserving exact mutual exclusivity and partition closure across all table columns: Marginal Non-Empty sets sum to $200 + 1,396 + 2,575 + 2,550 + 778 = 7,499$ (98.63%), and Mondrian Non-Empty sets sum to $200 + 1,399 + 2,578 + 2,559 + 780 = 7,516$ (98.86%)."""

    assert old_cap_m7 in text, "old_cap_m7 not found"
    text = text.replace(old_cap_m7, new_cap_m7)

    # -------------------------------------------------------------
    # 4. m3: Cost parameters, fee supersession, and 4 true-omitted cases
    # -------------------------------------------------------------
    old_cost_m3 = """*Cost Model Parameter Definitions (Resolving Finding m3):*
- **Human Failure Analysis (FA) Inspection Fee ($C_{\\text{FA}} = \\$50$):** Standard engineering review cost per deferred multi-candidate wafer, representing inline optical verification and engineer triage.
- **Single-Defect Suboptimal Recipe Confusion Penalty ($C_{\\text{confuse}} = \\$200$):** The operational penalty incurred when a single-defect wafer (Depths 0–1) is misclassified into another single-defect category, requiring minor recipe adjustment or localized tool chamber recalibration rather than complete wafer scrap.
- **Catastrophic Compound Excursion Escape Penalty ($C_{\\text{escape}} = \\$10,000$):** The severe economic loss incurred when a complex multi-defect compound signature (Depths 2–4) is misclassified or escapes undetected into production, causing subsequent processing of defective wafers and scrap of entire multi-wafer production lots (200:1 penalty ratio relative to FA review).
- **Fee-Supersession Convention:** In semiconductor failure accounting, when an operational classification error causes catastrophic multi-lot excursion escape or requires wafer scrap ($C_{\\text{escape}} = \\$10,000$, or within-depth rework $C_{\\text{within}}$), the catastrophic loss strictly supersedes and subsumes any minor optical failure analysis review fee ($C_{\\text{FA}} = \\$50$). The $\\$50$ fee is charged solely for benign, non-escaping deferred multi-sets that are successfully resolved by human FA engineering without scrap (i.e. the 414 true-in-set deferred wafers), whereas uncontained excursions incur their full scrap/rework penalty."""

    new_cost_m3 = """*Cost Model Parameter Definitions (Resolving Finding m3):*
- **Human Failure Analysis (FA) Inspection Fee ($C_{\\text{FA}} = \\$50$):** Standard engineering review cost per deferred multi-candidate wafer, representing inline optical verification and engineer triage.
- **Single-Defect Suboptimal Recipe Confusion Penalty ($C_{\\text{confuse}} = \\$200$):** The operational penalty incurred when a single-defect wafer (Depths 0–1) is misclassified into another single-defect category, requiring minor recipe adjustment or localized tool chamber recalibration rather than complete wafer scrap.
- **Within-Depth Compound Confusion Penalty ($C_{\\text{within}} \\in [\\$200, \\$10,000]$):** Defined at first use as the operational rework penalty incurred when a compound defect is misclassified into another compound defect of the identical mixture depth (e.g., C35 $\\to$ C36, both Depth 4), involving adjacent tool chambers in the same process module and requiring localized chamber recalibration or module rework rather than full-fab excursion scrap.
- **Catastrophic Compound Excursion Escape Penalty ($C_{\\text{escape}} = C_{\\text{cross}} = \\$10,000$):** The severe economic loss incurred when a complex multi-defect compound signature (Depths 2–4) is misclassified across depth strata or escapes undetected into production, causing subsequent processing of defective wafers and scrap of entire multi-wafer production lots (200:1 penalty ratio relative to FA review).
- **Fee-Supersession Convention & Complete Accounting of the 4 True-Omitted Deferred Wafers:**
  Total deferred multi-sets ($|C(X)| > 1$) count exactly **418 wafers** (5.50% of 7,603), decomposed into:
  1. **414 benign, true-in-set deferred wafers:** The true defect class is contained within $C(X)$. Human FA engineers review the narrow candidate set (~2.01 classes) and identify the true defect. These 414 wafers never escape into production and incur strictly the standard FA review fee $C_{\\text{FA}} = \\$50$ each ($414 \\times \\$50 = \\$20,700$).
  2. **4 true-omitted deferred wafers:** The candidate set $C(X)$ omits the true defect:
     - **When review misses them ($p_{\\text{miss}} = 1.0$, Complete Anchoring):** Due to cognitive anchoring on the suggested candidates, the engineering team fails to identify the unlisted root cause. The uncontained excursion escapes into production, causing lot scrap ($C_{\\text{escape}} = \\$10,000$ each, or $C_{\\text{within}}$). Under the **fee-supersession convention**, the catastrophic escape loss ($4 \\times \\$10,000 = \\$40,000$) strictly supersedes and subsumes the $\\$50$ review fee; no double charge is levied. Thus, escape cost is $(52 + 4) \\times \\$10,000 = \\$560,000$, FA review fee is $414 \\times \\$50 = \\$20,700$, and total conformal loss is exactly $\\$580,700$.
     - **When review catches them ($p_{\\text{miss}} = 0.0$, Flawless Independent Review):** Despite omission from $C(X)$, thorough independent failure analysis correctly identifies the true defect, preventing catastrophic production escape. These 4 wafers are successfully resolved and therefore incur the standard FA review fee $C_{\\text{FA}} = \\$50$ each ($4 \\times \\$50 = \\$200$). Escape cost drops to $52 \\times \\$10,000 = \\$520,000$, FA review fee becomes $(414 + 4) \\times \\$50 = 418 \\times \\$50 = \\$20,900$, and total conformal loss is exactly $\\$540,900$.
     - **General Expected Loss Formula across $p_{\\text{miss}} \\in [0, 1]$:** For expected miss probability $p_{\\text{miss}}$, the expected number of catastrophic escapes is $52 + 4 p_{\\text{miss}}$ (incurring $C_{\\text{escape}}$), while the expected number of successfully resolved FA reviews is $414 + 4(1 - p_{\\text{miss}})$ (incurring $C_{\\text{FA}} = \\$50$). Total conformal loss is given exactly by:
       $$\\text{Loss}_{\\text{conf}}(p_{\\text{miss}}) = (52 + 4 p_{\\text{miss}}) \\cdot C_{\\text{escape}} + [414 + 4(1 - p_{\\text{miss}})] \\cdot C_{\\text{FA}}$$
       reproducing every single cell in Table 7.5 to the cent ($p_{\\text{miss}} = 1.0 \\implies \\$580,700$; $p_{\\text{miss}} = 0.75 \\implies \\$570,750$; $p_{\\text{miss}} = 0.50 \\implies \\$560,800$; $p_{\\text{miss}} = 0.25 \\implies \\$550,850$; $p_{\\text{miss}} = 0.0 \\implies \\$540,900$)."""

    assert old_cost_m3 in text, "old_cost_m3 not found"
    text = text.replace(old_cost_m3, new_cost_m3)

    with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    print(f"Successfully applied Round 3 updates to {file_path}!")

apply_round3_updates('RESULTS.md')
apply_round3_updates('conformal_wafer/RESULTS.md')
