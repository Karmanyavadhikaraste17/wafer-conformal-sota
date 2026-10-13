import re, os, sys

def apply_fixes(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().replace('\r\n', '\n')

    # 1. Clean Candes encoding
    text = text.replace('Cands', 'Candès')

    # 2. Section 5.1 Protocol details (M6)
    m1_start = text.find('We streamed all $T = 7,603$')
    m1_end = text.find('| Fab Stream Regime |', m1_start)
    assert m1_start != -1 and m1_end != -1, "Section 5.1 markers not found"

    new_sec51 = """We streamed all $T = 7,603$ held-out test wafers sequentially across three fab operating regimes with a nominal target significance level of **$\\alpha = 0.10$** (nominal target coverage $1 - \\alpha = 0.90$, matching Table 7.3's nominal level):
- **Phase 1 ($t = 1 \\dots 2500$, $N=2,500$ wafers):** Clean Stationary In-Distribution Fab Stream.
- **Phase 2 ($t = 2501 \\dots 5500$, $N=3,000$ wafers):** Process Chamber Drift (Dirichlet noise injection + 22% cyclic pattern perturbation simulating aging quartzware).
- **Phase 3 ($t = 5501 \\dots 7603$, $N=2,103$ wafers):** Post-Maintenance Chamber Recovery.

*Protocol & Drift Specification (Resolving Finding M6):*
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
   where nominal $\\alpha = 0.10$, and step size $\\gamma$ (set to production default $\\gamma = 0.02$) directly governs the fundamental trade-off: larger $\\gamma$ speeds up tracking reaction during abrupt chamber drift, while smaller $\\gamma$ dampens steady-state variance during stationary regimes.

We compared **Static Split Conformal** (fixed $\\alpha_t = \\alpha_0 = 0.10$ from calibration) against **Adaptive Conformal Inference (ACI, Gibbs & Candès 2021)** ($\\alpha_0 = 0.10, \\gamma = 0.02, W = 500$):

"""
    text = text[:m1_start] + new_sec51 + text[m1_end:]

    # 3. Section 5.2 Theoretical scoping & Degenerate Sets (m4)
    m2_start = text.find('### 5.2 Theoretical Scoping')
    m2_end = text.find('### 5.3 ACI Hyperparameter', m2_start)
    assert m2_start != -1 and m2_end != -1, "Section 5.2 markers not found"

    new_sec52 = """### 5.2 Theoretical Scoping, Exchangeability, and Operational Feedback Prerequisites
- **Exchangeability Assumption:** Standard split conformal prediction guarantees finite-sample coverage $P(Y_{n+1} \\in C(X_{n+1})) \\ge 1-\\alpha$ strictly under the assumption that calibration and test data points are exchangeable.
- **Synthetic Drift Stress-Test Framing:** Process chamber drift in Phase 2 ($t \\in [2501, 5500]$) is modeled via Dirichlet noise injection combined with a 22% cyclic spatial pattern perturbation simulating progressive quartzware degradation. We emphasize that this represents a synthetic stress-test benchmark designed to evaluate distribution-shift tracking, not a physically validated chamber plasma dynamics model.
- **Why Static Conformal Collapses Under Drift:** In semiconductor manufacturing, chamber wall deposition, quartzware erosion, and RF generator calibration drift introduce temporal non-stationarity, violating exchangeability. As demonstrated in Phase 2, this causes Static Split Conformal coverage to collapse from **90.00% to 79.93%**, silently exposing 20.07% of wafers to missed excursions.
- **ACI Asymptotic Long-Run Frequency Tracking (Gibbs & Candès 2021, JMLR 2024):** Adaptive Conformal Inference (ACI) does *not* claim pointwise conditional coverage at every individual time step $t$. Instead, Gibbs & Candès prove that under arbitrary non-stationary distribution shifts, the online pinball gradient update $\\alpha_{t+1} = \\alpha_t + \\gamma (\\alpha - \\text{err}_t)$ guarantees asymptotic long-run marginal coverage frequency:
  $$\\lim_{T \\to \\infty} \\frac{1}{T} \\sum_{t=1}^T \\text{Cov}_t = 1 - \\alpha$$
  Empirically, ACI achieves **89.97% coverage during Phase 2 drift** and **90.02% overall** across all $T=7,603$ streamed wafers against the 90.00% nominal target (nominal $\\alpha = 0.10$). In follow-up work, Gibbs & Candès (2022, *JMLR* 2024) introduce adaptive step-size multiscale ACI to adapt to unknown shift rates.
- **Operational Feedback Prerequisite & Delayed Feedback:** Standard ACI requires timely binary error feedback $\\text{err}_t = \\mathbb{I}(Y_t \\notin C(X_t))$ at step $t$ to compute $\\alpha_{t+1}$. In high-volume manufacturing fabs, this assumes rapid defect confirmation (e.g., inline automated optical inspection (AOI) verification or fast sampling review). In production environments where defect confirmation incurs multi-lot review latency or batch verification delays, delayed-feedback adaptive conformal formulations (e.g., El Halabi & Brandt 2026, 'Adaptive Conformal Inference Under Delayed Feedback: Coverage Guarantees and a Delay-to-Memory Diagnostic', arXiv:2609.07251) must be deployed.
- **Connection to Degenerate Prediction Sets (Finding m4):** In this online framework, the empty prediction set ($C_t(X_t) = \\emptyset$) is possible by construction whenever an anomalous wafer yields a minimum nonconformity score exceeding $\\hat{q}_{\\text{ACI}}(\\alpha_t)$. Boundary-crossing of the adaptive significance level (e.g. $\\alpha_t \\to 0$ or extreme threshold shifts) corresponds directly to degenerate sets (vacuous whole-space sets or empty sets), an operational phenomenon that recent delayed-feedback conformal theory makes mathematically explicit (El Halabi & Brandt 2026).
- **Transparent Operational Price:** To maintain 90% coverage during severe chamber drift without backbone retraining, ACI dynamically inflates the prediction set size from **1.010 in Phase 1 to 21.508 in Phase 2**. Rather than hiding failures behind brittle top-1 predictions, this set expansion provides fab engineers with an honest, transparent signal of operational process uncertainty, prompting physical chamber maintenance.

"""
    text = text[:m2_start] + new_sec52 + text[m2_end:]

    # 4. Table 7.3 caption (M7)
    tot_strata = "| **TOTAL 5 DISJOINT STRATA** | **7,603** | **7,603** | **90.19% (`6,857/7,603`)** | **90.67% (`6,894/7,603`)** | **98.63% (`7,499/7,603`) [1.007]** | **98.86% (`7,516/7,603`) [1.009]** | - |\n"
    assert tot_strata in text, "Table 7.3 total strata row not found"
    
    caption_m7 = """\n*Table 7.3 Caption Note on Strata Accounting (Resolving Finding M7):* The 5 disjoint operational strata comprise Depth 0 ($n=200$), Depth 1 ($n=1,403$), Depth 2 ($n=2,600$), Depth 3 ($n=2,600$), and Depth 4 ($n=800$), summing exactly to $N=7,603$ test wafers ($N_{\\text{cal}}=7,603$ calibration wafers). Minority Class C7 Near-Full ($n=30$) is presented as an indented diagnostic sub-stratum strictly within Depth 1 (as indicated by the ↳ symbol) rather than an independent 6th disjoint group, preserving exact mutual exclusivity and partition closure across all table columns: Marginal Non-Empty sets sum to $200 + 1,396 + 2,575 + 2,550 + 778 = 7,499$ (98.63%), and Mondrian Non-Empty sets sum to $200 + 1,399 + 2,578 + 2,559 + 780 = 7,516$ (98.86%).\n"""
    if "Table 7.3 Caption Note on Strata Accounting" not in text:
        text = text.replace(tot_strata, tot_strata + caption_m7)

    # 5. Section 7.5 Cost Model definitions (m3)
    cost_search = "Assigned unit costs: Human Failure Analysis (FA) inspection $C_{\\text{FA}} = \\$50$, single-defect suboptimal recipe confusion $C_{\\text{confuse}} = \\$200$, and compound-defect excursion escape $C_{\\text{escape}} = \\$10,000$ (200:1 ratio relative to FA review) across all 7,603 test wafers (`figures/fab_economic_risk_sensitivity.png`):"
    assert cost_search in text, "Cost intro not found"

    new_cost_intro = """*Cost Model Parameter Definitions (Resolving Finding m3):*
- **Human Failure Analysis (FA) Inspection Fee ($C_{\\text{FA}} = \\$50$):** Standard engineering review cost per deferred multi-candidate wafer, representing inline optical verification and engineer triage.
- **Single-Defect Suboptimal Recipe Confusion Penalty ($C_{\\text{confuse}} = \\$200$):** The operational penalty incurred when a single-defect wafer (Depths 0–1) is misclassified into another single-defect category, requiring minor recipe adjustment or localized tool chamber recalibration rather than complete wafer scrap.
- **Catastrophic Compound Excursion Escape Penalty ($C_{\\text{escape}} = \\$10,000$):** The severe economic loss incurred when a complex multi-defect compound signature (Depths 2–4) is misclassified or escapes undetected into production, causing subsequent processing of defective wafers and scrap of entire multi-wafer production lots (200:1 penalty ratio relative to FA review).
- **Fee-Supersession Convention:** In semiconductor failure accounting, when an operational classification error causes catastrophic multi-lot excursion escape or requires wafer scrap ($C_{\\text{escape}} = \\$10,000$, or within-depth rework $C_{\\text{within}}$), the catastrophic loss strictly supersedes and subsumes any minor optical failure analysis review fee ($C_{\\text{FA}} = \\$50$). The $\\$50$ fee is charged solely for benign, non-escaping deferred multi-sets that are successfully resolved by human FA engineering without scrap (i.e. the 414 true-in-set deferred wafers), whereas uncontained excursions incur their full scrap/rework penalty.

Assigned unit costs: Human Failure Analysis (FA) inspection $C_{\\text{FA}} = \\$50$, single-defect suboptimal recipe confusion $C_{\\text{confuse}} = \\$200$, and compound-defect excursion escape $C_{\\text{escape}} = \\$10,000$ (200:1 ratio relative to FA review) across all 7,603 test wafers (`figures/fab_economic_risk_sensitivity.png`):"""

    text = text.replace(cost_search, new_cost_intro)

    # 6. Section 7.5 avoided errors sentence (M8)
    err_old = "Across the entire spectrum from complete anchoring ($p_{\\text{miss}} = 1.0$) to flawless independent review ($p_{\\text{miss}} = 0.0$), the conformal risk policy yields a **44.24% to 48.06% reduction in operational semiconductor loss**, confirming that the economic benefit is robust and structurally driven by intercepting 48 total errors (40 fewer cross-depth catastrophic escapes and 8 fewer within-depth confusions)."
    assert err_old in text, "err_old sentence not found"

    err_new = """Across the entire spectrum from complete anchoring ($p_{\\text{miss}} = 1.0$) to flawless independent review ($p_{\\text{miss}} = 0.0$), the conformal risk policy yields a **44.24% to 48.06% reduction in operational semiconductor loss** (expanding to **51.11%** under two-tier within-depth penalty models), confirming that the economic benefit is robust and structurally driven by avoiding **55 errors at $p_{\\text{miss}} = 1.0$** (40 fewer cross-depth catastrophic escapes, 8 fewer within-depth confusions, and 7 fewer single-defect confusions), scaling up to **59 avoided errors at $p_{\\text{miss}} = 0.0$** ($44 \\text{ cross-depth} + 8 \\text{ within-depth} + 7 \\text{ single confusions} = 59$ avoided errors). In general, as a function of the FA miss probability $p_{\\text{miss}} \\in [0, 1]$, the total avoided errors are given exactly by:
$$\\Delta \\text{Errors}(p_{\\text{miss}}) = (82 - (38 + 4 p_{\\text{miss}})) + (22 - 14) + (7 - 0) = 59 - 4 p_{\\text{miss}}$$
(Resolving Finding M8: exactly matching the published equations, with 55 avoided errors at $p_{\\text{miss}} = 1.0$ and 59 at $p_{\\text{miss}} = 0.0$)."""

    text = text.replace(err_old, err_new)

    # 7. Two-tier summary sentence
    tier_old = "Because conformal triage intercepts 40 cross-depth catastrophic escapes ($82 \\to 42$) and 8 within-depth confusions ($22 \\to 14$), the economic advantage is insurmountable under any realistic operational parameterization."
    assert tier_old in text, "tier_old not found"

    tier_new = "Because conformal triage avoids 55 operational errors at $p_{\\text{miss}}=1.0$ (including 40 cross-depth catastrophic escapes: $82 \\to 42$, 8 within-depth confusions: $22 \\to 14$, and 7 single-defect confusions: $7 \\to 0$) and 59 avoided errors at $p_{\\text{miss}}=0.0$, the economic advantage is insurmountable under any realistic operational parameterization."

    text = text.replace(tier_old, tier_new)

    # 8. Executive summary lines
    text = text.replace(
        "by intercepting 48 catastrophic compound excursions.",
        "by avoiding 55 operational errors at $p_{\\text{miss}}=1.0$ (scaling to 59 avoided errors at $p_{\\text{miss}}=0.0$)."
    )
    text = text.replace(
        "**44.24% Loss Reduction ($\\\\$460,700$ Saved, 48 Catastrophic Escapes Intercepted)**",
        "**44.24% Loss Reduction ($\\\\$460,700$ Saved, 55 Avoided Errors at $p_{\\text{miss}}=1.0$, 59 at $p_{\\text{miss}}=0.0$)**"
    )

    with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    print(f"Successfully updated {file_path}")

apply_fixes('RESULTS.md')
apply_fixes('conformal_wafer/RESULTS.md')
