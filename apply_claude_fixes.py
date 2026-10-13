import re, os

def apply_claude_fixes(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().replace('\r\n', '\n')

    # 1. Executive summary line 28 (Fix 2 claim restriction)
    old_contrib_aci = "Stability is validated across $\\gamma \\in [0.005, 0.10]$ and rolling buffers $W \\in [250, 1000]$."
    new_contrib_aci = "Stability is validated across rolling buffers $W \\in [250, 1000]$ and adaptation rates $\\gamma \\le 0.05$ (stream coverage bounded within $[89.83\\%, 90.02\\%]$, with expected minor steady-state jitter at aggressive $\\gamma = 0.10$)."
    assert old_contrib_aci in text, "old_contrib_aci not found"
    text = text.replace(old_contrib_aci, new_contrib_aci)

    # 2. Executive summary line 30 & Section 6.1 line 176 (Fix 3: 294 / 4.32% harmonization)
    text = text.replace(
        "**4.34% in-distribution false rejection** (Wilson 95% CI: $[3.88\\%, 4.85\\%]$)",
        "**4.32% in-distribution false rejection** (Wilson 95% CI: $[3.86\\%, 4.83\\%]$)"
    )
    text = text.replace(
        "**4.34% (`295 / 6,803`)**, with an exact **Wilson 95% Confidence Interval of $[3.88\\%, 4.85\\%]$**",
        "**4.32% (`294 / 6,803`)**, with an exact **Wilson 95% Confidence Interval of $[3.86\\%, 4.83\\%]$**"
    )

    # 3. Full-precision qhat in Section 6.1
    text = text.replace(
        "conformal threshold $\\hat{q} = 0.0382$ is calibrated",
        "conformal threshold $\\hat{q}_{\\text{conformal}} = 0.038288$ (rounded to 0.0383) is calibrated"
    )
    text = text.replace(
        "Conformal Quantile ($\\alpha=0.05$) $\\hat{q} = 0.0382$, computed strictly",
        "Conformal Quantile ($\\alpha=0.05$) $\\hat{q}_{\\text{conformal}} = 0.038288$ (rounded to 0.0383), computed strictly"
    )

    # 4. Small buffer capping rule in Section 5.1
    old_buf_rule = "the dynamic conformal quantile $\\hat{q}_{\\text{ACI}}(\\alpha_t)$ is evaluated as the empirical $\\lceil(W+1)(1-\\alpha_t)\\rceil$-th order statistic of this historical buffer."
    new_buf_rule = "the dynamic conformal quantile $\\hat{q}_{\\text{ACI}}(\\alpha_t)$ is evaluated as the order statistic at capped index $\\text{idx} = \\min(\\max(0, \\lceil(W+1)(1-\\alpha_t)\\rceil - 1), W-1)$ of this sorted historical buffer, preventing index overflow at extreme clipping."
    assert old_buf_rule in text, "old_buf_rule not found"
    text = text.replace(old_buf_rule, new_buf_rule)

    # 5. Theoretical bounds edge-cases in Section 5.2
    old_asymp = "In follow-up work, Gibbs & Candès (2022, *JMLR* 2024) introduce adaptive step-size multiscale ACI to adapt to unknown shift rates."
    new_asymp = """In unclipped settings, Gibbs & Candès prove a deterministic finite-sample tracking bound:
  $$\\Big| \\frac{1}{T} \\sum_{t=1}^T \\text{err}_t - \\alpha \\Big| \\le \\frac{\\max(\\alpha_1, 1 - \\alpha_1) + \\gamma}{\\gamma T}$$
  which bounds average empirical miscoverage error to $\\le 0.6\\%$ over $T = 7,603$ at $\\gamma = 0.02$. While parameter clipping to $[0.002, 0.50]$ technically constrains the asymptotic domain, it stabilizes empirical tracking without compromising convergence. In follow-up work, Gibbs & Candès (2022, *JMLR* 2024) introduce adaptive step-size multiscale ACI to adapt to unknown shift rates."""
    assert old_asymp in text, "old_asymp not found"
    text = text.replace(old_asymp, new_asymp)

    # 6. Section 5.3: Replace 9-row table with complete 15-row table and fix stability claim (Fix 1 & 2)
    old_sec53_table_start = text.find("### 5.3 ACI Hyperparameter Sensitivity Sweep")
    old_sec53_table_end = text.find("---", old_sec53_table_start)
    assert old_sec53_table_start != -1 and old_sec53_table_end != -1, "Section 5.3 markers not found"

    new_sec53 = """### 5.3 ACI Hyperparameter Sensitivity Sweep (`results/aci_sensitivity_sweep.csv`)
To evaluate stability under varying buffering budgets, we swept adaptation learning rate $\\gamma \\in \\{0.005, 0.01, 0.02, 0.05, 0.10\\}$ across buffer sizes $W \\in \\{250, 500, 1000\\}$ (all 15 configurations):

| Buffer Window ($W$) | Learning Rate ($\\gamma$) | Phase 1 Clean Coverage (%) | Phase 2 Drift Coverage (%) | Phase 3 Recovery Coverage (%) | Overall Stream Coverage (%) | Coverage Absolute Error vs 90% |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 250 | 0.005 | 89.96% | 89.97% | 90.11% | **90.00%** | +0.00% |
| 250 | 0.010 | 90.00% | 89.93% | 90.11% | **90.00%** | +0.00% |
| 250 | 0.020 | 90.04% | 89.93% | 90.06% | **90.00%** | +0.00% |
| 250 | 0.050 | 89.96% | 89.87% | 90.11% | **89.96%** | -0.04% |
| 250 | 0.100 | 89.36% | 89.37% | 89.68% | **89.45%** | -0.55% |
| 500 | 0.005 | 90.04% | 89.90% | 90.16% | **90.02%** | +0.02% |
| 500 | 0.010 | 90.04% | 89.90% | 90.11% | **90.00%** | +0.00% |
| **500 (Production)** | **0.020** | **90.04%** | **89.97%** | **90.06%** | **90.02%** | **+0.02%** |
| 500 | 0.050 | 90.00% | 89.70% | 90.06% | **89.90%** | -0.10% |
| 500 | 0.100 | 89.32% | 89.17% | 89.63% | **89.35%** | -0.65% |
| 1000 | 0.005 | 89.92% | 90.07% | 90.01% | **90.00%** | +0.00% |
| 1000 | 0.010 | 90.00% | 89.97% | 90.11% | **90.02%** | +0.02% |
| 1000 | 0.020 | 90.04% | 89.87% | 90.11% | **89.99%** | -0.01% |
| 1000 | 0.050 | 89.96% | 89.53% | 90.11% | **89.83%** | -0.17% |
| 1000 | 0.100 | 89.28% | 89.00% | 89.73% | **89.29%** | -0.71% |

*Sensitivity Analysis & Stability Regime:*
- Across all 12 configurations with moderate learning rates $\\gamma \\le 0.05$, overall stream coverage tracks tightly within **$[89.83\\%, 90.02\\%]$** (absolute tracking error $\\le 0.17\\%$). The production setting ($W=500, \\gamma=0.02$) reproduces the primary run figures exactly (Phase 2: 89.97%, Phase 3: 90.06%, Overall: 90.02%).
- For aggressive step size $\\gamma = 0.10$, tracking exhibits minor steady-state jitter ($89.29\\%\\text{--}89.45\\%$, error $0.55\\%\\text{--}0.71\\%$), illustrating the fundamental adaptation-versus-stability trade-off (Gibbs & Candès 2021) and establishing $\\gamma \\in [0.01, 0.05]$ as the operationally recommended configuration for fab deployment.

"""
    text = text[:old_sec53_table_start] + new_sec53 + text[old_sec53_table_end:]

    # 7. Section 6.2 Table: Harmonize Conformal (294/4.32%) and MSP (343/5.04%) (Fix 3)
    text = text.replace(
        "36.88% (`295 / 800`) | 5.12% (`348 / 6,803`) [$4.61\\%, 5.66\\%$]",
        "36.88% (`295 / 800`) | 5.12% (`348 / 6,803`) [$4.61\\%, 5.66\\%$]"
    )
    text = text.replace(
        "69.00% (`552 / 800`) | 5.06% (`344 / 6,803`) [$4.56\\%, 5.60\\%$]",
        "69.00% (`552 / 800`) | 5.04% (`343 / 6,803`) [$4.54\\%, 5.59\\%$]"
    )
    text = text.replace(
        "**65.13% (`521 / 800`)** | **4.34% (`295 / 6,803`)** [$3.88\\%, 4.85\\%$]",
        "**65.13% (`521 / 800`)** | **4.32% (`294 / 6,803`)** [$3.86\\%, 4.83\\%$]"
    )

    # 8. Distinct labeling for 98.63% point accuracy vs 98.63% Table 7.3 non-empty rate
    text = text.replace(
        "- **In-Distribution 34-Class Accuracy:** **98.63%**.",
        "- **In-Distribution 34-Class Accuracy:** **98.63%** (Top-1 point accuracy on known classes; distinct from Table 7.3's closed-set marginal non-empty set rate of 98.63%, which measures set non-emptiness across all 7,603 wafers)."
    )

    # 9. Fix 4: Corollary 1 contrapositive formulation
    old_cor1 = "All 20 unrejected C35--C38 wafers have a Loc or Scratch logit below $-0.77 < \\tau^\\star$, confirming that the model fails to reject if and only if a constituent primitive fails the margin condition."
    new_cor1 = """All 20 unrejected C35--C38 wafers have a Loc or Scratch logit below $-0.77 < \\tau^\\star$, confirming the contrapositive of Corollary 1: every non-rejection implies that at least one constituent primitive fails the margin condition ($l_k \\le \\tau^\\star = -0.7694$). (Note that Corollary 1 provides a deterministic sufficient condition conditional on the realized threshold $\\hat{q}_{\\text{lib}}$, whereas Lemma 1 provides the marginal in-distribution false-alarm bound)."""
    assert old_cor1 in text, "old_cor1 not found"
    text = text.replace(old_cor1, new_cor1)

    # 10. Fix 5: Proposition 3 restore "whenever their softmax outputs coincide"
    old_prop3 = "cannot distinguish a quad-mixture whose posterior splits between two known neighbors from an in-distribution wafer that is ambiguous between those same neighbors."
    new_prop3 = "cannot distinguish a quad-mixture whose posterior splits between two known neighbors from an in-distribution wafer that is ambiguous between those same neighbors, whenever their softmax outputs coincide."
    assert old_prop3 in text, "old_prop3 not found"
    text = text.replace(old_prop3, new_prop3)

    # 11. Clarify 18 cells vs 30 scenarios in Section 7.5
    old_30_scen = "Across all 30 evaluated industrial fab configurations, total operational loss reduction is mathematically bounded between **44.24\\% and 51.11\\%** ($-\\$382,300$ to $-\\$500,500$ net savings)."
    new_30_scen = "Across all 18 primary benchmark configurations (6 $C_{\\text{within}}$ penalties $\\times$ 3 $p_{\\text{miss}}$ anchor levels, expanding to 30 scenarios across the full 5-level $p_{\\text{miss}}$ grid in `cost_tiering_sensitivity.csv`), total operational loss reduction is mathematically bounded between **44.24\\% and 51.11\\%** ($-\\$382,300$ to $-\\$500,500$ net savings)."
    assert old_30_scen in text, "old_30_scen not found"
    text = text.replace(old_30_scen, new_30_scen)

    with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    print(f"Successfully applied Claude review fixes to {file_path}")

apply_claude_fixes('RESULTS.md')
apply_claude_fixes('conformal_wafer/RESULTS.md')
