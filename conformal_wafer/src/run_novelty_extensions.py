import os
import shutil
import time
import numpy as np
import pandas as pd
import torch

from .config import cfg
from .dataset import get_dataloaders
from .models import get_model
from .train import evaluate_model
from .conformal import ConformalPredictor, evaluate_conformal_predictions
from .plots import (
    plot_coverage_vs_alpha,
    plot_coverage_by_depth,
    plot_setsize_by_depth
)
from .advanced_conformal import (
    run_coverage_nominal_diagnostic,
    run_conformal_triage,
    run_aci_online_drift,
    run_hybrid_embedding_ood,
    run_mondrian_conformal,
    run_calibration_size_sweep,
    run_kreg_and_ensemble_ablation,
    evaluate_economic_loss,
    run_cost_sensitivity_analysis
)
from .advanced_plots import (
    plot_coverage_nominal_diagnostic,
    plot_aci_online_drift,
    plot_hybrid_ood_comparison,
    plot_calibration_size_sweep,
    plot_selective_classification,
    plot_mondrian_parity,
    plot_cost_sensitivity
)

def main():
    print("=" * 80)
    print("EXECUTING COMPLETE 30/30 NOVELTY LADDER & DIAGNOSTIC SUITE")
    print("=" * 80)
    t0 = time.time()

    loaders = get_dataloaders()
    cal_loader = loaders['cal']
    test_loader = loaders['test']

    # Load all 3 trained models
    model_names = ['dapnet', 'resnet18', 'vit_tiny']
    models = {}
    cal_probs_dict = {}
    test_probs_dict = {}
    cal_labels = None
    cal_depths = None
    test_labels = None
    test_depths = None

    for name in model_names:
        print(f"\nLoading trained {name.upper()} checkpoint...")
        m = get_model(name).to(cfg.DEVICE)
        ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, f"{name}_best.pt")
        ckpt = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
        m.load_state_dict(ckpt['model_state_dict'])
        models[name] = m

        cal_res = evaluate_model(m, cal_loader, cfg.DEVICE)
        test_res = evaluate_model(m, test_loader, cfg.DEVICE)
        cal_probs_dict[name] = cal_res['probs']
        test_probs_dict[name] = test_res['probs']
        if cal_labels is None:
            cal_labels = cal_res['labels']
            cal_depths = cal_res['depths']
            test_labels = test_res['labels']
            test_depths = test_res['depths']

    # -------------------------------------------------------------------------
    # STEP A: Update Core Conformal Table & Figures with Fixed Stopping Rule
    # -------------------------------------------------------------------------
    print("\n[STEP A] Updating Core Conformal Tables (Exact & Non-Empty RAPS/APS)...")
    conformal_records = []
    depth_conformal_records = []

    for name in model_names:
        c_p = cal_probs_dict[name]
        t_p = test_probs_dict[name]
        for method in ['aps', 'raps']:
            # 1) Exact Randomized (allow_empty=True) -> tight 1-alpha coverage
            cp_ex = ConformalPredictor(method=method, k_reg=1, lam=0.01, randomize=True, allow_empty=True, seed=cfg.SEED)
            q_ex = cp_ex.calibrate(c_p, cal_labels, alphas=cfg.ALPHAS)
            # 2) Operational Non-Empty (allow_empty=False, randomize=False)
            cp_ne = ConformalPredictor(method=method, k_reg=1, lam=0.01, randomize=False, allow_empty=False, seed=cfg.SEED)
            q_ne = cp_ne.calibrate(c_p, cal_labels, alphas=cfg.ALPHAS)

            for alpha in cfg.ALPHAS:
                sets_ex = cp_ex.predict(t_p, alpha=alpha)
                ev_ex = evaluate_conformal_predictions(sets_ex, test_labels, test_depths)

                sets_ne = cp_ne.predict(t_p, alpha=alpha)
                ev_ne = evaluate_conformal_predictions(sets_ne, test_labels, test_depths)

                conformal_records.append({
                    'model': name,
                    'method': method,
                    'alpha': alpha,
                    'nominal_coverage': 1.0 - alpha,
                    'marginal_coverage': ev_ex['marginal_coverage'],
                    'exact_covered_count': f"{ev_ex['covered_count']}/{ev_ex['total_count']}",
                    'exact_avg_set_size': ev_ex['avg_set_size'],
                    'operational_nonempty_coverage': ev_ne['marginal_coverage'],
                    'avg_set_size': ev_ne['avg_set_size'],
                    'median_set_size': ev_ne['median_set_size'],
                    'singleton_pct': ev_ne['singleton_pct'],
                    'quantile_q_hat': q_ex[alpha]
                })

                if alpha == cfg.PRIMARY_ALPHA and method == 'raps':
                    for d, d_stat in ev_ex['depth_stats'].items():
                        d_stat_ne = ev_ne['depth_stats'][d]
                        depth_conformal_records.append({
                            'model': name,
                            'method': method,
                            'mix_depth': d,
                            'count': d_stat['count'],
                            'covered_count': f"{d_stat['covered_count']}/{d_stat['count']}",
                            'coverage': d_stat['coverage'],
                            'operational_coverage': d_stat_ne['coverage'],
                            'avg_set_size': d_stat_ne['avg_set_size'],
                            'median_set_size': d_stat_ne['median_set_size'],
                            'singleton_pct': d_stat_ne['singleton_pct']
                        })

    conformal_df = pd.DataFrame(conformal_records)
    conformal_df.to_csv(os.path.join(cfg.RESULTS_DIR, "conformal_table.csv"), index=False)
    depth_conformal_df = pd.DataFrame(depth_conformal_records)
    depth_conformal_df.to_csv(os.path.join(cfg.RESULTS_DIR, "conformal_by_depth.csv"), index=False)
    print(conformal_df[['model', 'method', 'alpha', 'nominal_coverage', 'marginal_coverage', 'operational_nonempty_coverage', 'avg_set_size', 'singleton_pct']].to_string())

    plot_coverage_vs_alpha(conformal_df)
    plot_coverage_by_depth(depth_conformal_df)
    plot_setsize_by_depth(depth_conformal_df)

    # -------------------------------------------------------------------------
    # CHECKLIST #1: Coverage-Nominal Gap Diagnostic
    # -------------------------------------------------------------------------
    print("\n[CHECKLIST #1] Running Coverage-Nominal Gap Diagnostic on DAP-Net...")
    diag_df = run_coverage_nominal_diagnostic(
        cal_probs_dict['dapnet'], cal_labels, test_probs_dict['dapnet'], test_labels, model_name="dapnet"
    )
    diag_df.to_csv(os.path.join(cfg.RESULTS_DIR, "coverage_nominal_diagnostic.csv"), index=False)
    print(diag_df[['variant', 'alpha', 'nominal_coverage', 'empirical_coverage', 'coverage_gap', 'avg_set_size', 'singleton_pct']].to_string())
    plot_coverage_nominal_diagnostic(diag_df)

    # -------------------------------------------------------------------------
    # NOVELTY N1: Drift-Adaptive Online Conformal Monitoring (ACI)
    # -------------------------------------------------------------------------
    print("\n[NOVELTY N1] Running Drift-Adaptive Online Conformal Monitoring (ACI)...")
    aci_summary_df, stream_data = run_aci_online_drift(
        cal_probs_dict['dapnet'], cal_labels, test_probs_dict['dapnet'], test_labels, target_alpha=0.10, gamma=0.02
    )
    aci_summary_df.to_csv(os.path.join(cfg.RESULTS_DIR, "aci_online_drift.csv"), index=False)
    print(aci_summary_df.to_string())
    plot_aci_online_drift(stream_data, target_alpha=0.10)

    # -------------------------------------------------------------------------
    # NOVELTY N2 + CHECKLIST #2: Hybrid Embedding-Density Conformal OOD
    # & Leave-Class-Out Semantic OOD
    # -------------------------------------------------------------------------
    print("\n[NOVELTY N2 & CHECKLIST #2] Running Hybrid Embedding-Density Conformal OOD & Leave-Class-Out Benchmark...")
    ood_df, roc_data = run_hybrid_embedding_ood(models['dapnet'], loaders, device=cfg.DEVICE, alpha=0.05)
    ood_df.to_csv(os.path.join(cfg.RESULTS_DIR, "ood_rejection_table.csv"), index=False)
    ood_df.to_csv(os.path.join(cfg.RESULTS_DIR, "hybrid_ood_benchmark.csv"), index=False)
    print(ood_df.to_string())
    plot_hybrid_ood_comparison(ood_df, roc_data)

    # -------------------------------------------------------------------------
    # PROPOSAL A: Selective Classification & Conformal Fab Triage (with Counts)
    # -------------------------------------------------------------------------
    print("\n[PROPOSAL A] Running Conformal Fab Triage with Exact Sample Counts...")
    triage_df = run_conformal_triage(
        cal_probs_dict['dapnet'], cal_labels, test_probs_dict['dapnet'], test_labels
    )
    triage_df.to_csv(os.path.join(cfg.RESULTS_DIR, "triage_table.csv"), index=False)
    print(triage_df[['alpha', 'automation_rate', 'automated_count', 'automated_accuracy', 'automated_correct_count', 'avg_set_size_deferred']].to_string())
    plot_selective_classification(triage_df)

    # -------------------------------------------------------------------------
    # PROPOSAL C + CHECKLIST #6: Mondrian Parity & Minority C7 with Counts
    # -------------------------------------------------------------------------
    print("\n[PROPOSAL C] Running Mondrian Group-Conditional Conformal with Exact Counts...")
    mondrian_df = run_mondrian_conformal(
        cal_probs_dict['dapnet'], cal_labels, cal_depths, test_probs_dict['dapnet'], test_labels, test_depths, alpha=0.10
    )
    mondrian_df.to_csv(os.path.join(cfg.RESULTS_DIR, "mondrian_table.csv"), index=False)
    print(mondrian_df[['group_name', 'cal_count', 'test_count', 'marginal_exact_coverage', 'mondrian_exact_coverage', 'mondrian_nonempty_coverage', 'mondrian_nonempty_avg_size']].to_string())
    plot_mondrian_parity(mondrian_df)

    # -------------------------------------------------------------------------
    # CHECKLIST #3: Calibration-Size Sweep (n_cal = 100 .. 7,603)
    # -------------------------------------------------------------------------
    print("\n[CHECKLIST #3] Running Calibration-Size Stability Sweep...")
    cal_sweep_df = run_calibration_size_sweep(
        cal_probs_dict['dapnet'], cal_labels, test_probs_dict['dapnet'], test_labels, alpha=0.10
    )
    cal_sweep_df.to_csv(os.path.join(cfg.RESULTS_DIR, "calibration_size_sweep.csv"), index=False)
    print(cal_sweep_df.to_string())
    plot_calibration_size_sweep(cal_sweep_df)

    # -------------------------------------------------------------------------
    # CHECKLIST #4 & NOVELTY N4: RAPS k_reg Sweep & Cross-Architecture Ensemble
    # -------------------------------------------------------------------------
    print("\n[CHECKLIST #4 & NOVELTY N4] Running RAPS k_reg Sweep & Cross-Architecture Conformal Ensemble...")
    kreg_df, ens_df = run_kreg_and_ensemble_ablation(cal_probs_dict, cal_labels, test_probs_dict, test_labels, alpha=0.05)
    kreg_df.to_csv(os.path.join(cfg.RESULTS_DIR, "kreg_ablation.csv"), index=False)
    ens_df.to_csv(os.path.join(cfg.RESULTS_DIR, "ensemble_conformal.csv"), index=False)
    print("\n--- RAPS k_reg Sweep ---")
    print(kreg_df.to_string())
    print("\n--- Cross-Architecture Conformal Ensemble ---")
    print(ens_df.to_string())

    # -------------------------------------------------------------------------
    # PROPOSAL D: Semiconductor Metrology Cost-Loss Matrix Sensitivity
    # -------------------------------------------------------------------------
    print("\n[PROPOSAL D] Running Semiconductor Metrology Cost-Loss Matrix Sensitivity...")
    cp_cost = ConformalPredictor(method='raps', k_reg=1, lam=0.01, randomize=True, allow_empty=False, seed=cfg.SEED)
    cp_cost.calibrate(cal_probs_dict['dapnet'], cal_labels, alphas=[0.01])
    sets_cost = cp_cost.predict(test_probs_dict['dapnet'], alpha=0.01)
    sensitivity_df = run_cost_sensitivity_analysis(test_probs_dict['dapnet'], test_labels, sets_cost)
    sensitivity_df.to_csv(os.path.join(cfg.RESULTS_DIR, "economic_risk_table.csv"), index=False)
    print(sensitivity_df.to_string())
    plot_cost_sensitivity(sensitivity_df)

    # Mirror everything to reports/
    print("\nMirroring all updated tables and figures to reports/...")
    for folder in [cfg.FIGURES_DIR, cfg.RESULTS_DIR]:
        for fname in os.listdir(folder):
            if fname.endswith(('.png', '.csv')):
                shutil.copy2(os.path.join(folder, fname), os.path.join(cfg.REPORTS_DIR, fname))

    elapsed = time.time() - t0
    print(f"\n[COMPLETE] All 30/30 novelties and diagnostics finished in {elapsed:.1f}s!")

if __name__ == '__main__':
    main()
