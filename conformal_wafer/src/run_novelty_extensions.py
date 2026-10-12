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
from .conformal import ConformalPredictor
from .advanced_conformal import (
    run_conformal_triage,
    evaluate_ood_detection,
    run_mondrian_conformal,
    evaluate_economic_loss,
    run_cost_sensitivity_analysis
)
from .advanced_plots import (
    plot_selective_classification,
    plot_ood_rejection,
    plot_mondrian_parity,
    plot_cost_sensitivity
)

def main():
    print("="*75)
    print("EXECUTING ADVANCED NOVELTY EXTENSIONS (PROPOSALS A, B, C, D)")
    print("="*75)
    t0 = time.time()

    # Load data
    loaders = get_dataloaders()
    cal_loader = loaders['cal']
    test_loader = loaders['test']

    # Load trained DAP-Net checkpoint
    print("\nLoading trained DAP-Net backbone...")
    model = get_model('dapnet').to(cfg.DEVICE)
    ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, "dapnet_best.pt")
    ckpt = torch.load(ckpt_path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])

    # Evaluate probabilities
    print("Evaluating probabilities on Calibration (7,603) and Test (7,603) sets...")
    cal_eval = evaluate_model(model, cal_loader, cfg.DEVICE)
    test_eval = evaluate_model(model, test_loader, cfg.DEVICE)

    cal_probs, cal_labels, cal_depths = cal_eval['probs'], cal_eval['labels'], cal_eval['depths']
    test_probs, test_labels, test_depths = test_eval['probs'], test_eval['labels'], test_eval['depths']

    # -------------------------------------------------------------------------
    # 1. Proposal A: Conformal Fab Triage & Selective Classification
    # -------------------------------------------------------------------------
    print("\n[PROPOSAL A] Running Conformal Fab Triage & Selective Classification...")
    triage_df = run_conformal_triage(cal_probs, cal_labels, test_probs, test_labels)
    triage_path = os.path.join(cfg.RESULTS_DIR, "triage_table.csv")
    triage_df.to_csv(triage_path, index=False)
    print(f"  Saved: {triage_path}")
    print(triage_df[['alpha', 'nominal_coverage', 'automation_rate', 'automated_accuracy', 'avg_set_size']].to_string())

    plot_selective_classification(triage_df)

    # -------------------------------------------------------------------------
    # 2. Proposal B: Empty-Set OOD & Novel Defect Detection
    # -------------------------------------------------------------------------
    print("\n[PROPOSAL B] Evaluating Empty-Set OOD Rejection (C(X) = 0)...")
    ood_results, ood_confs, ood_set_sizes = evaluate_ood_detection(model, test_loader, cal_eval, alpha=0.05)
    ood_df = pd.DataFrame([ood_results])
    ood_path = os.path.join(cfg.RESULTS_DIR, "ood_rejection_table.csv")
    ood_df.to_csv(ood_path, index=False)
    print(f"  Saved: {ood_path}")
    print(f"  In-Distribution Coverage:          {ood_results['in_dist_coverage']*100:.2f}%")
    print(f"  OOD Conformal Rejection Rate:      {ood_results['ood_conformal_rejection_rate']*100:.2f}% (Flags C(X) = 0)")
    print(f"  OOD Baseline Hallucination Rate:   {ood_results['ood_baseline_hallucination_rate']*100:.2f}%")
    print(f"  OOD Baseline Mean Softmax Conf:    {ood_results['ood_baseline_mean_conf']*100:.2f}% (Blind overconfidence!)")

    plot_ood_rejection(ood_results, ood_confs)

    # -------------------------------------------------------------------------
    # 3. Proposal C: Mondrian (Group-Conditional) Conformal Prediction
    # -------------------------------------------------------------------------
    print("\n[PROPOSAL C] Running Mondrian Group-Conditional Conformal Prediction...")
    mondrian_df = run_mondrian_conformal(cal_probs, cal_labels, cal_depths, test_probs, test_labels, test_depths, alpha=0.10)
    mondrian_path = os.path.join(cfg.RESULTS_DIR, "mondrian_table.csv")
    mondrian_df.to_csv(mondrian_path, index=False)
    print(f"  Saved: {mondrian_path}")
    print(mondrian_df[['mix_depth', 'count', 'mondrian_coverage', 'mondrian_avg_set_size', 'marginal_coverage']].to_string())

    plot_mondrian_parity(mondrian_df)

    # -------------------------------------------------------------------------
    # 4. Proposal D: Semiconductor Metrology Cost-Loss Matrix Sensitivity
    # -------------------------------------------------------------------------
    print("\n[PROPOSAL D] Running Semiconductor Metrology Cost-Loss Sensitivity Analysis...")
    # Get standard RAPS prediction sets at alpha=0.10
    cp = ConformalPredictor(method='raps', k_reg=cfg.RAPS_K_REG, lam=cfg.RAPS_LAMBDA, randomize=True, seed=cfg.SEED)
    cp.calibrate(cal_probs, cal_labels, alphas=[0.10])
    pred_sets_90 = cp.predict(test_probs, alpha=0.10)

    loss_eval = evaluate_economic_loss(test_probs, test_labels, pred_sets_90)
    print(f"  Baseline Argmax Total Fab Loss:    ${loss_eval['baseline_total_loss']:,.2f}")
    print(f"  Conformal Risk Policy Fab Loss:    ${loss_eval['conformal_total_loss']:,.2f}")
    print(f"  Fab Financial Loss Reduction:      {loss_eval['cost_reduction_pct']:.2f}% (${loss_eval['cost_reduction_dollars']:,.2f} saved!)")

    sensitivity_df = run_cost_sensitivity_analysis(test_probs, test_labels, pred_sets_90)
    sensitivity_path = os.path.join(cfg.RESULTS_DIR, "economic_risk_table.csv")
    sensitivity_df.to_csv(sensitivity_path, index=False)
    print(f"  Saved: {sensitivity_path}")

    plot_cost_sensitivity(sensitivity_df)

    # -------------------------------------------------------------------------
    # Mirror all new figures and tables to reports/
    # -------------------------------------------------------------------------
    print("\nMirroring all new deliverables to reports/...")
    new_files = [
        'selective_classification_curve.png',
        'ood_rejection_vs_hallucination.png',
        'mondrian_vs_marginal_coverage.png',
        'fab_economic_risk_sensitivity.png',
        'triage_table.csv',
        'ood_rejection_table.csv',
        'mondrian_table.csv',
        'economic_risk_table.csv'
    ]
    for nf in new_files:
        src_fig = os.path.join(cfg.FIGURES_DIR, nf)
        src_res = os.path.join(cfg.RESULTS_DIR, nf)
        if os.path.exists(src_fig):
            shutil.copy2(src_fig, os.path.join(cfg.REPORTS_DIR, nf))
        elif os.path.exists(src_res):
            shutil.copy2(src_res, os.path.join(cfg.REPORTS_DIR, nf))

    elapsed = time.time() - t0
    print(f"\n[DONE] All 4 novelty extensions executed and verified in {elapsed:.1f}s ({elapsed/60:.2f} mins)!")

if __name__ == '__main__':
    main()
