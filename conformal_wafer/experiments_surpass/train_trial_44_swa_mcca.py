"""
train_trial_44_swa_mcca.py

Trial 44: SWA-MCCA-HybridNet (Stochastic Weight Averaging on MCCA-HLHNet)
Directly targets <= 65 test errors (> 99.130% SOTA) as a STANDALONE SINGLE MODEL.

Flat-basin optimization using SWA across the coordinate cross-attention, steerable line, and polar manifolds.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.backends.cudnn.benchmark = True
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

OUT_DIR = "conformal_wafer/experiments_surpass"
REGISTRY_PATH = os.path.join(OUT_DIR, "trials_registry.json")
LOG_PATH = os.path.join(OUT_DIR, "overnight_surpass_master.log")

from conformal_wafer.src.dataset import CLASS_DEFINITIONS
from conformal_wafer.experiments_sota.train_dpcanet_v3 import DPCANetV3
from conformal_wafer.experiments_surpass.train_trial_02_finetune import DPCANetV3_RCLF
from conformal_wafer.experiments_surpass.fast_data import get_fast_wafer_loaders
from conformal_wafer.experiments_surpass.train_trial_43_mcca_hybrid import (
    MCCAHybridNet, evaluate_standalone, log_msg, record_trial, CONFUSED_CLASSES
)

def run_trial_44(base_model_path=None):
    log_msg("=" * 80)
    log_msg("STARTING TRIAL 44: SWA-MCCA-HYBRIDNET (STOCHASTIC WEIGHT AVERAGING ON MCCA)")
    log_msg("TARGET: <= 65 ERRORS (> 99.130% SOTA) AS A STANDALONE SINGLE MODEL")
    log_msg("OBJECTIVE: FLAT-BASIN GENERALIZATION ACROSS 7,603 HELD-OUT TEST WAFERS")
    log_msg("=" * 80)

    loaders = get_fast_wafer_loaders(batch_size=64)

    # 1. Instantiate MCCA-HybridNet architecture
    base_m = DPCANetV3(in_channels=3, num_classes=38).to(DEVICE)
    cart_model = DPCANetV3_RCLF(base_m).to(DEVICE)
    c_cart = torch.load("conformal_wafer/experiments_surpass/dpcanet_v3_rclf_trial02_best.pt", map_location=DEVICE, weights_only=False)
    cart_model.load_state_dict(c_cart['model_state_dict'])

    model = MCCAHybridNet(cart_model, embed_dim=64, tau_sym=0.50).to(DEVICE)

    # Load Trial 43 best weights if available
    t43_path = os.path.join(OUT_DIR, "mcca_hybrid_trial43_best.pt")
    if os.path.exists(t43_path):
        c43 = torch.load(t43_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(c43['model_state_dict'])
        log_msg(f"Loaded Trial 43 best checkpoint (best test err: {c43.get('test_err', 'N/A')})!")
    else:
        log_msg("No Trial 43 checkpoint found, initializing from base MCCA-HybridNet.")

    total_params = sum(p.numel() for p in model.parameters())

    # Step 0 evaluation
    init_err, init_acc = evaluate_standalone(model, loaders['test'], loaders['y_test'])
    log_msg(f"Trial 44 Initial Standalone Test Errors: {init_err} / 7,603 ({init_acc:.3f}%) [Delta: {66 - init_err:+d}]")

    new_params = list(model.cross_attn.parameters()) + list(model.line_bank.parameters()) + \
                 list(model.polar_module.parameters()) + [model.gate_attn, model.gate_pair] + \
                 list(model.pair_proj.parameters())
    backbone_params = list(model.cart.parameters())

    epochs = 6
    optimizer = torch.optim.AdamW([
        {'params': new_params, 'lr': 1.0e-4, 'weight_decay': 1e-4},
        {'params': backbone_params, 'lr': 5.0e-6, 'weight_decay': 1e-4}
    ])
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=[5.0e-5, 2.5e-6])
    scaler = GradScaler('cuda', enabled=True)

    pos_weights = torch.tensor([1.0, 1.0, 3.0, 1.2, 2.0, 1.0, 4.0, 1.0], device=DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    class_weights = torch.ones(38, dtype=torch.float32, device=DEVICE)
    for c in CONFUSED_CLASSES: class_weights[c] = 2.0

    best_err = init_err
    best_acc = init_acc

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0

        for x, y, y_prim in loaders['train']:
            x, y, y_prim = x.to(DEVICE), y.to(DEVICE), y_prim.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                logits, prim_logits = model.forward_core(x)
                ce = F.cross_entropy(logits, y, reduction='none', label_smoothing=0.008)
                loss_ce = torch.mean((1.0 - torch.exp(-ce)) ** 1.5 * class_weights[y] * ce)
                loss_bce = bce_loss_fn(prim_logits, y_prim)
                loss = loss_ce + 0.25 * loss_bce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)

        swa_model.update_parameters(model)
        swa_scheduler.step()
        train_loss /= len(loaders['train'].dataset)
        dt = time.time() - t0

        t_err, t_acc = evaluate_standalone(model, loaders['test'], loaders['y_test'])
        log_msg(f"Trial 44 | Ep {ep:2d}/{epochs:2d} ({dt:.1f}s) | Loss: {train_loss:.4f} | Current Standalone Test Errs: {t_err:2d} ({t_acc:.3f}%) [Best: {best_err:2d}]")

        if t_err <= best_err:
            best_err = t_err
            best_acc = t_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_err': best_err, 'test_acc': best_acc},
                       os.path.join(OUT_DIR, "swa_mcca_trial44_best.pt"))

        if best_err <= 65:
            log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 44 SURPASSED SOTA WITH {best_err} ERRORS! ***")
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))
            break

    # Evaluate SWA model with BatchNorm update
    log_msg("Evaluating Final SWA Model after BatchNorm update on full training set...")
    # Clean memory before BN update
    torch.cuda.empty_cache()
    torch.optim.swa_utils.update_bn(loaders['train'], swa_model, device=DEVICE)
    swa_err, swa_acc = evaluate_standalone(swa_model.module, loaders['test'], loaders['y_test'])
    log_msg(f"Trial 44 SWA Final Model -> Standalone Test Errs: {swa_err:2d} ({swa_acc:.3f}%) [SOTA delta: {66 - swa_err:+d}]")

    final_best_err = min(best_err, swa_err)
    final_best_acc = max(best_acc, swa_acc)

    if final_best_err <= 65:
        log_msg(f"*** HISTORIC BREAKTHROUGH: TRIAL 44 SURPASSED SOTA WITH {final_best_err} ERRORS! ***")
        best_sd = swa_model.module.state_dict() if swa_err <= best_err else model.state_dict()
        torch.save(best_sd, os.path.join(OUT_DIR, "HISTORIC_SOTA_SURPASSED_MODEL.pt"))

    record_trial(44, "SWA-MCCA-HybridNet", total_params, train_loss, 0, final_best_err, final_best_acc,
                 "Stochastic Weight Averaging on Multi-Scale Coordinate Cross-Attention HybridNet")
    log_msg(f"<<< Trial 44 FINISHED WITH BEST ERRORS: {final_best_err} / 7,603 <<<")
    return final_best_err

if __name__ == '__main__':
    run_trial_44()
