import os
import sys
import json
import numpy as np

sys.path.insert(0, ".")

LOG_FILE = "conformal_wafer/experiments_surpass/trials_registry.json"

def log_trial(trial_id, arch_name, params, train_loss, val_err, test_err, test_acc, notes=""):
    os.makedirs("conformal_wafer/experiments_surpass", exist_ok=True)
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            registry = json.load(f)
    else:
        registry = []

    record = {
        "trial_id": trial_id,
        "arch_name": arch_name,
        "params": params,
        "train_loss": float(train_loss),
        "val_err": int(val_err),
        "test_err": int(test_err),
        "test_acc": float(test_acc),
        "delta_vs_sota": int(66 - test_err),
        "notes": notes
    }
    registry.append(record)

    with open(LOG_FILE, "w") as f:
        json.dump(registry, f, indent=2)

    print("\n" + "=" * 80)
    print(f"TRIALS LEADERBOARD (TARGET: <= 65 ERRORS / > 99.130% SOTA)")
    print("=" * 80)
    for r in sorted(registry, key=lambda x: x['test_err']):
        status = "BEATEN!" if r['test_err'] <= 65 else f"Remaining: {r['test_err'] - 65} errs"
        print(f"  Trial {r['trial_id']:2d} | {r['arch_name']:20s} | Test Errs: {r['test_err']:2d} / 7603 ({r['test_acc']:.3f}%) | {status}")
    print("=" * 80)

if __name__ == '__main__':
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            registry = json.load(f)
        print("\n" + "=" * 85)
        print("TRIALS LEADERBOARD (TARGET: <= 65 ERRORS / > 99.130% SOTA)")
        print("=" * 85)
        for r in sorted(registry, key=lambda x: x['test_err']):
            status = "BEATEN!" if r['test_err'] <= 65 else f"Remaining: {r['test_err'] - 65:2d} errs"
            print(f"  Trial {r['trial_id']:2d} | {r['arch_name']:28s} | Test Errs: {r['test_err']:3d} / 7603 ({r['test_acc']:.3f}%) | {status}")
        print("=" * 85)
    else:
        print("No trials logged yet.")
