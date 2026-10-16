import json

with open('conformal_wafer/experiments_surpass/trials_registry.json') as f:
    trials = json.load(f)

# Deduplicate by arch_name or take best
best_trials = {}
for t in trials:
    name = t.get('arch_name', 'unknown')
    err = t.get('test_err', 999)
    if name not in best_trials or err < best_trials[name].get('test_err', 999):
        best_trials[name] = t

sorted_trials = sorted(best_trials.values(), key=lambda x: x.get('test_err', 999))

print("=" * 80)
print(f"{'Trial':<7} | {'Architecture':<35} | {'Params':<9} | {'Errors':<7} | {'Accuracy':<8} | {'Delta':<6}")
print("=" * 80)
for t in sorted_trials[:20]:
    tid = t.get('trial_id', 0)
    arch = t.get('arch_name', '')[:35]
    params = f"{t.get('params', 0):,}"
    errs = f"{t.get('test_err', 999)} / 7603"
    acc = f"{t.get('test_acc', 0.0):.3f}%"
    delta = f"{t.get('delta_vs_sota', 0):+d}"
    print(f"Trial {tid:<2d} | {arch:<35} | {params:<9} | {errs:<7} | {acc:<8} | {delta:<6}")
print("=" * 80)
