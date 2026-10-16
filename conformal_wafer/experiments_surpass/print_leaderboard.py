import json

with open('conformal_wafer/experiments_surpass/trials_registry.json') as f:
    reg = json.load(f)

print(f"Total Trials Logged: {len(reg)}")
print("-" * 75)
print(f"{'Trial':<6} | {'Architecture':<22} | {'Params':<10} | {'Test Errs':<9} | {'Top-1 Acc':<9} | {'Delta'}")
print("-" * 75)
for r in reg:
    tid = str(r['trial_id'])
    name = r['arch_name'][:22]
    params = f"{r['params']:,}"
    errs = f"{r['test_err']} / 7603"
    acc = f"{r['test_acc']:.3f}%"
    delta = f"{r['delta_vs_sota']:+d}"
    print(f"{tid:<6} | {name:<22} | {params:<10} | {errs:<9} | {acc:<9} | {delta}")
print("-" * 75)
