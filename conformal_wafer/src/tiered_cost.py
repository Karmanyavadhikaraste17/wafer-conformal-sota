argmax_within, argmax_cross = 22, 82
conf_within, conf_cross = 14, 42

print("Within-Depth Compound Error Tiering Sensitivity Analysis:")
print("-" * 95)
print(f"{'Tier C_within':<15} | {'Baseline Argmax Loss':<22} | {'Conformal Risk Loss':<20} | {'Dollar Savings':<16} | {'% Reduction':<12}")
print("-" * 95)

for c_within in [200, 500, 1000, 2000, 5000, 10000]:
    cost_argmax = argmax_cross * 10000 + argmax_within * c_within + 7 * 200
    cost_conformal = conf_cross * 10000 + conf_within * c_within + 414 * 50
    diff = cost_argmax - cost_conformal
    pct = (diff / cost_argmax) * 100
    print(f"${c_within:<14,d} | ${cost_argmax:<21,d} | ${cost_conformal:<19,d} | ${diff:<15,d} | {pct:>6.2f}%")
print("-" * 95)
