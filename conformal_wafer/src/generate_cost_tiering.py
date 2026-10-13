import pandas as pd
import hashlib

rows = []
for c_within in [200, 500, 1000, 2000, 5000, 10000]:
    cost_argmax = 82 * 10000 + 22 * c_within + 7 * 200
    cost_conformal = 42 * 10000 + 14 * c_within + 414 * 50
    diff = cost_argmax - cost_conformal
    pct = (diff / cost_argmax) * 100
    rows.append({
        'within_depth_cost': f"${c_within:,}",
        'cross_depth_cost': "$10,000",
        'argmax_cross_depth_escapes': 82,
        'argmax_within_depth_confusions': 22,
        'argmax_total_loss': f"${cost_argmax:,}",
        'conformal_cross_depth_escapes': 42,
        'conformal_within_depth_escapes': 14,
        'conformal_fa_cost': "$20,700",
        'conformal_total_loss': f"${cost_conformal:,}",
        'dollar_reduction': f"${diff:,}",
        'loss_reduction_pct': f"{pct:.2f}%"
    })

df_tier = pd.DataFrame(rows)
df_tier.to_csv('conformal_wafer/results/cost_tiering_sensitivity.csv', index=False)
df_tier.to_csv('conformal_wafer/reports/cost_tiering_sensitivity.csv', index=False)
with open('conformal_wafer/results/cost_tiering_sensitivity.csv', 'rb') as f:
    h = hashlib.sha256(f.read()).hexdigest()
print(f"cost_tiering_sensitivity.csv SHA-256: {h}")
