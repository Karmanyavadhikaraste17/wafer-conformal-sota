import numpy as np, pandas as pd

rows = []
for c_within in [200, 500, 1000, 2000, 5000, 10000]:
    cost_bm = 82 * 10000 + 22 * c_within + 7 * 200
    for p_miss in [1.0, 0.75, 0.5, 0.25, 0.0]:
        escapes_cross = 38 + 4 * p_miss
        escapes_within = 14
        fa_reviewed = 414 + 4 * (1.0 - p_miss)
        cost_cp = escapes_cross * 10000 + escapes_within * c_within + fa_reviewed * 50
        diff = cost_bm - cost_cp
        pct = (diff / cost_bm) * 100
        rows.append({
            'within_depth_cost': f"${c_within:,}",
            'human_miss_prob': f"{p_miss:.2f}",
            'baseline_loss': f"${cost_bm:,}",
            'conformal_loss': f"${int(cost_cp):,}",
            'net_savings': f"${int(diff):,}",
            'reduction_pct': f"{pct:.2f}%"
        })

df = pd.DataFrame(rows)
df.to_csv('conformal_wafer/results/cost_tiering_pmiss_matrix.csv', index=False)
df.to_csv('conformal_wafer/reports/cost_tiering_pmiss_matrix.csv', index=False)
print("Generated cost_tiering_pmiss_matrix.csv:")
print(df.to_string())
