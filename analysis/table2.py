"""Table II and the accuracy-to-power analysis of Sec. VI, from results/_pred_dump.npz and the measured powers.
Per predictor: ATop-1 (mean and sample std over seeds), mean power-capture ratio, and the beam-selection loss (median,
mean and std, share above 3 dB, 95th percentile; tail statistics per seed, averaged over seeds). Then the least-squares
slope across the nine predictors, the predictor pairs ordered identically by accuracy and by received power, and the
slopes refitted on subsets (radar-only arms, Doppler-resolved radar arms, the ninety seed-level points, Scenario 31).
Writes results/analysis/table2.json and prints the rows."""
import json
import numpy as np
from paths import OUT, result, output_or_reference, pow_test

z = np.load(result("_pred_dump.npz")); OPT = z["OPT"].astype(int); POW = pow_test()
BEST = np.take_along_axis(POW, OPT[..., None], -1)[..., 0]
ARMS = ["RA only", "DA only", "RD only", "RA+DA", "RA+RD", "RA+DA+RD", "RA+RD+KD", "Vision-only", "Teacher"]
rows, pts = {}, []
for arm in ARMS:
    P = [z[k].astype(int) for k in z.files if k.startswith(arm + "::")]
    cap = [np.take_along_axis(POW, p[..., None], -1)[..., 0] / BEST for p in P]
    L = [-10 * np.log10(np.clip(c, 1e-6, None)) for c in cap]
    acc = [100 * (p == OPT).mean() for p in P]
    rows[arm] = dict(seeds=len(P), atop1=float(np.mean(acc)), atop1_sd=float(np.std(acc, ddof=1)),
                     capture=float(np.mean([c.mean() for c in cap])), median_db=float(np.mean([np.median(l) for l in L])),
                     mean_db=float(np.mean([l.mean() for l in L])), mean_db_sd=float(np.std([l.mean() for l in L], ddof=1)),
                     gt3_pct=float(np.mean([100 * (l > 3).mean() for l in L])), p95_db=float(np.mean([np.percentile(l, 95) for l in L])))
    pts += [(a, l.mean()) for a, l in zip(acc, L)]
acc9 = np.array([rows[a]["atop1"] for a in ARMS]); loss9 = np.array([rows[a]["mean_db"] for a in ARMS])
slope, icpt = np.polyfit(acc9, loss9, 1)
r2 = np.corrcoef(acc9, loss9)[0, 1] ** 2
agree = sum(1 for i in range(9) for j in range(i + 1, 9) if (acc9[i] - acc9[j]) * (loss9[i] - loss9[j]) < 0)
s31 = json.load(open(output_or_reference("_s31fix_test_summary.json")))
A20, L20 = s31["A7_RA"] + s31["A7_RD"], s31["L7_RA"] + s31["L7_RD"]
two = (np.mean(s31["L7_RD"]) - np.mean(s31["L7_RA"])) / (np.mean(s31["A7_RD"]) - np.mean(s31["A7_RA"]))
fit = lambda x, y: (float(np.polyfit(x, y, 1)[0]), float(np.corrcoef(x, y)[0, 1] ** 2))   # slope, R^2
subsets = {"radar-only arms": fit(acc9[:7], loss9[:7]), "Doppler-resolved radar arms": fit(acc9[1:7], loss9[1:7]),
           "ninety seed-level points": fit([a for a, _ in pts], [l for _, l in pts]),
           "Scenario 31, two arms": (float(two), None), "Scenario 31, twenty seed-level points": fit(A20, L20)}
out = dict(rows=rows, slope_db_per_pp=float(slope), intercept=float(icpt), r2=float(r2), pairs_ordered_identically=int(agree),
           spread_pp=float(acc9.max() - acc9.min()), spread_db=float(loss9.max() - loss9.min()),
           subset_slopes={k: v[0] for k, v in subsets.items()}, subset_r2={k: v[1] for k, v in subsets.items() if v[1] is not None})
json.dump(out, open(OUT / "table2.json", "w"), indent=1)
print(f"{'predictor':12s} ATop-1         capture median  mean dB      >3 dB  p95")
for a in ARMS:
    r = rows[a]
    print(f"{a:12s} {r['atop1']:5.1f} ± {r['atop1_sd']:3.1f}  {r['capture']:.3f}  {r['median_db']:.2f}   {r['mean_db']:.2f} ± {r['mean_db_sd']:.2f}  {r['gt3_pct']:4.1f}  {r['p95_db']:.2f}")
print(f"slope {slope:.4f} dB/pp (R2 {r2:.3f}); {agree} of 36 pairs ordered identically; spread {out['spread_pp']:.1f} pp, {out['spread_db']:.2f} dB")
print("subset slopes (R2): " + ", ".join(f"{k} {v[0]:.3f}" + (f" ({v[1]:.2f})" if v[1] is not None else "") for k, v in subsets.items()))
