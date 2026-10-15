"""Least-squares line of paper Fig. 5 over the nine predictors, its band, and the interval of its slope (supplement S3).

Fit: the nine predictors' mean top-1 accuracy (pp) and mean beam-selection loss (dB) over all seven slots and ten seeds.
Band (the shaded band of Fig. 5): the pointwise 95% confidence band of the least-squares line for this test set,
t(7) * s * sqrt(1/9 + (x - xbar)^2 / Sxx), reported at the centroid, at the outermost predictors and at the plot edges
(36.4% and 46.9%).
Slope interval: the cluster bootstrap of supplement S3. Each replicate draws the 28 test passes with replacement and,
separately for each predictor, as many seeds as it has with replacement; it recomputes the nine means and refits the line.
Reports the 2.5th and 97.5th percentiles of the slope. For comparison it also reports the interval from fitting once and
reading the residual standard error (narrower, since it ignores the passes the nine predictors share), and the resampled
band of the fitted line, which is wide because a draw of harder or easier passes moves all nine predictors together.
Inputs: the prediction dump (per-window top-1 beams of the nine predictors over ten seeds, the optimal beams and the
pass of each window) and the measured powers of the test windows.
Usage: python analysis/slope_interval.py [--dump results/_pred_dump.npz] [--pow POW.npz] [--reps 20000] [--seed 42]
(--pow defaults to the powers built from dataset/scenario9; the output is results/analysis/_slope_interval.json)"""
import argparse, json
from paths import OUT, result, pow_test
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--dump", default=str(result("_pred_dump.npz"))); ap.add_argument("--pow", default=None)
ap.add_argument("--reps", type=int, default=20000); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default=str(OUT / "_slope_interval.json"))
a = ap.parse_args()
z = np.load(a.dump, allow_pickle=True); POW = (np.load(a.pow)["POW"] if a.pow else pow_test()).astype(np.float64)
OPT, PASS = z["OPT"], z["PASS"]
BEST = np.take_along_axis(POW, OPT[..., None], -1)[..., 0]
arms = sorted({k.split("::")[0] for k in z.files if "::" in k})
passes = np.unique(PASS); P = len(passes); pidx = np.searchsorted(passes, PASS)
nwin = np.bincount(pidx, minlength=P).astype(float)
SA, SL = [], []                                   # per arm: (seeds, passes) sums of hits and losses
for arm in arms:
    keys = [k for k in z.files if k.startswith(arm + "::")]
    sa, sl = [], []
    for k in keys:
        pr = z[k]
        hit = (pr == OPT).mean(1)
        loss = (-10 * np.log10(np.clip(np.take_along_axis(POW, pr[..., None], -1)[..., 0] / BEST, 1e-6, None))).mean(1)
        sa.append(np.bincount(pidx, weights=hit, minlength=P)); sl.append(np.bincount(pidx, weights=loss, minlength=P))
    SA.append(np.array(sa)); SL.append(np.array(sl))
nseed = [s.shape[0] for s in SA]
acc0 = np.array([100 * s.sum() / (s.shape[0] * nwin.sum()) for s in SA])
los0 = np.array([s.sum() / (s.shape[0] * nwin.sum()) for s in SL])
b0, c0 = np.polyfit(acc0, los0, 1)
rng = np.random.default_rng(a.seed)
slopes = np.empty(a.reps); xs = np.array([acc0.mean(), acc0.min(), acc0.max()]); fit = np.empty((a.reps, 3))
for r in range(a.reps):
    k = np.bincount(rng.integers(0, P, P), minlength=P)            # pass multiplicities
    den = k @ nwin
    acc = np.empty(len(arms)); los = np.empty(len(arms))
    for i in range(len(arms)):
        m = np.bincount(rng.integers(0, nseed[i], nseed[i]), minlength=nseed[i])
        acc[i] = 100 * (m @ SA[i] @ k) / (nseed[i] * den); los[i] = (m @ SL[i] @ k) / (nseed[i] * den)
    b, c = np.polyfit(acc, los, 1); slopes[r] = b; fit[r] = b * xs + c
lo, hi = np.percentile(slopes, [2.5, 97.5])
rband = (np.percentile(fit, 97.5, 0) - np.percentile(fit, 2.5, 0)) / 2
res_lin = stats.linregress(acc0, los0); n = len(arms); t = stats.t.ppf(0.975, n - 2)
s = np.sqrt(((los0 - (b0 * acc0 + c0)) ** 2).sum() / (n - 2)); Sxx = ((acc0 - acc0.mean()) ** 2).sum()
lsq = lambda x: float(t * s * np.sqrt(1 / n + (x - acc0.mean()) ** 2 / Sxx))     # Fig. 5's shaded band, half-width
out = dict(arms=arms, acc=acc0.tolist(), loss=los0.tolist(), slope=b0, intercept=c0, r2=res_lin.rvalue ** 2,
           band_halfwidth=dict(centroid=lsq(acc0.mean()), outermost=[lsq(acc0.min()), lsq(acc0.max())],
                               plot_edges=[lsq(36.4), lsq(46.9)]),
           slope_interval=[lo, hi], residual_se_interval=[b0 - t * res_lin.stderr, b0 + t * res_lin.stderr],
           resampled_band_halfwidth=dict(centroid=rband[0], outermost=[rband[1], rband[2]]), reps=a.reps, seed=a.seed)
json.dump(out, open(a.out, "w"), indent=1, default=float)
bh = out["band_halfwidth"]
print(f"slope {b0:.4f} (R2 {res_lin.rvalue ** 2:.3f}); least-squares 95% band half-width {bh['centroid']:.3f} at the "
      f"centroid, {bh['plot_edges'][0]:.3f}/{bh['plot_edges'][1]:.3f} at the plot edges")
print(f"bootstrap slope interval [{lo:.3f}, {hi:.3f}]; residual-SE interval [{out['residual_se_interval'][0]:.3f}, "
      f"{out['residual_se_interval'][1]:.3f}]; resampled band half-width {rband[0]:.3f} at the centroid")
