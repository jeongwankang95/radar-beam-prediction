"""Values quoted in the text that are computed directly from the prediction dumps rather than read from a bootstrap file:
  Sec. V-B   spread of the mean loss over the five radar arms with a Doppler-resolved map;
  Sec. V-C   per-slot accuracy range of the reference student, and the persistence of the slot-t beam;
  Sec. VII-C per-slot Doppler gain at t and t+6, and (supplementary Table S14 and S12) the difference between the two
             ends with its interval: paired cluster bootstrap, 28 passes shared by both arms, seeds drawn separately per
             arm, 20,000 replicates, generator seed 0; in decibels from the prediction dump, in accuracy from the
             predictions behind the rows of Table S14 (results/_predcache.npz);
  supp. S1   Tables I and II read over t+1..t+6 instead of all seven slots;
  supp. S14  slopes of Sec. VI with one predictor left out.
Inputs: results/_pred_dump.npz, results/_predcache.npz, results/_w2pred_rdfeat.npz, results/_w2pred_rdfull.npz and the
measured powers.
Writes results/analysis/text_values.json."""
import json
import numpy as np
from paths import OUT, result, pow_test

z = np.load(result("_pred_dump.npz")); OPT = z["OPT"].astype(int); PASS = z["PASS"].astype(int)
POW = pow_test(); BEST = np.take_along_axis(POW, OPT[..., None], -1)[..., 0]
ARMS = ["RA only", "DA only", "RD only", "RA+DA", "RA+RD", "RA+DA+RD", "RA+RD+KD", "Vision-only", "Teacher"]
arm = lambda name, d=z: np.stack([d[k].astype(int) for k in d.files if k.startswith(name + "::")])
loss = lambda P: -10 * np.log10(np.clip(np.take_along_axis(POW, P[..., None], -1)[..., 0] / BEST, 1e-6, None))
out = {}

acc9 = np.array([np.mean([100 * (p == OPT).mean() for p in arm(a)]) for a in ARMS])
loss9 = np.array([np.mean([loss(p).mean() for p in arm(a)]) for a in ARMS])
out["doppler_arms_loss_spread_db"] = float(loss9[1:6].max() - loss9[1:6].min())
print(f"V-B   mean loss of the five Doppler-resolved radar arms spans {out['doppler_arms_loss_spread_db']:.2f} dB")

slot_acc = 100 * (arm("RA+RD") == OPT[None]).mean(axis=1).mean(axis=0)
pers = 100 * (OPT[:, :1] == OPT).mean(axis=0)
out["reference_slot_atop1_range"] = [float(slot_acc.min()), float(slot_acc.max())]
out["persistence_slot_atop1"] = pers.tolist()
print(f"V-C   reference student per slot: {slot_acc.min():.1f} to {slot_acc.max():.1f} %; persistence {pers[1]:.1f} % at t+1, "
      f"{pers[6]:.1f} % at t+6")

LRA, LRD = np.stack([loss(p) for p in arm("RA only")]), np.stack([loss(p) for p in arm("RA+RD")])
gd = LRA.mean(axis=(0, 1)) - LRD.mean(axis=(0, 1))
pi = np.searchsorted(np.unique(PASS), PASS); NP = pi.max() + 1; n = np.bincount(pi, minlength=NP).astype(float)
def pass_sums(L):
    s = np.zeros((L.shape[0], NP, 7))
    for j in range(L.shape[0]):
        np.add.at(s[j], pi, L[j])
    return s
def end_difference(A, D):
    """(gain at t+6) - (gain at t), gain = mean of A minus mean of D, with its paired cluster-bootstrap interval"""
    SA, SD = pass_sums(A), pass_sums(D)
    d0 = (A[:, :, 6].mean() - D[:, :, 6].mean()) - (A[:, :, 0].mean() - D[:, :, 0].mean())
    rng = np.random.default_rng(0); bs = np.empty(20000)
    for b in range(20000):
        w = np.bincount(rng.integers(0, NP, NP), minlength=NP).astype(float)
        sa = rng.integers(0, SA.shape[0], SA.shape[0]); sd = rng.integers(0, SD.shape[0], SD.shape[0])
        den = (w * n).sum()
        g = (SA[sa][:, :, [0, 6]] * w[None, :, None]).sum(1).mean(0) / den - (SD[sd][:, :, [0, 6]] * w[None, :, None]).sum(1).mean(0) / den
        bs[b] = g[1] - g[0]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(point=float(d0), lo=float(lo), hi=float(hi))
out["doppler_gain_db_slot_t_t6"] = [float(gd[0]), float(gd[6])]
e = out["doppler_gain_db_end_difference"] = end_difference(LRA, LRD)
print(f"VII-C Doppler gain {gd[0]:.3f} dB at t, {gd[6]:.3f} dB at t+6; difference {e['point']:+.3f} dB on [{e['lo']:+.3f}, {e['hi']:+.3f}]")
Z = np.load(result("_predcache.npz"))
fam = lambda f: np.stack([Z[k] for k in sorted((k for k in Z.files if k.startswith(f + "::")), key=lambda k: k.split("::")[1] + ".npz")])
CRA, CRD = (fam("ra") == Z["LABEL"][None]).astype(float), (fam("rd") == Z["LABEL"][None]).astype(float)
e = end_difference(100 * CRD, 100 * CRA)
out["doppler_gain_atop1_end_difference"] = e
print(f"      in ATop-1 (Table S14 predictions): difference {e['point']:+.2f} pp on [{e['lo']:+.2f}, {e['hi']:+.2f}]")

def reading(P, sl):
    L = [loss(p)[:, sl] for p in P]
    return (np.mean([100 * (p[:, sl] == OPT[:, sl]).mean() for p in P]), np.mean([l.mean() for l in L]),
            np.mean([np.percentile(l, 95) for l in L]), np.mean([np.median(l) for l in L]))
s7 = {a: reading(arm(a), slice(0, 7)) for a in ARMS}; s6 = {a: reading(arm(a), slice(1, 7)) for a in ARMS}
for name, dump, key in (("RA+RD + feature", "_w2pred_rdfeat.npz", "rdfeat"), ("RA+RD + feature + temporal", "_w2pred_rdfull.npz", "rdfull")):
    Ls = [loss(p) for p in arm(key, np.load(result(dump)))]   # Table I rows; accuracy as the share of zero-loss slots
    s7[name] = (np.mean([100 * (l <= 1e-6).mean() for l in Ls]), np.mean([l.mean() for l in Ls]), np.nan, np.nan)
    s6[name] = (np.mean([100 * (l[:, 1:] <= 1e-6).mean() for l in Ls]), np.mean([l[:, 1:].mean() for l in Ls]), np.nan, np.nan)
dacc = max(abs(s6[a][0] - s7[a][0]) for a in s7); dloss = max(abs(s6[a][1] - s7[a][1]) for a in s7)
dp95 = max(s6[a][2] - s7[a][2] for a in ARMS)
gap = [0.0] * 4; ks = list(s7)
for i, a in enumerate(ks):
    for b in ks[i + 1:]:
        for c in range(4):
            if not np.isnan(s7[a][c] + s7[b][c]) and (s7[a][c] - s7[b][c]) * (s6[a][c] - s6[b][c]) < 0:
                gap[c] = max(gap[c], abs(s7[a][c] - s7[b][c]))
out["six_slot_reading"] = dict(max_atop1_change_pp=float(dacc), max_mean_loss_change_db=float(dloss), max_p95_change_db=float(dp95),
                               largest_reordered_gap=dict(zip(("atop1_pp", "mean_db", "p95_db", "median_db"), map(float, gap))))
print(f"S1    over t+1..t+6: ATop-1 moves by at most {dacc:.2f} pp, mean loss by at most {dloss:.2f} dB, p95 by up to {dp95:.2f} dB; "
      f"reordered pairs differ by at most {gap[0]:.2f} pp / {max(gap[1:]):.3f} dB over seven slots")

loo = [np.polyfit(np.delete(acc9, i), np.delete(loss9, i), 1)[0] for i in range(9)]
lr2 = [np.corrcoef(np.delete(acc9, i), np.delete(loss9, i))[0, 1] ** 2 for i in range(9)]
out["leave_one_out_slope"] = dict(slope=[float(min(loo)), float(max(loo))], r2=[float(min(lr2)), float(max(lr2))])
print(f"S14   leave-one-out slope between {min(loo):.3f} and {max(loo):.3f} dB per point (R2 {min(lr2):.2f} to {max(lr2):.2f})")
json.dump(out, open(OUT / "text_values.json", "w"), indent=1)
