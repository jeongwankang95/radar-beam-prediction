"""Supplementary S3: alternative intervals for the two central contrasts (C2: RA+RD vs RA only;
C3: RA+RD+KD vs RA+RD), all on the 7-slot dB loss.
  (1) the paper's percentile cluster bootstrap (passes, seeds unpaired), for reference;
  (2) BCa for the same bootstrap, acceleration from the leave-one-pass-out jackknife;
  (3) paired-seed bootstrap (passes, seeds drawn jointly for both arms), for C3 where the arms share seeds;
  (4) pass-level sign-flip permutation test on the seed-averaged per-pass difference (window-weighted).
Also: per-optimal-beam-band Doppler gain (RA+RD vs RA only) with the paper's bootstrap."""
import numpy as np, collections, re, json
from paths import OUT, result, pow_test
from scipy.stats import norm
z = np.load(result("_pred_dump.npz")); OPT = z["OPT"].astype(int); PASS = z["PASS"].astype(int)
POW = pow_test(); BEST = np.take_along_axis(POW, OPT[..., None], -1)[..., 0]
loss = lambda P: -10 * np.log10(np.clip(np.take_along_axis(POW, P[..., None], -1)[..., 0] / BEST, 1e-6, None))
def arm(name):
    d = collections.OrderedDict((k.split("::", 1)[1], z[k].astype(int)) for k in z.files if k.startswith(name + "::"))
    return d
def seed_of(run):
    m = re.search(r"_s(\d+)", run); return int(m.group(1)) if m else None
passes = np.unique(PASS); idx = [np.where(PASS == q)[0] for q in passes]; nP = len(passes)
rng = np.random.default_rng(0); B = 20000
out = {}
def contrast(new, ref, label, paired=False):
    A, R = arm(new), arm(ref)
    LA = np.stack([loss(P) for P in A.values()]).mean(-1); LR = np.stack([loss(P) for P in R.values()]).mean(-1)   # seeds x windows
    point = LR.mean() - LA.mean()                     # dB recovered by `new` (positive: new better)
    # (1) percentile, unpaired seeds
    bs = np.empty(B)
    for b in range(B):
        w = np.concatenate([idx[i] for i in rng.integers(0, nP, nP)])
        sa = rng.integers(0, LA.shape[0], LA.shape[0]); sr = rng.integers(0, LR.shape[0], LR.shape[0])
        bs[b] = LR[sr][:, w].mean() - LA[sa][:, w].mean()
    pct = np.percentile(bs, [2.5, 97.5])
    # (2) BCa: bias from the bootstrap, acceleration from leave-one-pass-out jackknife
    z0 = norm.ppf(np.clip((bs < point).mean(), 1e-6, 1 - 1e-6))
    jk = np.array([LR[:, np.concatenate([idx[j] for j in range(nP) if j != i])].mean() - LA[:, np.concatenate([idx[j] for j in range(nP) if j != i])].mean() for i in range(nP)])
    dj = jk.mean() - jk; acc = (dj ** 3).sum() / (6 * ((dj ** 2).sum()) ** 1.5)
    al = [norm.cdf(z0 + (z0 + norm.ppf(p)) / (1 - acc * (z0 + norm.ppf(p)))) for p in (0.025, 0.975)]
    bca = np.percentile(bs, [100 * al[0], 100 * al[1]])
    res = dict(point=float(point), percentile=[float(v) for v in pct], bca=[float(v) for v in bca], z0=float(z0), acc=float(acc))
    # (3) paired seeds
    if paired:
        sA = {seed_of(k): i for i, k in enumerate(A)}; sR = {seed_of(k): i for i, k in enumerate(R)}
        common = sorted(set(sA) & set(sR)); ia = [sA[s] for s in common]; ir = [sR[s] for s in common]
        Dm = LR[ir] - LA[ia]                                   # common seeds x windows
        pb = np.empty(B)
        for b in range(B):
            w = np.concatenate([idx[i] for i in rng.integers(0, nP, nP)]); s = rng.integers(0, len(common), len(common))
            pb[b] = Dm[s][:, w].mean()
        res["paired_seeds"] = dict(n=len(common), seeds=common, point=float(Dm.mean()), interval=[float(v) for v in np.percentile(pb, [2.5, 97.5])])
    # (4) pass-level sign-flip permutation on window-weighted per-pass differences
    dpass = np.array([(LR[:, ii].mean() - LA[:, ii].mean()) for ii in idx]); wts = np.array([len(ii) for ii in idx], float)
    obs = (dpass * wts).sum() / wts.sum()
    flips = rng.choice([-1.0, 1.0], size=(200000, nP))
    perm = (flips * dpass * wts).sum(1) / wts.sum()
    res["signflip_p_two_sided"] = float(((np.abs(perm) >= abs(obs) - 1e-12).sum() + 1) / (len(perm) + 1))
    res["passes_positive"] = int((dpass > 0).sum())
    out[label] = res
    print(f"{label}: {point:+.3f} dB | percentile [{pct[0]:+.3f}, {pct[1]:+.3f}] | BCa [{bca[0]:+.3f}, {bca[1]:+.3f}] (z0 {z0:+.2f}, a {acc:+.3f})"
          + (f" | paired seeds ({res['paired_seeds']['n']}) {res['paired_seeds']['point']:+.3f} [{res['paired_seeds']['interval'][0]:+.3f}, {res['paired_seeds']['interval'][1]:+.3f}]" if paired else "")
          + f" | sign-flip p = {res['signflip_p_two_sided']:.4f}, {res['passes_positive']}/{nP} passes positive")
contrast("RA+RD", "RA only", "C2_doppler")
contrast("RA+RD+KD", "RA+RD", "C3_kd", paired=True)

# per-band Doppler gain with the paper's bootstrap (unpaired seeds, passes resampled; slots restricted to the band)
LA = np.stack([loss(P) for P in arm("RA+RD").values()]); LR = np.stack([loss(P) for P in arm("RA only").values()])
bands = [(0, 20), (20, 28), (28, 38), (38, 46), (46, 64)]
out["bands"] = []
for lo, hi in bands:
    M = (OPT >= lo) & (OPT < hi)
    def stat(sa, sr, w):
        m = M[w]; return (LR[sr][:, w][:, m].mean() if m.any() else np.nan) - (LA[sa][:, w][:, m].mean() if m.any() else np.nan)
    pt = LR[:, M].mean() - LA[:, M].mean()
    bs = np.array([stat(rng.integers(0, 10, 10), rng.integers(0, 10, 10), np.concatenate([idx[i] for i in rng.integers(0, nP, nP)])) for _ in range(5000)])
    ci = np.nanpercentile(bs, [2.5, 97.5])
    row = dict(band=[lo, hi - 1], share=float(100 * M.mean()), ra_db=float(LR[:, M].mean()), rd_db=float(LA[:, M].mean()),
               ra_gt3=float(100 * (LR[:, M] > 3).mean()), rd_gt3=float(100 * (LA[:, M] > 3).mean()), gain=float(pt), ci=[float(ci[0]), float(ci[1])])
    out["bands"].append(row)
    print(f"beams {lo:2d}-{hi-1:2d}: {row['share']:4.1f}% of slots | RA {row['ra_db']:.2f} dB ({row['ra_gt3']:.1f}% >3) | RA+RD {row['rd_db']:.2f} dB ({row['rd_gt3']:.1f}% >3) | gain {pt:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
json.dump(out, open(OUT / "_w4_intervals.json", "w"), indent=1)
