"""Contrast of one arm against a reference arm: ATop-1, mean beam-selection loss (dB),
tail statistics, paired point estimates and the paper's cluster bootstrap (test passes as clusters,
seeds drawn unpaired per arm, 20,000 replicates, 2.5/97.5 percentiles).

  python analysis/bootstrap_contrast.py --new _w2pred_twostem.npz:twostem --ref _pred_dump.npz:"RA+RD" [--slots 0-6|1-6]
POW is built once from the dataset/scenario9 power files of the 774 test windows (Scenario 9 only). Inputs are read
from results/; the output is results/analysis/_boot_<label>.json."""
import argparse, os, json, sys, collections
import numpy as np
from paths import OUT, result, pow_test
p = argparse.ArgumentParser()
p.add_argument("--new", required=True, help="file:name  (arrays 'name::run')")
p.add_argument("--ref", required=True, help="file:name")
p.add_argument("--slots", default="0-6"); p.add_argument("--reps", type=int, default=20000); p.add_argument("--seed", type=int, default=0)
p.add_argument("--label", default="")
a = p.parse_args()

def load_arm(spec):
    f, name = spec.split(":", 1); f = f if os.path.isabs(f) else str(result(f))
    z = np.load(f); arrs = collections.OrderedDict((k.split("::", 1)[1], z[k].astype(int)) for k in z.files if k.startswith(name + "::"))
    return z["OPT"].astype(int), z["PASS"].astype(int), arrs
OPT, PASS, NEW = load_arm(a.new); OPT2, PASS2, REF = load_arm(a.ref)
assert (OPT == OPT2).all() and (PASS == PASS2).all(), "test sets differ"
lo, hi = map(int, a.slots.split("-")); S = slice(lo, hi + 1)

# power vectors (774 x 7 x 64) from the local dataset
POW = pow_test()
assert (POW.argmax(-1) == OPT).all(), "OPT mismatch"
BEST = np.take_along_axis(POW, OPT[..., None], -1)[..., 0]

def loss(pred):  # (N,7) -> dB loss
    cap = np.take_along_axis(POW, pred[..., None], -1)[..., 0] / BEST
    return -10 * np.log10(np.clip(cap, 1e-6, None))
def stats(preds):
    P = np.stack(preds); L = np.stack([loss(p) for p in preds])[:, :, S]; H = (P == OPT[None])[:, :, S]
    return dict(n=len(preds), atop1=100 * H.mean(), atop1_sd=100 * H.mean((1, 2)).std(ddof=1) if len(preds) > 1 else 0.0,
                mean_db=L.mean(), mean_db_sd=L.mean((1, 2)).std(ddof=1) if len(preds) > 1 else 0.0,
                gt3=100 * (L > 3).mean(), p95=float(np.mean([np.percentile(l, 95) for l in L])), p99=float(np.mean([np.percentile(l, 99) for l in L])))
sn, sr = stats(list(NEW.values())), stats(list(REF.values()))
print(f"[{a.label or a.new}] slots {a.slots}")
print(f"  new {sn['n']:2d} seeds: ATop-1 {sn['atop1']:.2f}±{sn['atop1_sd']:.2f}  mean {sn['mean_db']:.3f}±{sn['mean_db_sd']:.3f} dB  >3dB {sn['gt3']:.2f}%  p95 {sn['p95']:.2f}  p99 {sn['p99']:.2f}")
print(f"  ref {sr['n']:2d} seeds: ATop-1 {sr['atop1']:.2f}±{sr['atop1_sd']:.2f}  mean {sr['mean_db']:.3f}±{sr['mean_db_sd']:.3f} dB  >3dB {sr['gt3']:.2f}%  p95 {sr['p95']:.2f}  p99 {sr['p99']:.2f}")

# per-seed arrays for the bootstrap
def per_seed(arms):
    P = np.stack(list(arms.values())); L = np.stack([loss(p) for p in P]); H = (P == OPT[None]).astype(float)
    return H[:, :, S], L[:, :, S]
Hn, Ln = per_seed(NEW); Hr, Lr = per_seed(REF)
passes = np.unique(PASS); idx = [np.where(PASS == q)[0] for q in passes]
rng = np.random.default_rng(a.seed)
def summ(H, L, w):   # windows index array w (with repeats)
    Hs = H[:, w]; Ls = L[:, w]
    return Hs.mean(), Ls.mean(), (Ls > 3).mean(), np.mean([np.percentile(x, 95) for x in Ls])
d = np.zeros((a.reps, 4))
for r in range(a.reps):
    pw = np.concatenate([idx[i] for i in rng.integers(0, len(passes), len(passes))])
    sn_ = rng.integers(0, Hn.shape[0], Hn.shape[0]); sr_ = rng.integers(0, Hr.shape[0], Hr.shape[0])
    n_ = summ(Hn[sn_], Ln[sn_], pw); r_ = summ(Hr[sr_], Lr[sr_], pw)
    d[r] = [100 * (n_[0] - r_[0]), r_[1] - n_[1], 100 * (n_[2] - r_[2]), n_[3] - r_[3]]
pt = [100 * (Hn.mean() - Hr.mean()), Lr.mean() - Ln.mean(), 100 * ((Ln > 3).mean() - (Lr > 3).mean()), sn['p95'] - sr['p95']]
names = ["dATop-1 [pp] (new-ref)", "dB recovered (ref-new)", "d(>3dB share) [pp]", "d p95 [dB]"]
out = {}
for k, nm in enumerate(names):
    lo_, hi_ = np.percentile(d[:, k], [2.5, 97.5]); out[nm] = dict(point=float(pt[k]), lo=float(lo_), hi=float(hi_))
    print(f"  {nm:24s} {pt[k]:+7.3f}  [{lo_:+.3f}, {hi_:+.3f}]")
json.dump(dict(new=sn, ref=sr, deltas=out, slots=a.slots), open(os.path.join(OUT, f"_boot_{(a.label or 'x').replace(' ','_')}.json"), "w"), indent=1)
