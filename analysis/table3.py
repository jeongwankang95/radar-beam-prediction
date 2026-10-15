"""Table III: controls on radar-map content, as ATop-1 differences in percentage points with 95% intervals.

Panel (a), axis transformations. Input: results/_predcache.npz, the per-window top-1 predictions (774 Scenario 9 test
windows x 7 slots) of the transformed arms (src/predcache_ladder.py) and of their comparators, ten seeds per arm (five
for the range-permuted range-Doppler arm), with the labels they are scored against. Bootstrap: the 28 test passes are
resampled as clusters together with the seeds of each arm (unpaired), B = 40,000 replicates from one generator seeded
with 0. For every delta three intervals are drawn in turn (seeds only, passes only, passes and seeds); the table reports
the last. Because all deltas share the generator, the calls run in their original order: the first twelve reproduce the
original computation line for line. The two range-permuted range-Doppler rows were computed separately and are appended
here; their endpoints can differ from earlier text by a few hundredths of a point (Monte Carlo error).
The statistic of a replicate is each arm's mean accuracy over the resampled windows, averaged over the resampled seeds;
it is computed from per-pass sums, which is the same quantity.
Panel (b), moment maps, is read from the bootstrap files of analysis/run_bootstraps.sh. The footnote's ATop-1 of each
transformed arm is printed as well. Writes results/analysis/table3.json.
Options: --cache DIR uses per-run files <family>_s<seed>.npz written by src/predcache_ladder.py in place of the shipped
predictions of the same name; --nboot N changes B (the intervals then differ from the paper's)."""
import argparse, glob, json, os
import numpy as np
from paths import OUT, result, output_or_reference, pow_test

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=None)
ap.add_argument("--nboot", type=int, default=40000)
a = ap.parse_args()
B = a.nboot

z = np.load(result("_predcache.npz"))
LABEL = z["LABEL"]
runs = {k.split("::")[1]: z[k] for k in z.files if "::" in k}
if a.cache:
    for f in glob.glob(os.path.join(a.cache, "*_s*.npz")):
        d = np.load(f); runs[os.path.basename(f)[:-4]] = d["pred"]
arms = {}
for name in sorted(runs, key=lambda n: n + ".npz"):   # the order in which the original script globbed the cache files
    arms.setdefault(name.rsplit("_s", 1)[0], []).append((runs[name] == LABEL).astype(float))

pas = np.load(result("_pred_dump.npz"))["PASS"].astype(int)
passes = [np.where(pas == p)[0] for p in dict.fromkeys(pas.tolist())]   # passes in order of first appearance
PS = {k: np.array([[x[p].sum() for p in passes] for x in v]) for k, v in arms.items()}   # seeds x passes
PC = np.array([7.0 * len(p) for p in passes])
P_ALL = np.arange(len(passes))
rng = np.random.default_rng(0)


def stat(fam, pidx, seeds):
    return 100 * np.mean([PS[fam][j][pidx].sum() / PC[pidx].sum() for j in seeds])


def intervals(fams, value):
    out = []
    for rp, rs in ((False, True), (True, False), (True, True)):
        v = []
        for _ in range(B):
            pidx = rng.integers(0, len(passes), len(passes)) if rp else P_ALL
            sd = {f: (rng.integers(0, len(arms[f]), len(arms[f])) if rs else np.arange(len(arms[f]))) for f in fams}
            x = value(pidx, sd)
            if x is not None:
                v.append(x)
        out.append([float(q) for q in np.percentile(v, [2.5, 97.5])])
    return out


res = {}


def delta(new, ref, label):
    sd0 = {f: np.arange(len(arms[f])) for f in (new, ref)}
    pt = stat(new, P_ALL, sd0[new]) - stat(ref, P_ALL, sd0[ref])
    ci = intervals((new, ref), lambda p, s: stat(new, p, s[new]) - stat(ref, p, s[ref]))
    res[label] = dict(new=new, ref=ref, point=float(pt), seeds_only=ci[0], passes_only=ci[1], passes_and_seeds=ci[2])
    print(f"{label:34s} {pt:+6.2f}  [{ci[2][0]:+6.2f}, {ci[2][1]:+6.2f}]   (seeds only [{ci[0][0]:+.2f}, {ci[0][1]:+.2f}], "
          f"passes only [{ci[1][0]:+.2f}, {ci[1][1]:+.2f}])", flush=True)


def share(x, base, top, label):
    f = lambda fam, p, s: stat(fam, p, s[fam])
    sd0 = {k: np.arange(len(arms[k])) for k in (x, base, top)}
    pt = 100 * (f(x, P_ALL, sd0) - f(base, P_ALL, sd0)) / (f(top, P_ALL, sd0) - f(base, P_ALL, sd0))

    def value(p, s):
        den = f(top, p, s) - f(base, p, s)
        return 100 * (f(x, p, s) - f(base, p, s)) / den if abs(den) > 1e-9 else None
    ci = intervals((x, base, top), value)
    res[label] = dict(point=float(pt), seeds_only=ci[0], passes_only=ci[1], passes_and_seeds=ci[2])
    print(f"{label:34s} {pt:6.1f}%  [{ci[2][0]:+6.1f}, {ci[2][1]:+6.1f}]", flush=True)


print("Panel (a): ATop-1 difference [pp], point and 95% interval (passes and seeds)")
delta("dascr", "daonly", "DA rows permuted - DA only")          # Table III(a), row 1
delta("damarg", "ra", "DA per-angle mean - RA only")            # supplementary S14
delta("damarg", "dascr", "DA per-angle mean - DA permuted")     # row 2
delta("dabin", "daonly", "DA binarized - DA only")              # row 3
delta("dabin", "dascr", "DA binarized - DA permuted")
delta("dabin", "ra", "DA binarized - RA only")
delta("rdmarg", "rdonly", "RD per-velocity mean - RD only")     # row 5
delta("rdmarg", "ra", "RD per-velocity mean - RA only")         # supplementary S14
print("Retained share of the gap to the full map (not reported in the paper)")
share("dascr", "ra", "daonly", "DA permuted / DA-only gap")
share("damarg", "ra", "daonly", "DA per-angle mean / DA-only gap")
share("dabin", "ra", "daonly", "DA binarized / DA-only gap")
share("rdmarg", "ra", "rdonly", "RD per-velocity mean / RD-only gap")
print("Appended: the range-permuted range-Doppler arm")
delta("rdscr", "rd", "RD rows permuted - RA+RD")                # row 4
delta("rdscr", "ra", "RD rows permuted - RA only")              # supplementary S14

print("\nPanel (b): ATop-1 difference [pp] against DA only (analysis/run_bootstraps.sh)")
PANEL_B = (("DA total at centroid", "_boot_delta_vs_full_10.json"), ("DA Gaussian moments", "_boot_P5_gauss_vs_full_10.json"),
           ("DA Gaussian + floor", "_boot_gaussfl_vs_full.json"), ("DA spread only", "_boot_spread_vs_full.json"))
for label, name in PANEL_B:
    d = json.load(open(output_or_reference(name)))["deltas"]["dATop-1 [pp] (new-ref)"]
    res[label] = dict(source=name, point=d["point"], passes_and_seeds=[d["lo"], d["hi"]])
    print(f"{label:34s} {d['point']:+6.2f}  [{d['lo']:+6.2f}, {d['hi']:+6.2f}]")

print("\nTransformed-arm ATop-1 (%), mean over seeds")
OPT = np.load(result("_pred_dump.npz"))["OPT"].astype(int)
acc = {}
for fam, label in (("dascr", "DA permuted"), ("damarg", "per-angle mean"), ("dabin", "binarized"), ("rdscr", "RD permuted"),
                   ("rdmarg", "RD per-velocity mean")):
    acc[label] = float(np.mean([100 * x.mean() for x in arms[fam]]))
for dump, key, label in (("_w2pred_dadelta.npz", "dadelta", "total at centroid"), ("_w2pred_dagauss.npz", "dagauss", "Gaussian"),
                         ("_w2pred_dagaussfl.npz", "dagaussfl", "Gaussian + floor"), ("_w2pred_daspread.npz", "daspread", "spread only")):
    w = np.load(result(dump))
    acc[label] = float(np.mean([100 * (w[k].astype(int) == OPT).mean() for k in w.files if k.startswith(key + "::")]))
print(", ".join(f"{k} {v:.1f}" for k, v in acc.items()))
json.dump(dict(nboot=B, deltas=res, transformed_arm_atop1=acc), open(OUT / "table3.json", "w"), indent=1)
