"""Angles under the 90-degree-FoV codebook grid and the miss-structure decomposition (supplementary S9), from _pred_dump.npz."""
import numpy as np, json, sys, collections
from pathlib import Path
from paths import OUT, result
f = sys.argv[1] if len(sys.argv) > 1 else str(result("_pred_dump.npz"))
z = np.load(f); OPT = z["OPT"].astype(int)
arms = collections.OrderedDict()
for k in z.files:
    if "::" in k:
        arms.setdefault(k.split("::")[0], []).append(z[k].astype(int))
def ang(idx, fov):
    s = (2.0 * idx / 63.0 - 1.0) * np.sin(np.radians(fov / 2)); return np.degrees(np.arcsin(np.clip(s, -1, 1)))
res = {}
print(f"{'arm':12s} {'n':>2s} {'ATop1':>6s} {'|dIdx|':>6s} {'deg@180(paper)':>14s} {'deg@90(sin-unif)':>16s} {'deg@90 angle-unif':>17s} | {'adj/miss%':>9s} {'<=2/miss%':>9s} {'>5/miss%':>8s}")
for a, preds in arms.items():
    P = np.stack(preds)                       # seeds x 774 x 7
    d = np.abs(P - OPT[None])
    hit = d == 0
    r = dict(n=len(preds), atop1=100 * hit.mean(), mean_idx=d.mean(),
             deg180=np.abs(ang(P, 180) - ang(OPT[None], 180)).mean(),
             deg90=np.abs(ang(P, 90) - ang(OPT[None], 90)).mean(),
             deg90u=(d * (90.0 / 63.0)).mean(),
             adj=100 * (d[~hit] == 1).mean(), le2=100 * (d[~hit] <= 2).mean(), gt5=100 * (d[~hit] > 5).mean(),
             miss_mean_idx=d[~hit].mean())
    res[a] = {k: float(v) for k, v in r.items()}
    print(f"{a:12s} {r['n']:2d} {r['atop1']:6.2f} {r['mean_idx']:6.3f} {r['deg180']:14.2f} {r['deg90']:16.2f} {r['deg90u']:17.2f} | {r['adj']:9.1f} {r['le2']:9.1f} {r['gt5']:8.1f}")
json.dump(res, open(OUT / Path(f).name.replace(".npz", "_analysis.json"), "w"), indent=1)
