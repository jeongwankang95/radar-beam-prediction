"""Per-slot top-1 accuracy and mean beam-selection loss over the seven-slot horizon (Scenario 9 test, 774 windows),
from results/_pred_dump.npz (predictions per arm and seed) and the measured sweep powers.
Loss as in analysis/bootstrap_contrast.py: -10 log10(P(pred)/P(best)), floored at 1e-6. Persistence holds the slot-t beam (exact at t)."""
import json, collections, numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from paths import OUT, result, pow_test
z = np.load(result("_pred_dump.npz")); POW = pow_test()
OPT = z["OPT"].astype(int); assert (POW.argmax(-1) == OPT).all()
BEST = np.take_along_axis(POW, OPT[..., None], -1)[..., 0]
def loss(pred):
    r = np.take_along_axis(POW, pred[..., None], -1)[..., 0] / BEST
    return -10 * np.log10(np.clip(r, 1e-6, None))
arms = collections.OrderedDict()
for k in z.files:
    if "::" in k: arms.setdefault(k.split("::")[0], []).append(z[k].astype(int))
arms["Persistence"] = [np.repeat(OPT[:, :1], 7, axis=1)]
out = {}
print(f"{'arm':12s} n  ATop1  dB   | per-slot top-1 [%] t..t+6            | per-slot loss [dB] t..t+6")
for a, preds in arms.items():
    P = np.stack(preds)                                   # seeds x 774 x 7
    acc = 100 * (P == OPT[None]).mean(axis=1)             # seeds x 7
    L = np.stack([loss(p) for p in P]).mean(axis=1)       # seeds x 7
    out[a] = dict(n=len(P), acc_mean=acc.mean(0).tolist(), acc_sd=(acc.std(0, ddof=1) if len(P) > 1 else 0 * acc[0]).tolist(),
                  db_mean=L.mean(0).tolist(), db_sd=(L.std(0, ddof=1) if len(P) > 1 else 0 * L[0]).tolist(),
                  atop1=float(acc.mean()), atop1_sd=float(acc.mean(1).std(ddof=1)) if len(P) > 1 else 0.0,
                  db=float(L.mean()), db_sd_all=float(L.mean(1).std(ddof=1)) if len(P) > 1 else 0.0,
                  atop1_h=float(acc[:, 1:].mean()), db_h=float(L[:, 1:].mean()))
    o = out[a]
    print(f"{a:12s} {o['n']:2d} {o['atop1']:5.1f} {o['db']:5.2f} | " + " ".join(f"{v:5.1f}" for v in o["acc_mean"]) +
          " | " + " ".join(f"{v:4.2f}" for v in o["db_mean"]))
json.dump(out, open(OUT / "perslot.json", "w"), indent=1)
