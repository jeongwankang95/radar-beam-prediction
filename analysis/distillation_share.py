"""Joint cluster bootstrap of the distillation recovery G, the source teacher's lead H
and their ratio R = G/H on Scenario 9 (28 test passes resampled as clusters; seeds resampled within each arm,
unpaired; the source teacher is the single seed-42 range-Doppler checkpoint). Writes _boot_joint_R.json."""
import json
from paths import OUT, result, pow_test
import numpy as np
z = np.load(result("_pred_dump.npz")); OPT = z["OPT"].astype(int); PASS = z["PASS"].astype(int)
POW = pow_test(); BEST = np.take_along_axis(POW, OPT[..., None], -1)[..., 0]
loss = lambda P: (-10 * np.log10(np.clip(np.take_along_axis(POW, P[..., None], -1)[..., 0] / BEST, 1e-6, None))).mean(-1)
arm = lambda f, n: np.stack([loss(np.load(result(f))[k].astype(int)) for k in np.load(result(f)).files if k.startswith(n + "::")])
L0, L1, V = arm("_pred_dump.npz", "RA+RD"), arm("_pred_dump.npz", "RA+RD+KD"), arm("_pred_dump.npz", "Vision-only")
tz = np.load(result("_w2pred_rdteacher.npz")); T = loss(tz["rdteacher::_v3rd_teacher"].astype(int))[None]
passes = np.unique(PASS); idx = [np.where(PASS == q)[0] for q in passes]; nP = len(passes); rng = np.random.default_rng(0); B = 20000
G, H, R, Vr, Vg = (np.empty(B) for _ in range(5))
for b in range(B):
    w = np.concatenate([idx[i] for i in rng.integers(0, nP, nP)])
    a0 = L0[rng.integers(0, 10, 10)][:, w].mean(); a1 = L1[rng.integers(0, 10, 10)][:, w].mean(); t = T[:, w].mean(); v = V[rng.integers(0, 10, 10)][:, w].mean()
    G[b], H[b] = a0 - a1, a0 - t; R[b] = G[b] / H[b]; Vr[b] = a0 - v; Vg[b] = v - t
pct = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]
out = {
    "n_seeds": {"noKD": len(L0), "KD": len(L1), "vision": len(V), "teacher": 1}, "n_passes": int(nP), "B": B,
    "mean_loss_dB": {"noKD": float(L0.mean()), "KD": float(L1.mean()), "source_teacher": float(T.mean()), "vision": float(V.mean())},
    "G": [float(L0.mean() - L1.mean())] + pct(G), "H": [float(L0.mean() - T.mean())] + pct(H),
    "R": [float((L0.mean() - L1.mean()) / (L0.mean() - T.mean()))] + pct(R),
    "R_one_sided_95_upper": float(np.percentile(R, 95)), "P_R_gt_fifth": float((R > 0.2).mean()), "P_R_gt_quarter": float((R > 0.25).mean()),
    "vision_reach": [float(L0.mean() - V.mean())] + pct(Vr), "vision_short_of_teacher": [float(V.mean() - T.mean())] + pct(Vg),
    "vision_share_of_H": float((L0.mean() - V.mean()) / (L0.mean() - T.mean())),
}
json.dump(out, open(OUT / "_boot_joint_R.json", "w"), indent=1); print(json.dumps(out, indent=1))
