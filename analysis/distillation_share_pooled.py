"""Teacher-seed replication: joint cluster bootstrap of the share recovered when the thirty
distilled runs of the three range-Doppler teachers are pooled (G_all) against the three teachers' mean lead (H_all).
Per teacher as well (G_k / H_k). Passes resampled as clusters; seeds (and the three teachers, for H_all) resampled
within each arm, unpaired. Writes _boot_joint_R_all.json."""
import json
from paths import OUT, result, pow_test
import numpy as np
z = np.load(result("_pred_dump.npz")); OPT = z["OPT"].astype(int); PASS = z["PASS"].astype(int)
POW = pow_test(); BEST = np.take_along_axis(POW, OPT[..., None], -1)[..., 0]
loss = lambda P: (-10 * np.log10(np.clip(np.take_along_axis(POW, P[..., None], -1)[..., 0] / BEST, 1e-6, None))).mean(-1)
arm = lambda f, n: np.stack([loss(np.load(result(f))[k].astype(int)) for k in np.load(result(f)).files if k.startswith(n + "::")])
L0 = arm("_pred_dump.npz", "RA+RD")
KD = {"T0": arm("_pred_dump.npz", "RA+RD+KD"), "T1": arm("_w2pred_rdkdT1.npz", "rdkdT1"), "T2": arm("_w2pred_rdkdT2.npz", "rdkdT2")}
tz = np.load(result("_w2pred_rdteacher.npz"))
TE = {"T0": loss(tz["rdteacher::_v3rd_teacher"].astype(int)), "T1": loss(tz["rdteacher::_teacher_rd_s1"].astype(int)), "T2": loss(tz["rdteacher::_teacher_rd_s2"].astype(int))}
KDall = np.concatenate(list(KD.values())); TEall = np.stack(list(TE.values()))
passes = np.unique(PASS); idx = [np.where(PASS == q)[0] for q in passes]; nP = len(passes); rng = np.random.default_rng(0); B = 20000
keys = ["all", "T0", "T1", "T2"]; G = {k: np.empty(B) for k in keys}; H = {k: np.empty(B) for k in keys}
for b in range(B):
    w = np.concatenate([idx[i] for i in rng.integers(0, nP, nP)])
    a0 = L0[rng.integers(0, 10, 10)][:, w].mean()
    G["all"][b] = a0 - KDall[rng.integers(0, 30, 30)][:, w].mean(); H["all"][b] = a0 - TEall[rng.integers(0, 3, 3)][:, w].mean()
    for k in ("T0", "T1", "T2"):
        G[k][b] = a0 - KD[k][rng.integers(0, 10, 10)][:, w].mean(); H[k][b] = a0 - TE[k][w].mean()
pct = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]
out = {}
for k in keys:
    g = L0.mean() - (KDall.mean() if k == "all" else KD[k].mean()); h = L0.mean() - (TEall.mean() if k == "all" else TE[k].mean())
    R = G[k] / H[k]
    out[k] = {"G": [float(g)] + pct(G[k]), "H": [float(h)] + pct(H[k]), "R": [float(g / h)] + pct(R),
              "R_one_sided_95_upper": float(np.percentile(R, 95)), "P_R_gt_quarter": float((R > 0.25).mean())}
json.dump(out, open(OUT / "_boot_joint_R_all.json", "w"), indent=1)
for k in keys:
    o = out[k]; print(f"{k:>3}: G {o['G'][0]:+.3f} [{o['G'][1]:+.3f},{o['G'][2]:+.3f}]  H {o['H'][0]:.3f} [{o['H'][1]:.3f},{o['H'][2]:.3f}]  R {o['R'][0]:+.3f} [{o['R'][1]:+.3f},{o['R'][2]:+.3f}]  P(R>1/4)={o['P_R_gt_quarter']:.4f}")
