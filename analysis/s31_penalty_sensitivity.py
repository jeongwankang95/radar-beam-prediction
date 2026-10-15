"""How Scenario 31's loss treats a selected beam whose power was not measured.

s31_test.py replaces missing power entries by 0 and floors the power ratio at 1e-6, so such a selection scores
60 dB, a numerical penalty rather than a measured loss. This script counts those selections per arm in the stored
per-slot losses and recomputes the gain (range-angle minus range-Doppler pairing, seven slots) under two treatments:
the penalty set to 0 dB on those predictions only, and a common mask that drops their window-slots from both arms
and all seeds. Intervals use s31_test.py's cluster bootstrap (eleven passes with replacement, seeds unpaired,
20,000 replicates, generator seed 42). Inputs: results/_s31fix_test.npz (corrected labels, the reported arms) and
results/_s31_test.npz (before the correction). Output: results/analysis/_s31_penalty_sens.json."""
import json
from paths import OUT, result
import numpy as np

PEN = 59.9  # losses at the 1e-6 floor are 60 dB; no measured loss in these files comes near it


def boot(LA, LB, PASS, nb=20000):
    rng = np.random.default_rng(42)
    passes = np.unique(PASS)
    widx = [np.where(PASS == p)[0] for p in passes]
    n = LA.shape[0]
    mA, mB = LA.mean(2), LB.mean(2)  # every window has seven slots, so window means average to the slot mean
    r = np.empty(nb)
    for k in range(nb):
        idx = np.concatenate([widx[p] for p in rng.integers(0, len(passes), len(passes))])
        sa, sb = rng.integers(0, n, n), rng.integers(0, n, n)
        r[k] = mA[sa][:, idx].mean() - mB[sb][:, idx].mean()
    return [float(x) for x in np.percentile(r, [2.5, 97.5])]


out = {}
for tag, path in (("corrected", result("_s31fix_test.npz")), ("pre_correction", result("_s31_test.npz"))):
    z = np.load(path)
    LA, LB, PASS, ok = z["LA"].astype(np.float64), z["LB"].astype(np.float64), z["pass_id"], z["ok"]
    hA, hB = LA > PEN, LB > PEN
    keep = ~(hA.any(0) | hB.any(0))
    d = lambda A, B: float((A.mean((1, 2)) - B.mean((1, 2))).mean())
    A0, B0 = np.where(hA, 0.0, LA), np.where(hB, 0.0, LB)
    masked = float(np.mean([LA[s][keep].mean() - LB[s][keep].mean() for s in range(LA.shape[0])]))
    rec = dict(seeds=int(LA.shape[0]), windows=int(LA.shape[1]), slots=int(LA.shape[1] * LA.shape[2]),
               unscored_slots=int((~ok).sum()), predictions_per_arm=int(LA.size),
               penalized_RA=int(hA.sum()), penalized_RD=int(hB.sum()),
               penalized_RD_per_seed=[int(x) for x in hB.sum((1, 2))], window_slots_involved=int((~keep).sum()),
               dL7_reported=d(LA, LB), dL7_penalty0=d(A0, B0), dL7_common_mask=masked)
    if tag == "corrected":
        rec["ci_reported"], rec["ci_penalty0"] = boot(LA, LB, PASS), boot(A0, B0, PASS)
    out[tag] = rec
    print(tag, json.dumps(rec))
json.dump(out, open(OUT / "_s31_penalty_sens.json", "w"), indent=1)
print("saved _s31_penalty_sens.json")
