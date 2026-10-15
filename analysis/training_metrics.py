"""Supplementary numbers that come from the test metrics each run wrote at the end of training (results/runs.json),
not from the prediction dumps: the teacher-fusion table and its paired deltas (supplementary S11), the temperature
sweep at seed 42 and the MLP projector (S10), and the Doppler-resolution sweep (S13). Accuracy is the mean over the
seven slots t..t+6, in percent. Writes results/analysis/training_metrics.json."""
import json
import numpy as np
from paths import OUT, result

doc = json.load(open(result("runs.json")))
RUNS = {r["tag"]: r for f in doc["families"] for r in f["runs"]}
KEYS = ("top1", "top3", "top5", "dba")


def metrics(tag, phase=""):
    """Seven-slot test metrics of the tag's latest run directory of the given phase."""
    ph = [p for p in RUNS[tag]["phases"] if "test" in p and p["dir"].split("/")[-1].startswith(phase)]
    return np.array([100 * np.mean(ph[-1]["test"][k]) for k in KEYS])


out = {}
TEN = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)
print("Teacher fusion at the task-loss stage (phase 1), Doppler-angle second map: ATop-1, ATop-3, ATop-5, ADBA")
arms = {"radar-query cross-attention": np.array([metrics("_v3_teacher" if s == 42 else f"_v3ca_s{s}", "phase1") for s in TEN]),
        "concatenation": np.array([metrics(f"_v3cc_s{s}", "phase1") for s in TEN]),
        "vision-only student": np.array([metrics(f"_v3_visiononly_s{s}") for s in TEN])}
for name, X in arms.items():
    out[name] = dict(mean=X.mean(0).tolist(), std=X.std(0, ddof=1).tolist())
    print(f"  {name:28s} " + "  ".join(f"{m:5.1f} ±{s:3.1f}" for m, s in zip(X.mean(0), X.std(0, ddof=1))))
for a, b in (("concatenation", "radar-query cross-attention"), ("concatenation", "vision-only student"),
             ("radar-query cross-attention", "vision-only student")):
    d = arms[a] - arms[b]   # paired by seed; a tie (difference 0 to four decimals) does not count as positive
    pos = (np.round(d, 4) > 0).sum(0)
    out[f"{a} - {b}"] = dict(mean=d.mean(0).tolist(), positive_seeds=pos.tolist())
    print(f"  {a + ' - ' + b:56s} " + "  ".join(f"{m:+.1f} ({p}/10)" for m, p in zip(d.mean(0), pos)))

print("\nLogit-distillation temperature, reference student, seed 42: test ATop-1")
temp = {"1": "_kd_T1_s42", "2": "_kd_T2_s42", "3 (reported)": "_v3rdkd_s42", "5": "_kd_T5_s42", "no distillation": "_v3rd_s42"}
out["temperature_seed42"] = {k: float(metrics(t)[0]) for k, t in temp.items()}
print("  " + ", ".join(f"{k}: {v:.2f}" for k, v in out["temperature_seed42"].items()))

mlp = np.array([metrics(f"_v3rd_featmlp_s{s}")[0] for s in (42, 1, 2, 3, 4, 5, 6, 7, 8)])
out["mlp_projector"] = dict(seeds=9, mean=float(mlp.mean()), std=float(mlp.std(ddof=1)))
print(f"\nMLP projector (logit + feature, reference student, nine seeds): ATop-1 {mlp.mean():.1f} ±{mlp.std(ddof=1):.1f}")

print("\nDoppler-resolution sweep (Doppler-angle map pooled into B groups, five seeds each): ATop-1 and its difference to B = 128")
pool = {B: np.array([metrics(f"_gridDApool{B}_s{s}")[0] for s in (42, 1, 2, 3, 4)]) for B in (1, 2, 4, 8, 16, 32, 64, 128)}
out["doppler_pooling"] = {}
for B, v in pool.items():
    d = v.mean() - pool[128].mean()
    out["doppler_pooling"][str(B)] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)), minus_B128=float(d))
    print(f"  B = {B:3d}: {v.mean():6.2f} ±{v.std(ddof=1):4.2f}   {d:+6.2f} pp")
json.dump(out, open(OUT / "training_metrics.json", "w"), indent=1)
