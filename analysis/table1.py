"""Table I: long-term accuracy (ATop-1, ATop-3, ATop-5) and ADBA, averaged over the seven slots t..t+6, as mean and
sample std over the ten seeds of each row. Input: results/eval_table1.json (per-slot metrics of every run, collected
from the runs' eval_results.npz by src/collect_table1.py). Also prints the same means over t+1..t+6 only (the reading
of supplementary Sec. S1). Writes results/analysis/table1.json."""
import json
import numpy as np
from paths import OUT, result

doc = json.load(open(result("eval_table1.json")))
KEYS = ("top1", "top3", "top5", "dba")
rows = {}
print(f"{'row':36s} {'n':>2s}  {'ATop-1':>10s} {'ATop-3':>10s} {'ATop-5':>10s} {'ADBA':>10s}   t+1..t+6: ATop-1 ATop-3 ATop-5 ADBA")
for r in doc["rows"]:
    m7 = np.array([[100 * np.mean(x[k]) for k in KEYS] for x in r["runs"]])
    m6 = np.array([[100 * np.mean(x[k][1:]) for k in KEYS] for x in r["runs"]])
    mean, sd = m7.mean(0), m7.std(0, ddof=1)
    rows[r["row"]] = dict(seeds=[x["seed"] for x in r["runs"]], mean=mean.tolist(), std=sd.tolist(), mean_t1_t6=m6.mean(0).tolist())
    print(f"{r['row']:36s} {len(m7):2d}  " + " ".join(f"{mean[i]:5.1f} ±{sd[i]:3.1f}" for i in range(4))
          + "   " + " ".join(f"{v:6.1f}" for v in m6.mean(0)))
json.dump(dict(columns=["ATop-1", "ATop-3", "ATop-5", "ADBA"], rows=rows), open(OUT / "table1.json", "w"), indent=1)
