"""Flatness of the beam-power peak over all Scenario 9 power vectors (paper Sec. VI).

For every 64-beam power vector listed in the Scenario 9 CSV, compares the best and the second-best beam: their gap in dB
and whether they are immediate neighbours in the codebook. Prints the median gap and the neighbour share.
Usage: python flat_peak.py --root <path to dataset/scenario9> [--out flat_peak.json]"""
import argparse, csv, json, os
from paths import DATA, OUT
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(DATA))
ap.add_argument("--out", default=str(OUT / "flat_peak.json"))
a = ap.parse_args()
rows = list(csv.DictReader(open(os.path.join(a.root, "scenario9.csv"))))
gaps, nb = [], []
for r in rows:
    p = np.loadtxt(os.path.join(a.root, r["unit1_pwr_60ghz"].lstrip("./")))
    p = np.nan_to_num(p, nan=0.0)
    o = np.argsort(p)[::-1]
    gaps.append(10 * np.log10(p[o[0]] / p[o[1]]))
    nb.append(abs(int(o[0]) - int(o[1])) == 1)
res = dict(vectors=len(rows), median_gap_dB=float(np.median(gaps)), neighbour_share=float(np.mean(nb)))
json.dump(res, open(a.out, "w"), indent=1)
print(f"{res['vectors']} vectors: median best-to-second gap {res['median_gap_dB']:.2f} dB; second best is an "
      f"immediate neighbour {100 * res['neighbour_share']:.1f}% of the time")
