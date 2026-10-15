#!/usr/bin/env python3
"""Collect the per-slot test metrics behind paper Table I from the training runs.

Every run of train_cmkd.py / train_vision_only.py ends by writing eval_results.npz (per-slot top-1, top-3, top-5
accuracy and DBA score over the slots t, t+1, ..., t+6) into its run directory. This script reads those files for
the fourteen rows of Table I (ten seeds each, the run tags below) and writes them to one JSON file in the format of
the shipped results/eval_table1.json, which analysis/table1.py turns into the table.
Usage (from the repository root, after training): python src/collect_table1.py [--out cmkd_results/eval_table1.json]
To use a regenerated file, copy it over results/eval_table1.json."""
import argparse, glob, json, os
import numpy as np

S = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ROWS = [  # (Table I row, run subdirectory holding eval_results.npz, run tag per seed)
    ("Vision-radar teacher", "phase2_*", {42: "_v3_teacher", **{s: f"_v3ca_s{s}" for s in S[1:]}}),
    ("Vision-only", "", {s: f"_v3_visiononly_s{s}" for s in S}),
    ("RA only", "phase0_*", {42: "_v3_noKD_RA", 1: "_v3s1_noKD_RA", 2: "_v3s2_noKD_RA", **{s: f"_c1ra_s{s}" for s in S[3:]}}),
    ("DA only", "phase0_*", {s: f"_m6DAonly_s{s}" for s in S}),
    ("RD only", "phase0_*", {s: f"_m6RDonly_s{s}" for s in S}),
    ("RA+DA", "phase0_*", {42: "_v3_noKD_RADA", 1: "_v3s1_noKD_RADA", 2: "_v3s2_noKD_RADA", **{s: f"_c1da_s{s}" for s in S[3:]}}),
    ("RA+RD (reference)", "phase0_*", {s: f"_v3rd_s{s}" for s in S}),
    ("RA+DA+RD", "phase0_*", {s: f"_m4_3ch_s{s}" for s in S}),
    ("RA+RD + logit", "phase3_*", {s: f"_v3rdkd_s{s}" for s in S}),
    ("RA+RD + logit + feature", "phase3_*", {s: f"_v3rd_feat_s{s}" for s in S}),
    ("RA+RD + logit + feature + temporal", "phase3_*", {s: f"_v3rd_full_s{s}" for s in S}),
    ("RA+DA + logit", "phase3_*", {42: "_v3_logit", 1: "_v3s1_logit", 2: "_v3s2_logit", **{s: f"_v3_logit_s{s}" for s in S[3:]}}),
    ("RA+DA + logit + feature", "phase3_*", {42: "_v3_feat", **{s: f"_v3_feat_s{s}" for s in S[1:]}}),
    ("RA+DA + logit + feature + temporal", "phase3_*", {42: "_v3_full", **{s: f"_v3_full_s{s}" for s in S[1:]}}),
]

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="cmkd_results")
ap.add_argument("--out", default="cmkd_results/eval_table1.json")
a = ap.parse_args()
out, missing = [], []
for label, sub, tags in ROWS:
    runs = []
    for s in S:
        f = sorted(glob.glob(os.path.join(a.runs, tags[s], sub, "eval_results.npz")))
        if not f:
            missing.append(tags[s]); continue
        z = np.load(f[-1], allow_pickle=True)   # the latest run directory of the tag
        runs.append(dict(seed=s, run=os.path.relpath(os.path.dirname(f[-1]), a.runs),
                         **{k: [float(x) for x in np.asarray(z[k]).ravel()] for k in ("top1", "top3", "top5", "dba")}))
    out.append(dict(row=label, runs=runs))
doc = dict(description="Per-slot test metrics of every run behind paper Table I, as written by train_cmkd.py / train_vision_only.py "
           "to cmkd_results/<run>/eval_results.npz at the end of training (top1, top3, top5: per-slot top-k accuracy; dba: per-slot DBA "
           "score; slots t, t+1, ..., t+6; fractions, not percent). Collected with src/collect_table1.py.", rows=out)
json.dump(doc, open(a.out, "w"), indent=1)
print(f"wrote {a.out}: {sum(len(r['runs']) for r in out)} runs" + (f"; missing {len(missing)}: {' '.join(missing)}" if missing else ""))
