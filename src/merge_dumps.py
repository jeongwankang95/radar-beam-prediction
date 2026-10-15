"""Merge per-arm prediction dumps (from src/w2_pred.py) into one file, keeping the given order of files and of runs.
Usage: python src/merge_dumps.py OUT.npz IN1.npz IN2.npz ...   (every input must share OPT and PASS)."""
import sys
import numpy as np

out_f, ins = sys.argv[1], sys.argv[2:]
out = {}
for f in ins:
    z = np.load(f)
    if not out:
        out = {"OPT": z["OPT"], "PASS": z["PASS"]}
    assert (z["OPT"] == out["OPT"]).all() and (z["PASS"] == out["PASS"]).all(), f"{f}: a different test set"
    out.update({k: z[k] for k in z.files if "::" in k})
np.savez(out_f, **out)
print(f"{out_f}: {sum('::' in k for k in out)} runs from {len(ins)} files")
