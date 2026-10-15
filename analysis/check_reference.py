"""Compare the outputs regenerated in results/analysis/ with the copies behind the paper in results/reference/.
Every JSON present in both folders is compared field by field (floats to a relative tolerance of 1e-9); files missing
from results/analysis/ are listed as not yet regenerated. Without arguments it also compares the prediction files that
analysis scripts rebuild with those in results/. Usage: python analysis/check_reference.py [name ...]"""
import json, math, sys
import numpy as np
from paths import OUT, REFERENCE, RESULTS


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}.{k}: missing" for k in a.keys() ^ b.keys()]
        for k in a.keys() & b.keys():
            out += diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in diff(u, v, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        same = (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
        return [] if same else [f"{path}: {a!r} vs {b!r}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


names = sys.argv[1:] or sorted(p.name for p in REFERENCE.glob("*.json"))
ok = bad = missing = 0
for n in names:
    if not (OUT / n).exists():
        missing += 1
        continue
    d = diff(json.load(open(OUT / n)), json.load(open(REFERENCE / n)))
    if d:
        bad += 1
        print(f"DIFF {n}: " + "; ".join(d[:4]) + (" ..." if len(d) > 4 else ""))
    else:
        ok += 1
print(f"{ok} identical, {bad} different, {missing} not regenerated (of {len(names)})")
if not sys.argv[1:]:   # prediction files that analysis scripts rebuild (tracker, pooled teacher seeds), against results/
    for f in sorted(OUT.glob("*.npz")):
        if (RESULTS / f.name).exists():
            a, b = np.load(f), np.load(RESULTS / f.name)
            same = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
            print(f"{'identical' if same else 'DIFF'}: {f.name} against results/{f.name}")
