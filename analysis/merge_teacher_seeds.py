"""Pooled arm of the teacher-seed replication (supplementary S10): the ten distilled students of each of the three
range-Doppler teachers (the seed-42 teacher's from results/_pred_dump.npz, teachers 1 and 2 from the rdkdT1 and rdkdT2
dumps), written as results/analysis/_w2pred_rdkdAll.npz. The shipped results/_w2pred_rdkdAll.npz is this file."""
import numpy as np
from paths import OUT, result

a, b, c = (np.load(result(f)) for f in ("_pred_dump.npz", "_w2pred_rdkdT1.npz", "_w2pred_rdkdT2.npz"))
assert all((z["OPT"] == a["OPT"]).all() and (z["PASS"] == a["PASS"]).all() for z in (b, c)), "test sets differ"
out = {"OPT": a["OPT"], "PASS": a["PASS"]}
for z, name, tag in ((a, "RA+RD+KD", "T0"), (b, "rdkdT1", "T1"), (c, "rdkdT2", "T2")):
    ks = [k for k in z.files if k.startswith(name + "::")]
    assert len(ks) == 10, (name, len(ks))
    for k in ks:
        out["rdkdAll::" + tag + "_" + k.split("::", 1)[1]] = z[k]
np.savez(OUT / "_w2pred_rdkdAll.npz", **out)
print("pooled", sum(k.startswith("rdkdAll::") for k in out), "distilled runs ->", OUT / "_w2pred_rdkdAll.npz")
