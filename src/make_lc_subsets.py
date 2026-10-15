"""Learning-curve subsets (Sec. VII-B, supplementary S11): dataset/scenario9/train_seqs_RA_tr_p{10,20,45}.csv.
Each subset keeps the rows of train_seqs_RA_tr.csv whose pass (seq_index) is among the first P passes of a fixed random
permutation of the 91 training passes; the permutation and the nested pass lists are recorded in
train_subsets_provenance.json, which this script follows. Run from the repository root: python src/make_lc_subsets.py.
The p10 and p20 files shipped with the repository are reproduced byte for byte; p45 is regenerated here."""
import json
import os
import numpy as np
import pandas as pd

D = os.path.join("dataset", "scenario9")
prov = json.load(open(os.path.join(D, "train_subsets_provenance.json")))
tr = pd.read_csv(os.path.join(D, "train_seqs_RA_tr.csv"))
perm = [int(p) for p in prov["permutation"]]
assert sorted(perm) == sorted(int(p) for p in np.unique(tr["seq_index"])), "permutation does not cover the 91 training passes"
for key, sub in prov["subsets"].items():
    P = int(key[1:])
    passes = sorted(perm[:P])
    assert passes == sub["passes"], f"{key}: pass list differs from the provenance file"
    out = tr[tr["seq_index"].isin(passes)]
    assert len(out) == sub["n_sequences"], f"{key}: {len(out)} windows, expected {sub['n_sequences']}"
    f = os.path.join(D, f"train_seqs_RA_tr_{key}.csv")
    out.to_csv(f, index=False)
    print(f"{f}: {P} passes, {len(out)} windows")
