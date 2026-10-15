"""Paths shared by the analysis and figure scripts; everything is relative to the repository root."""
import csv, os
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CMKD_DATA", REPO / "dataset" / "scenario9"))   # DeepSense 6G Scenario 9 (CMKD_DATA overrides)
RESULTS = REPO / "results"                  # prediction dumps behind the paper (inputs)
REFERENCE = RESULTS / "reference"           # analysis outputs reported in the paper
OUT = RESULTS / "analysis"                  # outputs written by these scripts
CACHE = RESULTS / "cache"                   # files derived from the dataset, not distributed
for _d in (OUT, CACHE):
    _d.mkdir(parents=True, exist_ok=True)


def result(name):
    """An input: results/<name>, else a file written by an earlier analysis step (results/analysis/<name>)."""
    for d in (RESULTS, OUT):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def output_or_reference(name):
    """An analysis output for the figure scripts: the regenerated one if present, else the paper's reference copy."""
    return OUT / name if (OUT / name).exists() else REFERENCE / name


def pow_test():
    """Measured 64-beam powers of the 774 Scenario 9 test windows and seven slots (774 x 7 x 64), built once."""
    f = CACHE / "_pow_test_s9.npz"
    if not f.exists():
        rows = list(csv.DictReader(open(DATA / "test_seqs_RA.csv")))
        cols = ["beam8"] + [f"future_beam{j}" for j in range(1, 7)]
        np.savez(f, POW=np.stack([[np.loadtxt(DATA / r[c].lstrip("./")) for c in cols] for r in rows]))
    return np.load(f)["POW"]
