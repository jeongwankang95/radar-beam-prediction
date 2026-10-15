#!/usr/bin/env python3
"""Print the training command of any run in results/runs.json.

Usage, from the repository root:  python scripts/train_command.py TAG [TAG ...]
    e.g.  python scripts/train_command.py _lam0p3_rdkd_s1 _dascr_s42 _v3rd_teacher
For every run directory of the tag (a teacher has two: phase 1, then phase 2) it prints the environment switches that
src/DataFeed.py reads and the src/train_cmkd.py arguments recorded in that directory's args.json. A teacher checkpoint
is written as a glob over that teacher's run directories, so the command uses the teacher you trained (or a released
checkpoint placed under the same path). Vision-only runs record no arguments; the launcher listed with their family
trains them. Families whose launcher says "from this file" have no script in scripts/; use the printed commands."""
import json, re, shlex, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
doc = json.load(open(REPO / "results" / "runs.json"))
RUNS = {r["tag"]: (f, r) for f in doc["families"] for r in f["runs"]}
FLAGS = {"two_stem", "preserve_angle", "debug", "zero_second"}   # store_true options of train_cmkd.py

if len(sys.argv) < 2:
    sys.exit(__doc__)
for tag in sys.argv[1:]:
    if tag not in RUNS:
        sys.exit(f"{tag}: not in results/runs.json")
    fam, run = RUNS[tag]
    print(f"# {tag}: {fam['family']} (seed {run['seed']}; launcher: {fam['launcher']})")
    env = "".join(f"{k}={v} " for k, v in run["env"].items())
    for ph in run["phases"]:
        a = ph.get("args")
        if a is None:
            print(f"#   {ph['dir']}: no recorded arguments; see the launcher")
            continue
        parts = []
        for k, v in a.items():
            if k in FLAGS:
                parts += [f"--{k}"] if v else []
            elif k == "teacher_ckpt" and v:
                g = re.sub(r"/(phase\d)_\d{4}_\d{4}/", r"/\1_*/", v)
                parts.append(f'--teacher_ckpt "$(ls {g} | tail -1)"')
            else:
                parts.append(f"--{k} {shlex.quote(str(v))}")
        print(f"#   recorded in cmkd_results/{ph['dir']}")
        print(f"{env}python src/train_cmkd.py " + " ".join(parts))
