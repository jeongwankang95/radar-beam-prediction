#!/usr/bin/env python3
"""Carve a leakage-free validation split from train_seqs_RA.csv.

Sequences are sliding windows (stride 1) so they overlap WITHIN a vehicle pass, but
passes are separated by large frame gaps. Splitting at pass (segment) boundaries is
therefore leakage-free. We take the temporally LAST ~15% of passes as validation,
mirroring the train->test temporal shift.
"""
import csv, re, sys

SRC = 'dataset/scenario9/train_seqs_RA.csv'
OUT_TR = 'dataset/scenario9/train_seqs_RA_tr.csv'
OUT_VA = 'dataset/scenario9/val_seqs_RA.csv'
VAL_FRAC = 0.15

with open(SRC) as f:
    rdr = csv.reader(f); header = next(rdr); rows = list(rdr)
ci = header.index('camera1')
frame = lambda r: int(re.search(r'image_BS2_(\d+)_', r[ci]).group(1))

# group rows into segments (passes) by frame discontinuity
segs, cur = [], [rows[0]]
for prev, row in zip(rows, rows[1:]):
    if frame(row) - frame(prev) == 1:
        cur.append(row)
    else:
        segs.append(cur); cur = [row]
segs.append(cur)

n = len(rows); target = int(round(VAL_FRAC * n))
# take whole segments from the END until we reach ~target sequences
val_segs, count = [], 0
for s in reversed(segs):
    if count >= target: break
    val_segs.append(s); count += len(s)
val_segs.reverse()
n_val_segs = len(val_segs)
tr_segs = segs[: len(segs) - n_val_segs]

tr_rows = [r for s in tr_segs for r in s]
va_rows = [r for s in val_segs for r in s]
assert len(tr_rows) + len(va_rows) == n

# leakage check: no frame overlap between the two splits
tr_frames = {frame(r) for r in tr_rows}; va_frames = {frame(r) for r in va_rows}
assert not (tr_frames & va_frames), 'frame overlap between train and val!'
gap = min(va_frames) - max(tr_frames)

for path, rws in [(OUT_TR, tr_rows), (OUT_VA, va_rows)]:
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rws)

print(f'total segments={len(segs)}  sequences={n}')
print(f'  train: {len(tr_segs):3d} segs, {len(tr_rows):4d} seqs  (frames {min(tr_frames)}..{max(tr_frames)})')
print(f'  val  : {n_val_segs:3d} segs, {len(va_rows):4d} seqs  (frames {min(va_frames)}..{max(va_frames)})  = {100*len(va_rows)/n:.1f}%')
print(f'  boundary frame gap = {gap}  (>=14 needed for no window overlap) -> {"OK" if gap >= 14 else "LEAK"}')
print(f'wrote {OUT_TR}, {OUT_VA}')
