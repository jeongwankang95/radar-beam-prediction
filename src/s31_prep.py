#!/usr/bin/env python3
"""Preprocess scenario 31 (daytime urban, LOS+NLOS) into the S9 pipeline format.
Same radar projection chain as s3x_prep (32/33); chronological pass-boundary split
in S32's proportions (~2/3 train, ~1/8 val, rest test)."""
import numpy as np, pandas as pd, os, glob
from concurrent.futures import ProcessPoolExecutor

L, J = 8, 6
root = 'dataset/scenario31'


def project(raw):
    r = np.fft.fft(raw, n=256, axis=1)
    r = r - r.mean(axis=2, keepdims=True)
    d = np.fft.fft(r, n=256, axis=2)
    cube = np.abs(np.fft.fftshift(np.fft.fft(d, n=64, axis=0), axes=(0, 2)))
    ra = cube.mean(2).T
    da = cube.mean(1).T
    rd = cube.mean(0)
    pool = lambda m, fy, fx: m.reshape(m.shape[0]//fy, fy, m.shape[1]//fx, fx).mean((1, 3))
    return (np.log2(1 + pool(ra, 2, 1)).astype(np.float32),
            np.log2(1 + pool(da, 2, 1)).astype(np.float32),
            np.log2(1 + pool(rd, 2, 4)).astype(np.float32))


def one(args):
    root, rel = args
    src = os.path.join(root, rel.lstrip('./'))
    base = os.path.splitext(os.path.basename(rel))[0]
    outs = {k: os.path.join(root, 'unit1', f'radar_{k}', f'{base}_{k}.npy')
            for k in ('RA', 'DA', 'RD')}
    if all(os.path.exists(p) for p in outs.values()):
        return f'./unit1/radar_RA/{base}_RA.npy'
    try:
        raw = np.load(src)
        if raw.shape != (4, 256, 250):
            return ('SHAPE', str(raw.shape))
    except Exception as e:
        return ('ERR', str(e)[:60])
    ra, da, rd = project(raw)
    for k, m in zip(('RA', 'DA', 'RD'), (ra, da, rd)):
        np.save(outs[k], m)
    return f'./unit1/radar_RA/{base}_RA.npy'


cands = sorted(glob.glob(f'{root}/scenario31*dev*.csv') or glob.glob(f'{root}/scenario31*.csv'))
assert cands, f'no dev csv under {root}'
df = pd.read_csv(cands[0])
print(f'dev csv: {cands[0]}  rows={len(df)}  passes={df["seq_index"].nunique()}')
for k in ('RA', 'DA', 'RD'):
    os.makedirs(f'{root}/unit1/radar_{k}', exist_ok=True)
with ProcessPoolExecutor(max_workers=32) as ex:
    ra_paths = list(ex.map(one, [(root, r) for r in df['unit1_radar']], chunksize=16))
bad = [p for p in ra_paths if isinstance(p, tuple)]
if bad:
    print(f'first problem: {bad[0]}')
ra_paths = [p if isinstance(p, str) else None for p in ra_paths]
df['ra_path'] = ra_paths
print(f'S31: {len(df)-len(bad)}/{len(df)} radar maps written ({len(bad)} skipped)')


def windows(rows):
    out = []
    idx = rows.index.to_numpy()
    for a in range(len(rows) - (L - 1) - J):
        w = idx[a:a + L + J]
        if rows.loc[w, 'ra_path'].isna().any():
            continue
        rec = {}
        for i in range(L):
            r = rows.loc[w[i]]
            rec[f'camera{i+1}'] = r['unit1_rgb']
            rec[f'radar{i+1}'] = r['ra_path']
            rec[f'beam{i+1}'] = r['unit1_pwr_60ghz']
        for j in range(J):
            rec[f'future_beam{j+1}'] = rows.loc[w[L + j], 'unit1_pwr_60ghz']
        rec['seq_index'] = int(rows.loc[w[0], 'seq_index'])
        out.append(rec)
    return out


per_pass = {p: windows(g.sort_values('index')) for p, g in df.groupby('seq_index')}
ps = sorted(per_pass)
n = len(ps)
n_tr = max(1, round(n * 2 / 3))
n_va = max(1, round(n / 8))
splits = {'train_seqs_RA.csv': ps[:n_tr],
          'val_seqs_RA.csv': ps[n_tr:n_tr + n_va],
          'test_seqs_RA.csv': ps[n_tr + n_va:]}
cols = ([f'camera{i}' for i in range(1, L + 1)] + [f'radar{i}' for i in range(1, L + 1)]
        + [f'beam{i}' for i in range(1, L + 1)] + [f'future_beam{j}' for j in range(1, J + 1)]
        + ['seq_index'])
for name, passes in splits.items():
    rows = [r for p in passes for r in per_pass[p]]
    pd.DataFrame(rows)[cols].to_csv(f'{root}/{name}', index=False)
    print(f'  {name}: {len(rows)} sequences from {len(passes)} passes')
print('S31_PREP_DONE')
