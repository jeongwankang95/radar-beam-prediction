#!/usr/bin/env python3
"""Preprocess scenarios 32/33 into the S9 pipeline's format, for the night-transfer
gate G3.

Radar: raw (4, 256, 250) complex -> 3D FFT (angle 64, range 256, Doppler zero-padded
to 256) -> RA / DA / RD projections, each pooled to (128, 64) so every trained
architecture runs unchanged. MTI (mean over chirps) as in S9. Range axis is kept
whole (no near-range clipping): D1 showed the range calibration is unusable, so no
informed crop exists; 2x pooling preserves the full extent instead.

Sequences: L=8 past + J=6 future windows inside each pass. S32 splits by pass
chronologically (10 train / 2 val / 3 test). S33 is emitted as one big test CSV --
it is the night OOD set, never trained on.

Output CSVs use the S9 column layout (camera1..8, radar1..8 = RA paths, beam1..8 and
future_beam1..6 = power-vector paths, seq_index), so DataFeed works as-is.

Run from the repository root:  python src/s3x_prep.py
"""
import numpy as np, pandas as pd, os, sys
from concurrent.futures import ProcessPoolExecutor

L, J = 8, 6


def project(raw):
    r = np.fft.fft(raw, n=256, axis=1)
    r = r - r.mean(axis=2, keepdims=True)                     # MTI
    d = np.fft.fft(r, n=256, axis=2)
    cube = np.abs(np.fft.fftshift(np.fft.fft(d, n=64, axis=0), axes=(0, 2)))  # (64a,256r,256v)
    ra = cube.mean(2).T                                       # (256 r, 64 a)
    da = cube.mean(1).T                                       # (256 v, 64 a)
    rd = cube.mean(0)                                         # (256 r, 256 v)
    pool = lambda m, fy, fx: m.reshape(m.shape[0]//fy, fy, m.shape[1]//fx, fx).mean((1, 3))
    return (np.log2(1 + pool(ra, 2, 1)).astype(np.float32),   # (128, 64)
            np.log2(1 + pool(da, 2, 1)).astype(np.float32),   # (128, 64)
            np.log2(1 + pool(rd, 2, 4)).astype(np.float32))   # (128, 64)


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
            return None
    except Exception:
        return None
    ra, da, rd = project(raw)
    for k, m in zip(('RA', 'DA', 'RD'), (ra, da, rd)):
        np.save(outs[k], m)
    return f'./unit1/radar_RA/{base}_RA.npy'


def prep(sc):
    root = f'dataset/scenario{sc}'
    df = pd.read_csv(f'{root}/scenario{sc}_dev.csv')
    for k in ('RA', 'DA', 'RD'):
        os.makedirs(f'{root}/unit1/radar_{k}', exist_ok=True)
    with ProcessPoolExecutor(max_workers=32) as ex:
        ra_paths = list(ex.map(one, [(root, r) for r in df['unit1_radar']], chunksize=16))
    bad = sum(p is None for p in ra_paths)
    df['ra_path'] = ra_paths
    print(f'S{sc}: {len(df)-bad}/{len(df)} radar maps written ({bad} unreadable)')

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
    return root, per_pass


def emit(root, name, passes, per_pass):
    rows = [r for p in passes for r in per_pass[p]]
    # DataFeed's create_samples slices label RANGES ('beam1':'beam8'), so the column
    # order must exactly match S9's grouped layout -- interleaved columns would leak
    # camera paths into the beam slice.
    cols = ([f'camera{i}' for i in range(1, L + 1)]
            + [f'radar{i}' for i in range(1, L + 1)]
            + [f'beam{i}' for i in range(1, L + 1)]
            + [f'future_beam{j}' for j in range(1, J + 1)]
            + ['seq_index'])
    pd.DataFrame(rows)[cols].to_csv(f'{root}/{name}', index=False)
    print(f'  {name}: {len(rows)} sequences from {len(passes)} passes')


root32, pp32 = prep(32)
p32 = sorted(pp32)
emit(root32, 'train_seqs_RA.csv', p32[:10], pp32)
emit(root32, 'val_seqs_RA.csv',   p32[10:12], pp32)
emit(root32, 'test_seqs_RA.csv',  p32[12:], pp32)

root33, pp33 = prep(33)
emit(root33, 'test_seqs_RA.csv', sorted(pp33), pp33)
print('PREP_DONE')
