#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_radar_rd.py: range-Doppler maps, the second radar map of the reference RA+RD student
(the Doppler-angle maps come from preprocess_radar.py).

Both pairings keep one axis shared with the range-angle map:
    RA  = (range 128, angle    64)   ]- stacked on the angle axis   (RA+DA)
    DA  = (velocity 128, angle 64)   ]
    RA  = (range 128, angle    64)   ]- stacked on the range axis   (RA+RD)
    RD  = (range 128, velocity 64)   ]

The velocity axis is decimated 128->64 by pair-averaging (not cropping: at 0.84 km/h
per bin the unambiguous span is +-54 km/h, and cropping to the central 64 bins would
discard the vehicle speeds of interest). Decimation preserves the full span at half
resolution. The angle axis is natively 64 bins, so the Doppler-angle map keeps its full
velocity resolution while the range-Doppler map's is halved.

Writes dataset/scenario9/unit1/radar_RD/<base>_RD.npy, matching the _RA/_DA naming so
DataFeed's path substitution picks them up with --second_map RD.
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from preprocess_radar import radar_cube, FFT_ANGLE, FFT_RANGE, FFT_VELOCITY

CLIPPED_RANGE = 128     # same near-range clipping the RA map receives in DataFeed
OUT_VELOCITY = 64       # match the RA map's second-axis width so channels stack


def range_doppler_map(cube, log_scale=True):
    """cube (angle, range, velocity) -> (CLIPPED_RANGE, OUT_VELOCITY).

    Average over angle (the axis this representation gives up), clip range exactly as
    the RA map is clipped, then decimate velocity by pair-averaging.
    """
    rd = np.mean(np.abs(cube), axis=0)          # (range, velocity)
    rd = rd[:CLIPPED_RANGE, :]                  # (128, 128)
    v = rd.shape[1] // OUT_VELOCITY             # 2
    rd = rd.reshape(CLIPPED_RANGE, OUT_VELOCITY, v).mean(axis=2)   # (128, 64)
    if log_scale:
        rd = np.log2(1 + rd)
    return rd


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
    data_root = os.path.join(root, 'dataset', 'scenario9')
    rd_dir = os.path.join(data_root, 'unit1', 'radar_RD')
    os.makedirs(rd_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(data_root, 'scenario9.csv'))
    print(f"samples: {len(df)}   ->  {rd_dir}")

    errors = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="range-Doppler"):
        rel = row['unit1_radar']
        base = os.path.splitext(os.path.basename(rel))[0]
        out = os.path.join(rd_dir, f"{base}_RD.npy")
        if os.path.exists(out):
            continue
        try:
            mat = loadmat(os.path.join(data_root, rel.lstrip('./')))
            key = next(k for k in mat if not k.startswith('__'))
            raw = mat[key]
            if raw.ndim == 2:
                raw = raw.reshape(4, FFT_RANGE, FFT_VELOCITY)
            np.save(out, range_doppler_map(radar_cube(raw)).astype(np.float32))
        except Exception as e:
            errors.append((rel, str(e)))

    n = len(os.listdir(rd_dir))
    print(f"\nwrote {n} files")
    if errors:
        print(f"{len(errors)} errors:")
        for rel, e in errors[:5]:
            print(f"  {rel}: {e}")
    sample = np.load(os.path.join(rd_dir, sorted(os.listdir(rd_dir))[0]))
    print(f"RD map shape {sample.shape}  (expect (128, 64))"
          f"   range [{sample.min():.2f}, {sample.max():.2f}]")


if __name__ == '__main__':
    main()
