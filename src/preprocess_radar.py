#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_radar.py: Preprocess raw Scenario 9 radar data

Reads raw .mat radar files → computes Range-Angle and Doppler-Angle maps → saves as .npy
Also creates scenario9_RA.csv and scenario9_DA.csv with updated radar paths.

Usage (from the repository root):
    python src/preprocess_radar.py
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

# ── Radar FFT parameters (from Radar_KPI.py) ──
FFT_ANGLE = 64
FFT_RANGE = 256
FFT_VELOCITY = 128


def radar_cube(raw_data, fft_tuple=(FFT_ANGLE, FFT_RANGE, FFT_VELOCITY), remove_mean=True):
    """
    raw_data: complex array of shape (N_rx, N_samples, N_chirps) = (4, 256, 128)
    Returns: radar cube after 3D FFT, shape (fft_angle, fft_range, fft_velocity)
    """
    fft_angle, fft_range, fft_velocity = fft_tuple
    # Range FFT
    range_DFT = np.fft.fft(raw_data, n=fft_range, axis=1)
    # Remove DC offset (static objects)
    if remove_mean:
        range_DFT = range_DFT - np.mean(range_DFT, axis=2, keepdims=True)
    # Doppler FFT
    doppler_DFT = np.fft.fft(range_DFT, n=fft_velocity, axis=2)
    # Angle FFT
    angle_DFT = np.fft.fft(doppler_DFT, n=fft_angle, axis=0)
    cube = np.fft.fftshift(angle_DFT, axes=(0, 2))
    return cube


def range_angle_map(cube, log_scale=True):
    """
    Averages over velocity axis → Range-Angle map
    Returns: (fft_range, fft_angle)
    """
    ra = np.mean(np.abs(cube), axis=2).T  # (fft_range, fft_angle)
    if log_scale:
        ra = np.log2(1 + ra)
    return ra


def doppler_angle_map(cube, log_scale=True):
    """
    Averages over range axis → Doppler-Angle map
    Returns: (fft_velocity, fft_angle)
    """
    da = np.mean(np.abs(cube), axis=1).T  # (fft_velocity, fft_angle)
    if log_scale:
        da = np.log2(1 + da)
    return da


def process_single_mat(mat_path):
    """
    Load a single .mat file and return RA and DA maps.
    """
    mat = loadmat(mat_path)
    # The key varies; common keys: 'data', 'radar_data', or first non-meta key
    data_key = None
    for key in mat:
        if not key.startswith('__'):
            data_key = key
            break
    if data_key is None:
        raise ValueError(f"No data key found in {mat_path}")

    raw = mat[data_key]  # expected shape: (4, 256, 128) or similar

    # Handle different possible shapes
    if raw.ndim == 2:
        # Sometimes data is stored as (N_samples * N_rx, N_chirps); reshape
        # Try (4, 256, 128)
        try:
            raw = raw.reshape(4, 256, 128)
        except ValueError:
            raise ValueError(f"Unexpected shape {raw.shape} in {mat_path}")

    cube = radar_cube(raw)
    ra = range_angle_map(cube)
    da = doppler_angle_map(cube)
    return ra, da


def main():
    # ── Paths ──
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
    data_root = os.path.join(script_dir, 'dataset', 'scenario9')
    csv_path = os.path.join(data_root, 'scenario9.csv')

    # Output directories for preprocessed radar
    ra_dir = os.path.join(data_root, 'unit1', 'radar_RA')
    da_dir = os.path.join(data_root, 'unit1', 'radar_DA')
    os.makedirs(ra_dir, exist_ok=True)
    os.makedirs(da_dir, exist_ok=True)

    # ── Read CSV ──
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # ── Process each radar file ──
    radar_ra_paths = []
    radar_da_paths = []
    errors = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing radar"):
        radar_rel_path = row['unit1_radar']  # e.g. ./unit1/radar_data/data_4589_23_45_05.mat
        mat_path = os.path.join(data_root, radar_rel_path.lstrip('./'))

        # Output filenames
        base_name = os.path.splitext(os.path.basename(radar_rel_path))[0]
        ra_filename = f"{base_name}_RA.npy"
        da_filename = f"{base_name}_DA.npy"
        ra_path = os.path.join(ra_dir, ra_filename)
        da_path = os.path.join(da_dir, da_filename)

        # Relative paths for CSV
        ra_rel = f"./unit1/radar_RA/{ra_filename}"
        da_rel = f"./unit1/radar_DA/{da_filename}"

        if not os.path.exists(ra_path):
            try:
                ra, da = process_single_mat(mat_path)
                np.save(ra_path, ra.astype(np.float32))
                np.save(da_path, da.astype(np.float32))
            except Exception as e:
                errors.append((idx, mat_path, str(e)))
                ra_rel = "ERROR"
                da_rel = "ERROR"

        radar_ra_paths.append(ra_rel)
        radar_da_paths.append(da_rel)

    # ── Create new CSVs ──
    # scenario9_RA.csv: replace unit1_radar with RA path
    df_ra = df.copy()
    df_ra['unit1_radar'] = radar_ra_paths
    ra_csv_path = os.path.join(data_root, 'scenario9_RA.csv')
    df_ra.to_csv(ra_csv_path, index=False)
    print(f"\nSaved: {ra_csv_path}")

    # scenario9_DA.csv: replace unit1_radar with DA path
    df_da = df.copy()
    df_da['unit1_radar'] = radar_da_paths
    da_csv_path = os.path.join(data_root, 'scenario9_DA.csv')
    df_da.to_csv(da_csv_path, index=False)
    print(f"Saved: {da_csv_path}")

    if errors:
        print(f"\n⚠ {len(errors)} files had errors:")
        for idx, path, err in errors[:10]:
            print(f"  Row {idx}: {path} → {err}")
    else:
        print("\n✓ All files processed successfully!")

    # ── Verify ──
    sample_ra = np.load(os.path.join(ra_dir, os.listdir(ra_dir)[0]))
    sample_da = np.load(os.path.join(da_dir, os.listdir(da_dir)[0]))
    print(f"\nRange-Angle map shape: {sample_ra.shape}")    # expected: (256, 64)
    print(f"Doppler-Angle map shape: {sample_da.shape}")    # expected: (128, 64)


if __name__ == '__main__':
    main()
