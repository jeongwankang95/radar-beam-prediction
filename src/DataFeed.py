import os as _os
_SCRAMBLE = _os.environ.get('SCRAMBLE_DOPPLER', '0') == '1'
_MAPTF = _os.environ.get('MAP_TRANSFORM', 'none')
_BINQ = float(_os.environ.get('MAP_BIN_Q', '0.90'))
import numpy as _np

_LASTF = _os.environ.get('LAST_FRAME_ONLY', '0') == '1'
_RAV = _os.environ.get('RA_VARIANT', '')

def _moment_map(m, mode):
    """Moment-map controls on the Doppler-angle map (rows = Doppler, columns = angle).
    Per column: floor = median over rows, w = max(m - floor, 0), total = sum(w),
    centroid c and standard deviation s of w over the row index.
      dadelta  : total placed in the single row nearest c   (keeps total + centroid)
      dagauss  : total spread as N(c, max(s, 0.5))          (keeps total, centroid, spread)
      daspread : total spread as N(H/2, max(s, 0.5))        (keeps total + spread, kills centroid)
    Pre-specified in prespecification/MECHANISM_PRESPEC_2026-09-27.md."""
    m = _np.asarray(m, dtype=_np.float64)
    H = m.shape[0]
    keep_floor = mode.endswith('fl'); mode = mode[:-2] if keep_floor else mode
    floor = _np.median(m, axis=0, keepdims=True)
    w = _np.maximum(m - floor, 0.0)
    tot = w.sum(axis=0)
    rows = _np.arange(H, dtype=_np.float64)[:, None]
    safe = _np.maximum(tot, 1e-12)
    cen = (w * rows).sum(axis=0) / safe
    std = _np.sqrt(_np.maximum((w * (rows - cen[None, :]) ** 2).sum(axis=0) / safe, 0.0))
    out = _np.zeros_like(m)
    if mode == 'dadelta':
        r = _np.clip(_np.rint(cen), 0, H - 1).astype(int)
        out[r, _np.arange(m.shape[1])] = tot
    else:
        mu = cen if mode == 'dagauss' else _np.full_like(cen, (H - 1) / 2.0)
        sd = _np.maximum(std, 0.5)
        k = _np.exp(-0.5 * ((rows - mu[None, :]) / sd[None, :]) ** 2)
        k /= _np.maximum(k.sum(axis=0, keepdims=True), 1e-12)
        out = k * tot[None, :]
    out[:, tot <= 0] = 0.0
    if keep_floor:
        out = out + floor          # '...fl' variants add the column floor back (as the collapse control keeps it)
    return out.astype(_np.float32)
_SCR_RNG = None
def _scr_rng():
    global _SCR_RNG
    if _SCR_RNG is None:
        import torch.utils.data as _tud
        _w = _tud.get_worker_info()
        _SCR_RNG = _np.random.default_rng(
            int(_os.environ.get('SCRAMBLE_SEED', '1234')) + 7919 * (_w.id if _w else 0))
    return _SCR_RNG
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@file: DataFeed.py
@time: 2025/12/22 12:30
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transf
from skimage import io
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from Radar_KPI import *
from scipy.io import loadmat

def create_samples(root, portion=1.):
    f = pd.read_csv(root, na_values='')
    f = f.fillna(-99)
    Total_Num = len(f)
    num_data = int(Total_Num * portion)
    data_samples_rgb = []
    data_samples_radar = []
    pred_beam = []
    inp_beam = []
    for idx, row in f.head(num_data).iterrows():
        vision_data = row['camera1':'camera8'].tolist()
        data_samples_rgb.append(vision_data)
        radar_data = row['radar1':'radar8'].tolist()
        data_samples_radar.append(radar_data)

        # Dynamic approach: get all future_beam columns
        future_beam_cols = [col for col in f.columns if col.startswith('future_beam')]
        future_beam_cols.sort()  # Ensure consistent ordering (future_beam1, future_beam2, etc.)
        future_beam = row[future_beam_cols].tolist()
        pred_beam.append(future_beam)

        input_beam = row['beam1':'beam8'].tolist()

        inp_beam.append(input_beam)

    # print('list is ready')
    return data_samples_rgb, data_samples_radar, inp_beam, pred_beam


class DataFeed(Dataset):
    def __init__(self, data_root, root_csv, seq_len=8, transform=None,
    fft_tuple=(64, 256,128), clipped_range=128, portion=1., second_map='DA',
    extra_map=''):

        self.data_root = data_root
        self.samples_rgb, self.samples_radar, self.inp_val, self.pred_val = create_samples(root_csv, portion=portion)
        self.seq_len = seq_len
        self.transform = transform
        self.fft_tuple = fft_tuple
        self.clipped_range = clipped_range
        # Second radar map paired with the range-angle map: 'DA' (Doppler-angle) or
        # 'RD' (range-Doppler, the reference student's). Both are (128, 64)
        # and resolved from the RA path by substitution, so only the suffix changes.
        self.second_map = second_map
        self.extra_map = extra_map


    def __len__(self):
        return len(self.samples_rgb)

    def __getitem__(self, idx):
        samples_rgb = self.samples_rgb[idx]
        samples_radar = self.samples_radar[idx]
        beam_val = self.pred_val[idx]
        input_beam = self.inp_val[idx]


        # out_beam = torch.zeros((3,))
        image_val = np.zeros((self.seq_len, 224,224))
        image_dif = np.zeros((self.seq_len-1, 224, 224))
        image_motion_masks = np.zeros((self.seq_len - 1, 224, 224))

        beam_past = []
        clipped_range = self.clipped_range

        radar_val_range_angle = np.zeros((self.seq_len, clipped_range, self.fft_tuple[0]))
        radar_val_doppler_angle = np.zeros((self.seq_len, self.fft_tuple[2], self.fft_tuple[0]))
        radar_dif_RA = np.zeros((self.seq_len - 1, clipped_range, self.fft_tuple[0]))
        radar_dif_DA = np.zeros((self.seq_len - 1, self.fft_tuple[2], self.fft_tuple[0]))
        radar_val_extra = (np.zeros((self.seq_len, self.fft_tuple[2], self.fft_tuple[0]))
                           if self.extra_map else None)

        def _p(rel_path):
            # CSV paths start with '/', so join safely without duplicating separators
            return os.path.join(self.data_root, rel_path.lstrip("/"))

        for i, (smp_rgb_path,smp_radar_path) in enumerate(zip(samples_rgb,samples_radar)):
            beam_past_i = int(np.nanargmax(np.loadtxt(_p(input_beam[i])))) # start with 0
            beam_past.append(beam_past_i)
            # Load the image
            img = self.transform(io.imread(_p(smp_rgb_path)))
            # Load the radar
            range_angle_map = np.load(_p(smp_radar_path.replace('_RA', f'_{_RAV}') if _RAV else smp_radar_path))

            range_angle_clipped = range_angle_map[:clipped_range, ...]
            smp_radar_path_DA = smp_radar_path.replace('_RA', f'_{self.second_map}')
            doppler_angle_map = np.load(_p(smp_radar_path_DA))
            if _SCRAMBLE:
                doppler_angle_map = doppler_angle_map[_scr_rng().permutation(
                    doppler_angle_map.shape[0]), :]
            if _MAPTF == 'marginal':
                _pr = doppler_angle_map.mean(axis=0, keepdims=True)
                doppler_angle_map = _np.repeat(_pr, doppler_angle_map.shape[0], axis=0)
            elif _MAPTF.startswith('pool'):
                # velocity resolution sweep: B bins along the Doppler axis, repeated
                # back to full height so the map shape and the encoder are unchanged
                _b = int(_MAPTF[4:])
                _h = doppler_angle_map.shape[0]
                assert _h % _b == 0, f'{_h} rows do not split into {_b} bins'
                _pr = doppler_angle_map.reshape(_b, _h // _b, -1).mean(axis=1)
                doppler_angle_map = _np.repeat(_pr, _h // _b, axis=0)
            elif _MAPTF == 'binary':
                _t = _np.quantile(doppler_angle_map, _BINQ)
                doppler_angle_map = (doppler_angle_map > _t).astype(doppler_angle_map.dtype)
            elif _MAPTF in ('dadelta', 'dagauss', 'daspread', 'dadeltafl', 'dagaussfl', 'daspreadfl'):
                doppler_angle_map = _moment_map(doppler_angle_map, _MAPTF)
            # # Store the smoothed image
            radar_val_range_angle[i,...] = range_angle_clipped #/np.max(smp_radar[:clipped_range, ...]+ 1e-6) # normalize the radar data
            radar_val_doppler_angle[i,...] = doppler_angle_map #/np.max(smp_radar[:clipped_range, ...]+ 1e-6) # normalize the radar data
            if self.extra_map:
                radar_val_extra[i, ...] = np.load(
                    _p(smp_radar_path.replace('_RA', f'_{self.extra_map}')))


            img = rgb2gray(img)  # Convert to grayscale

            # Apply Gaussian filtering
            img_smoothed = gaussian_filter(img, sigma=1)  # Adjust sigma for smoothing strength

            # Store the smoothed image
            image_val[i, ...] = img_smoothed

            # Compute the difference with the previous frame
            if i >= 1:
                diff = np.abs(image_val[i, ...] - image_val[i - 1, ...])
                image_dif[i - 1, ...] = diff

                # Calculate the dynamic threshold as 10% of the maximum pixel value in the difference image
                max_pixel_value = np.max(diff)
                threshold_value = 0.1 * max_pixel_value

                # Generate binary mask of significant changes
                motion_mask = (diff > threshold_value).astype(np.uint8)
                image_motion_masks[i - 1, ...] = motion_mask
                #------------------------------------below is the radar part------------------------------------
                diff_radar_RA = np.abs(radar_val_range_angle[i,...] - radar_val_range_angle[i - 1,...])
                diff_radar_DA = np.abs(radar_val_doppler_angle[i,...] - radar_val_doppler_angle[i - 1,...])
                radar_dif_RA[i - 1, ...] = diff_radar_RA
                radar_dif_DA[i - 1, ...] = diff_radar_DA
        if _LASTF:
            # single-frame control: every observed frame replaced by the last one
            radar_val_range_angle[:] = radar_val_range_angle[-1]
            radar_val_doppler_angle[:] = radar_val_doppler_angle[-1]
            if self.extra_map:
                radar_val_extra[:] = radar_val_extra[-1]
        image_masks = torch.tensor(image_motion_masks,dtype=torch.float32)

        radar_RA = torch.tensor(radar_val_range_angle,dtype=torch.float32)
        radar_DA = torch.tensor(radar_val_doppler_angle,dtype=torch.float32)
        if self.extra_map:                      # (L, 2, H, W): second and third maps
            radar_DA = torch.stack(
                [radar_DA, torch.tensor(radar_val_extra, dtype=torch.float32)], dim=1)

        beam_future = []
        for i in range(len(beam_val)):
            beam_future_i = int(np.nanargmax(np.loadtxt(_p(beam_val[i])))) 
            beam_future.append(beam_future_i)

        input_beam = torch.tensor(beam_past,dtype=torch.int64)
        out_beam = torch.tensor(beam_future,dtype=torch.int64)
        pass
        return image_masks, radar_RA, radar_DA, input_beam.long(), torch.squeeze(out_beam.long())

