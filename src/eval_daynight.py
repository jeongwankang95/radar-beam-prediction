#!/usr/bin/env python3
"""Day-to-night transfer at a fixed site, over the full prediction horizon.

Supplementary S7. Scenarios 32 and 33 are the same road recorded by day and by night, so
replaying the same Scenario-32-trained models on both was meant to isolate illumination from
every other factor (the paper reports the attempt as inconclusive). All seven predicted slots
are scored.

Self-check: the Scenario-32 vision seed-42 run must reproduce the per-slot Top-1
already stored in cmkd_results/_s32vis_s42/eval_results.npz.

Usage: eval_daynight.py
"""
import numpy as np, pandas as pd, os, sys, glob, torch
import torchvision.transforms as transf
from torch.utils.data import DataLoader
from MyFunc import set_seed, calculate_dba_score
from model_cmkd import RadarOnlyStudent
from model_vision_only import VisionOnlyStudent
from DataFeed import DataFeed

DS = 'dataset'          # run from the repository root
CR = 'cmkd_results'
set_seed(42); torch.set_grad_enabled(False)
dev = torch.device('cpu')
proc = transf.Compose([transf.ToPILImage(), transf.Resize((224, 224))])
SEEDS = [42, 1, 2]


def ckpt(arm, seed):
    if arm == 'vision':
        return f'{CR}/_s32vis_s{seed}/vision_best.pth'
    g = sorted(glob.glob(f'{CR}/_s32rd_s{seed}/phase0_*/radar_baseline.pth'))
    return g[0] if g else None


def run(arm, seed, root, seq_csv):
    """Full-horizon logits for one arm/seed on one scenario."""
    if arm == 'radar':
        m = RadarOnlyStudent(64, 64, 64, gru_layers=1, num_heads_mha=8,
                             radar_h=128, radar_w=64, radar_channels=2)
        second = 'RD'
    else:
        m = VisionOnlyStudent(64, 64, 64, gru_layers=1, num_heads_mha=8)
        second = 'DA'
    ck = ckpt(arm, seed)
    if ck is None or not os.path.exists(ck):
        return None
    sd = torch.load(ck, map_location='cpu')
    m.load_state_dict(sd.get('model', sd) if isinstance(sd, dict) else sd, strict=True)
    m.eval().to(dev)
    ld = DataLoader(DataFeed(root, seq_csv, seq_len=8, transform=proc,
                             clipped_range=128, second_map=second),
                    batch_size=64, shuffle=False, num_workers=0)
    LG, LB = [], []
    for img, ra, x2, beam, label in ld:
        B = ra.shape[0]
        if arm == 'radar':
            r = torch.cat([torch.stack([ra[:, 1:], x2[:, 1:]], 2),
                           torch.zeros(B, 7, 2, 128, 64)], 1).to(dev)
            lg = m(r, 7)[0]
        else:
            v = torch.cat([img.unsqueeze(2), torch.zeros(B, 7, 1, 224, 224)], 1).to(dev)
            lg = m(v, 7)[0]
        # the seven predicted slots are the current one plus six future ones; the
        # current slot's ground truth is the last observed beam (beam8)
        LG.append(lg[:, -7:, :].cpu())
        LB.append(torch.cat([beam[:, -1:].cpu(), label.cpu()], dim=1))
    return torch.cat(LG), torch.cat(LB)


def score(lg, lb):
    """Per-slot Top-1/Top-5, their horizon averages, and ADBA."""
    top1 = (lg.argmax(-1) == lb).float().mean(0).numpy() * 100
    t5 = lg.topk(5, dim=-1).indices
    top5 = (t5 == lb.unsqueeze(-1)).any(-1).float().mean(0).numpy() * 100
    dba = calculate_dba_score(lg, lb, delta=5)      # returns 1 - normalized distance
    return top1, top5, np.asarray(dba, dtype=float) * 100


SCEN = [('day  (S32 test)', f'{DS}/scenario32', f'{DS}/scenario32/test_seqs_RA.csv'),
        ('night(S33 all )', f'{DS}/scenario33', f'{DS}/scenario33/test_seqs_RA.csv')]

print('Same Scenario-32-trained models, replayed on the same road by day and by night.')
print('All figures are percentages, averaged over the seven predicted slots.\n')
print(f"{'arm':7s} {'scenario':16s} {'ATop-1':>7s} {'ATop-5':>7s} {'ADBA':>7s}   per-slot Top-1")
res = {}
for arm in ('vision', 'radar'):
    for tag, root, csv in SCEN:
        a1, a5, ad = [], [], []
        for sd_ in SEEDS:
            out = run(arm, sd_, root, csv)
            if out is None:
                print(f'  {arm} seed {sd_}: checkpoint missing'); continue
            t1, t5, db = score(*out)
            a1.append(t1); a5.append(t5); ad.append(db)
            if arm == 'vision' and sd_ == 42 and tag.startswith('day'):
                ref = np.load(f'{CR}/_s32vis_s42/eval_results.npz')['top1']
                ref = ref * (100 if ref.max() <= 1.5 else 1)
                ok = np.allclose(t1, ref, atol=0.6)
                print(f'  [self-check] S32 vision seed42 per-slot Top-1 vs stored: '
                      f'{"MATCH" if ok else "MISMATCH"}')
                if not ok:
                    print('     mine  ', np.round(t1, 2))
                    print('     stored', np.round(ref, 2))
        if not a1: continue
        A1, A5, AD = np.mean(a1, 0), np.mean(a5, 0), np.mean(ad, 0)
        res[(arm, tag)] = (A1, A5, AD)
        print(f'{arm:7s} {tag:16s} {A1.mean():7.1f} {A5.mean():7.1f} {AD.mean():7.1f}   '
              + ' '.join(f'{v:4.1f}' for v in A1))

print('\n--- day -> night change (percentage points, averaged over seeds and slots) ---')
for arm in ('vision', 'radar'):
    k_d, k_n = (arm, SCEN[0][0]), (arm, SCEN[1][0])
    if k_d in res and k_n in res:
        d, n = res[k_d], res[k_n]
        print(f'  {arm:7s} ATop-1 {d[0].mean():5.1f} -> {n[0].mean():5.1f} '
              f'({n[0].mean()-d[0].mean():+5.1f})   '
              f'ADBA {d[2].mean():5.1f} -> {n[2].mean():5.1f} ({n[2].mean()-d[2].mean():+5.1f})')
np.savez('cmkd_results/_daynight.npz',
         **{f'{a}_{t.split()[0]}_{m}': v for (a, t), r in res.items()
            for m, v in zip(('top1', 'top5', 'dba'), r)})
print('\nEVAL_DAYNIGHT_DONE')
