#!/usr/bin/env python3
"""Tail of the measured beam-selection loss per predictor (Scenario 9), from the trained checkpoints.

The per-window, per-slot loss of every seed is kept (cmkd_results/_power_tail.npz) so that tail
metrics and the paper's cluster bootstrap (28 test passes x seeds, seeds drawn unpaired) can be
computed on top. Two self-checks run before anything is reported: every arm's forward ATop-1 is
compared with the eval_results.npz written at training time (guards the channel order), and the
mean dB of every reported row is compared with the paper's Table II. The analysis scripts use the
prediction dumps in results/ instead (analysis/table2.py).
"""
import numpy as np, pandas as pd, os, glob, time, json, torch
import torchvision.transforms as transf
from torch.utils.data import DataLoader
from MyFunc import set_seed
from model_cmkd import RadarOnlyStudent, MultimodalTeacher
from model_vision_only import VisionOnlyStudent
from DataFeed import DataFeed

ROOT = 'dataset/scenario9'
set_seed(42); torch.set_grad_enabled(False)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEV == 'cpu':
    torch.set_num_threads(max(8, (os.cpu_count() or 16) // 2))
print('device', DEV, flush=True)
T0 = time.time()
seq = pd.read_csv(f'{ROOT}/test_seqs_RA.csv')
PASS = seq['seq_index'].to_numpy()
SLOT_COLS = ['beam8'] + [f'future_beam{j}' for j in range(1, 7)]
POW = np.stack([[np.loadtxt(os.path.join(ROOT, str(seq.iloc[i][c]).lstrip('./'))) for c in SLOT_COLS]
                for i in range(len(seq))])
OPT = POW.argmax(-1)
BEST = np.take_along_axis(POW, OPT[..., None], -1).squeeze(-1)
N = len(seq)
print(f'{N} windows, {len(np.unique(PASS))} passes, power loaded in {time.time()-T0:.0f}s', flush=True)

proc = transf.Compose([transf.ToPILImage(), transf.Resize((224, 224))])
_CACHE = {}


def batches(second_map, extra_map=''):
    key = (second_map, extra_map)
    if key not in _CACHE:
        ld = DataLoader(DataFeed(ROOT, f'{ROOT}/test_seqs_RA.csv', seq_len=8, transform=proc,
                                 clipped_range=128, second_map=second_map, extra_map=extra_map),
                        batch_size=32, shuffle=False, num_workers=8)
        t = time.time(); _CACHE[key] = [b for b in ld]
        print(f'  cached loader {key} in {time.time()-t:.0f}s', flush=True)
    return _CACHE[key]


def predictions(kind, ckpt, second_map, chan, kw, extra_map=''):
    ck = sorted(glob.glob(ckpt))
    if not ck:
        return None
    m = kind(**kw)
    sd = torch.load(ck[-1], map_location='cpu')
    m.load_state_dict(sd.get('model', sd) if isinstance(sd, dict) else sd, strict=True)
    m.to(DEV).eval()
    out, tgt = [], []
    for img, ra, x2, beam, label in batches(second_map, extra_map):
        B = ra.shape[0]
        if chan == 'both':
            r = torch.stack([ra[:, 1:], x2[:, 1:]], 2)
        elif chan == 'second':
            r = x2[:, 1:].unsqueeze(2)
        elif chan == 'three':                       # x2 is (B, L, 2, H, W): second and third maps
            r = torch.cat([ra[:, 1:].unsqueeze(2), x2[:, 1:]], 2)
        else:
            r = ra[:, 1:].unsqueeze(2)
        C = r.shape[2]
        radar = torch.cat([r, torch.zeros(B, 7, C, 128, 64)], 1).to(DEV)
        vis = torch.cat([img.unsqueeze(2), torch.zeros(B, 7, 1, 224, 224)], 1).to(DEV)
        if kind is VisionOnlyStudent:
            lg = m(vis, 7)[0]
        elif kind is MultimodalTeacher:
            lg = m(vis, radar, 7)[0]
        else:
            lg = m(radar, 7)[0]
        out.append(lg[:, -7:].argmax(-1).cpu())
        tgt.append(torch.cat([beam[:, -1:], label[:, :6]], -1))
    return torch.cat(out).numpy(), torch.cat(tgt).numpy()


def stored_top1(tag):
    f = sorted(glob.glob(f'cmkd_results/{tag}/*/eval_results.npz') + glob.glob(f'cmkd_results/{tag}/eval_results.npz'))
    return 100 * float(np.mean(np.load(f[-1])['top1'])) if f else np.nan


S = dict(feature_size=64, num_classes=64, gru_hidden=64, gru_layers=1, num_heads_mha=8, radar_h=128, radar_w=64)
V = dict(feature_size=64, num_classes=64, gru_hidden=64, gru_layers=1, num_heads_mha=8)
T = dict(feature_size=64, num_classes=64, gru_hidden=64, gru_layers=2, num_heads_ca=4, num_heads_mha=8, radar_h=128, radar_w=64)
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
RA10 = ['_v3_noKD_RA', '_v3s1_noKD_RA', '_v3s2_noKD_RA'] + [f'_c1ra_s{s}' for s in range(3, 10)]
DA10 = ['_v3_noKD_RADA', '_v3s1_noKD_RADA', '_v3s2_noKD_RADA'] + [f'_c1da_s{s}' for s in range(3, 10)]
#        name                   class              dirs                                       leaf                             map  chan     kwargs                     Table II mean dB  extra
ARMS = [
    ('Radar, RA only',      RadarOnlyStudent,  RA10,                                     'phase0_*/radar_baseline.pth',    'DA', 'first',  S | dict(radar_channels=1), 0.91, ''),
    ('Radar, DA only',      RadarOnlyStudent,  [f'_m6DAonly_s{s}' for s in SEEDS],       'phase0_*/radar_baseline.pth',    'DA', 'second', S | dict(radar_channels=1), 0.67, ''),
    ('Radar, RD only',      RadarOnlyStudent,  [f'_m6RDonly_s{s}' for s in SEEDS],       'phase0_*/radar_baseline.pth',    'RD', 'second', S | dict(radar_channels=1), 0.75, ''),
    ('Radar, RA+DA no KD',  RadarOnlyStudent,  DA10,                                     'phase0_*/radar_baseline.pth',    'DA', 'both',   S | dict(radar_channels=2), 0.65, ''),
    ('Radar, RA+RD no KD',  RadarOnlyStudent,  [f'_v3rd_s{s}' for s in SEEDS],           'phase0_*/radar_baseline.pth',    'RD', 'both',   S | dict(radar_channels=2), 0.64, ''),
    ('Radar, RA+DA+RD',     RadarOnlyStudent,  [f'_m4_3ch_s{s}' for s in SEEDS],         'phase0_*/radar_baseline.pth',    'DA', 'three',  S | dict(radar_channels=3), 0.61, 'RD'),
    ('Radar, RA+RD +KD',    RadarOnlyStudent,  [f'_v3rdkd_s{s}' for s in SEEDS],         'phase3_*/student_cmkd_best.pth', 'RD', 'both',   S | dict(radar_channels=2), 0.64, ''),
    ('Vision-only',         VisionOnlyStudent, [f'_v3_visiononly_s{s}' for s in SEEDS],  'vision_best.pth',                'DA', None,     V,                          0.44, ''),
    ('Teacher (V+R)',       MultimodalTeacher, ['_v3_teacher'] + [f'_v3ca_s{s}' for s in range(1, 10)], 'phase2_*/teacher_selfkd.pth', 'DA', 'both', T,           0.41, ''),
    ('  +feature KD',       RadarOnlyStudent,  [f'_v3rd_feat_s{s}' for s in SEEDS],      'phase3_*/student_cmkd_best.pth', 'RD', 'both',   S | dict(radar_channels=2), None, ''),
    ('  +feat+temp KD',     RadarOnlyStudent,  [f'_v3rd_full_s{s}' for s in SEEDS],      'phase3_*/student_cmkd_best.pth', 'RD', 'both',   S | dict(radar_channels=2), None, ''),
]
THR = (1.0, 3.0, 6.0)


def stat(v, k):
    if k == 'mean': return float(v.mean())
    if k == 'median': return float(np.median(v))
    if k.startswith('p'): return float(np.percentile(v, int(k[1:])))
    if k.startswith('gt'): return float(100 * (v > float(k[2:])).mean())
    raise KeyError(k)


KEYS = ['mean', 'median', 'gt1', 'gt3', 'gt6', 'p90', 'p95', 'p99']
LOSS, SUMMARY, ACC = {}, {}, {}
print('\n' + '=' * 132)
print(f"{'predictor':22s} {'n':>2s} {'ATop-1 fwd':>10s} {'stored':>7s} | {'mean dB':>12s} {'TabII':>5s} | {'median':>6s} {'P>1dB%':>7s} {'P>3dB%':>7s} {'P>6dB%':>7s} {'p90':>5s} {'p95':>5s} {'p99':>6s}")
print('-' * 132)
for name, cls, tags, leaf, sm, chan, kw, pub, extra in ARMS:
    per, acc, used, flags = [], [], [], []
    for t in tags:
        r = predictions(cls, f'cmkd_results/{t}/{leaf}', sm, chan, kw, extra)
        if r is None:
            continue
        pred, tgt = r
        if used == []:
            assert (tgt == OPT).mean() > 0.999, f'labels != power argmax for {name}'
        cap = np.take_along_axis(POW, pred[..., None], -1).squeeze(-1) / (BEST + 1e-12)
        L = (-10 * np.log10(np.clip(cap, 1e-6, None))).astype(np.float32)
        a_f, a_s = 100 * (pred == tgt).mean(), stored_top1(t)
        if np.isfinite(a_s) and abs(a_f - a_s) > 0.05:
            flags.append(f'{t}:{a_f:.2f}vs{a_s:.2f}')
        per.append(L); acc.append((a_f, a_s)); used.append(t)
    if not per:
        print(f'{name:22s}  (no runs found)'); continue
    LS = np.stack(per)                                   # (n_seeds, N, 7)
    LOSS[name] = LS; ACC[name] = np.array(acc)
    ms = {k: np.array([stat(l.ravel(), k) for l in LS]) for k in KEYS}
    SUMMARY[name] = {k: (float(ms[k].mean()), float(ms[k].std(ddof=1)) if len(LS) > 1 else 0.0) for k in KEYS}
    SUMMARY[name]['n'] = len(LS); SUMMARY[name]['runs'] = used
    SUMMARY[name]['atop1_fwd'] = float(ACC[name][:, 0].mean()); SUMMARY[name]['atop1_stored'] = float(np.nanmean(ACC[name][:, 1]))
    chk = ('' if pub is None else ('  OK' if abs(ms['mean'].mean() - pub) < 0.015 else f'  <-- DIFF {ms["mean"].mean()-pub:+.3f}'))
    print(f"{name:22s} {len(LS):2d} {ACC[name][:,0].mean():10.2f} {np.nanmean(ACC[name][:,1]):7.2f} | "
          f"{ms['mean'].mean():5.2f}+-{ms['mean'].std(ddof=1):4.2f} {('' if pub is None else f'{pub:5.2f}'):>5s} | "
          f"{ms['median'].mean():6.2f} {ms['gt1'].mean():7.2f} {ms['gt3'].mean():7.2f} {ms['gt6'].mean():7.2f} "
          f"{ms['p90'].mean():5.2f} {ms['p95'].mean():5.2f} {ms['p99'].mean():6.2f}{chk}", flush=True)
    if flags:
        print('    forward-vs-stored ATop-1 mismatch > 0.05 pp:', ', '.join(flags), flush=True)
print('-' * 132)
print('  pooled over 774 windows x 7 slots per seed, then mean over seeds (+-std over seeds for the mean).')
print(f'  forward passes done in {time.time()-T0:.0f}s', flush=True)

np.savez_compressed('cmkd_results/_power_tail.npz', pass_id=PASS, opt=OPT,
                    **{f'{n}::{t}': LOSS[n][i] for n in LOSS for i, t in enumerate(SUMMARY[n]['runs'])})
print('  saved cmkd_results/_power_tail.npz', flush=True)

# ---- cluster bootstrap on deltas: 28 passes with replacement (whole passes), seeds drawn unpaired per arm
rng = np.random.default_rng(42)
passes = np.unique(PASS); widx = [np.where(PASS == p)[0] for p in passes]
PAIRS = [('Doppler: RA only -> RA+RD',        'Radar, RA only',     'Radar, RA+RD no KD'),
         ('KD: RA+RD -> RA+RD +KD',            'Radar, RA+RD no KD', 'Radar, RA+RD +KD'),
         ('camera removal: Vision -> RA+RD',   'Vision-only',        'Radar, RA+RD no KD'),
         ('teacher: Vision -> Teacher',        'Vision-only',        'Teacher (V+R)'),
         ('pairing: DA only -> RA+RD',         'Radar, DA only',     'Radar, RA+RD no KD'),
         ('third map: RA+RD -> RA+DA+RD',      'Radar, RA+RD no KD', 'Radar, RA+DA+RD')]
BKEYS = ['mean', 'gt1', 'gt3', 'gt6', 'p95', 'p99']
NB = int(os.environ.get('NBOOT', '20000'))
print(f'\n{"=" * 132}\nDELTAS (second minus first; negative = less loss), paired point estimate over seeds and {NB}-replicate cluster bootstrap [2.5%, 97.5%]\n{"=" * 132}')
BOOT = {}
for label, a, b in PAIRS:
    if a not in LOSS or b not in LOSS:
        print(f'{label}: missing arm'); continue
    LA, LB = LOSS[a], LOSS[b]
    n = min(len(LA), len(LB))
    point = {k: float(np.mean([stat(LB[i].ravel(), k) - stat(LA[i].ravel(), k) for i in range(n)])) for k in BKEYS}
    t = time.time(); acc = {k: np.empty(NB) for k in BKEYS}
    for r in range(NB):
        idx = np.concatenate([widx[p] for p in rng.integers(0, len(passes), len(passes))])
        va = LA[rng.integers(0, len(LA), len(LA))][:, idx].ravel()
        vb = LB[rng.integers(0, len(LB), len(LB))][:, idx].ravel()
        for k in BKEYS:
            acc[k][r] = stat(vb, k) - stat(va, k)
    ci = {k: (float(np.percentile(acc[k], 2.5)), float(np.percentile(acc[k], 97.5))) for k in BKEYS}
    BOOT[label] = dict(point=point, ci=ci, arms=(a, b), n=(len(LA), len(LB)))
    print(f'{label}   ({len(LA)} vs {len(LB)} seeds, {time.time()-t:.0f}s)')
    for k in BKEYS:
        unit = 'dB' if k in ('mean', 'p95', 'p99') else 'pp of slots'
        print(f"    {k:>5s}: {point[k]:+7.3f}  [{ci[k][0]:+7.3f}, {ci[k][1]:+7.3f}]  {unit}", flush=True)
json.dump(dict(summary=SUMMARY, boot=BOOT, nboot=NB, device=DEV),
          open('cmkd_results/_power_tail_summary.json', 'w'), indent=1)
print(f'\nsaved cmkd_results/_power_tail_summary.json   total {time.time()-T0:.0f}s\nPOWER_TAIL_DONE')
