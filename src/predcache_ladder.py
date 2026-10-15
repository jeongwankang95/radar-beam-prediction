#!/usr/bin/env python3
"""Test-set predictions of the transformed-input arms of paper Table III(a), the input of analysis/table3.py.

Usage, from the repository root after the arms are trained (scripts/run_grid.sh, scripts/run_m5.sh and the permuted
Doppler-angle arm of the README):
  MAP_TRANSFORM=marginal  python src/predcache_ladder.py damarg   # DA to per-angle mean    (_gridDAmarg_s*)
  MAP_TRANSFORM=marginal  python src/predcache_ladder.py rdmarg   # RD to per-velocity mean (_gridRDmarg_s*)
  MAP_TRANSFORM=binary    python src/predcache_ladder.py dabin    # DA binarized            (_gridDAbin_s*)
  SCRAMBLE_DOPPLER=1      python src/predcache_ladder.py dascr    # DA rows permuted        (_dascr_s*)
  SCRAMBLE_DOPPLER=1      python src/predcache_ladder.py rdscr    # RD rows permuted        (_m5scr_s*)

DataFeed reads the transform at import time, so it must be set in the shell; the script checks that it is. The permuted
arms draw a fresh permutation per frame from a generator seeded by SCRAMBLE_SEED (2000 + seed for dascr, 1000 + seed for
rdscr, as in training) and offset by the worker id, so the loader uses the number of workers training used (16;
PREDCACHE_WORKERS overrides). A run whose test accuracy does not reproduce the value stored at training time is reported
as MISMATCH and not cached. Output: cmkd_results/_predcache/<family>_s<seed>.npz (analysis/table3.py --cache reads them).
"""
import torch, numpy as np, json, glob, os, sys
from argparse import Namespace
from torch.utils.data import DataLoader
import torchvision.transforms as transf

FAM = sys.argv[1] if len(sys.argv) > 1 else ''
NEED = {'damarg': ('MAP_TRANSFORM', 'marginal'), 'rdmarg': ('MAP_TRANSFORM', 'marginal'),
        'dabin':  ('MAP_TRANSFORM', 'binary'),   'dascr':  ('SCRAMBLE_DOPPLER', '1'),
        'rdscr':  ('SCRAMBLE_DOPPLER', '1')}
# scramble seed offsets differ by arm: 2000+S for the Doppler-angle arm, 1000+S for run_m5.sh
SCR_OFFSET = {'dascr': 2000, 'rdscr': 1000}
if FAM not in NEED:
    sys.exit(f'usage: predcache_ladder.py [{"|".join(NEED)}]')
k, v = NEED[FAM]
if os.environ.get(k) != v:
    sys.exit(f'FATAL: {FAM} needs {k}={v} in the environment before import; got {os.environ.get(k)!r}')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
os.chdir(REPO)
import train_cmkd as T
from model_cmkd import RadarOnlyStudent
from DataFeed import DataFeed

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = 'dataset/scenario9'
OUT = 'cmkd_results/_predcache'
os.makedirs(OUT, exist_ok=True)
NW = int(os.environ.get('PREDCACHE_WORKERS', '16'))     # match training
DEFAULTS = dict(preserve_angle=False, student_radar_channels=0, second_map='DA',
                radar_channels=2, radar_h=128, radar_w=64, seq_len=8, num_pred=6,
                feature_size=64, gru_hidden=64, num_classes=64, gru_layers_student=1,
                num_heads_mha=8, batch_size=32)

PATTERNS = {'damarg': 'cmkd_results/_gridDAmarg_s*/phase0_*',
            'dabin':  'cmkd_results/_gridDAbin_s*/phase0_*',
            'rdmarg': 'cmkd_results/_gridRDmarg_s*/phase0_*',
            'dascr':  'cmkd_results/_dascr_s*/phase0_*',
            'rdscr':  'cmkd_results/_m5scr_s*/phase0_*'}


def loader(a):
    """Built fresh per run: the scramble arm needs new workers to pick up the
    SCRAMBLE_SEED set for that run, and a cached DataLoader would not."""
    pipe = transf.Compose([transf.ToPILImage(), transf.Resize((224, 224))])
    return DataLoader(
        DataFeed(ROOT, os.path.join(ROOT, 'test_seqs_RA.csv'), seq_len=a.seq_len,
                 transform=pipe, clipped_range=a.radar_h, second_map=a.second_map),
        batch_size=a.batch_size, shuffle=False, num_workers=NW)


def run(d):
    a = Namespace(**{**DEFAULTS, **json.load(open(f'{d}/args.json'))})
    ck = [c for c in ([f'{d}/radar_baseline.pth'] + sorted(glob.glob(f'{d}/student*.pth'))
                      + sorted(glob.glob(f'{d}/*best*.pth'))) if os.path.exists(c)]
    if not ck:
        return None, 'NO CKPT'
    m = RadarOnlyStudent(
        feature_size=a.feature_size, num_classes=a.num_classes, gru_hidden=a.gru_hidden,
        gru_layers=a.gru_layers_student, num_heads_mha=a.num_heads_mha,
        radar_h=a.radar_h, radar_w=a.radar_w,
        radar_channels=(a.student_radar_channels or a.radar_channels),
        preserve_angle=a.preserve_angle).to(DEV)
    m.load_state_dict(torch.load(ck[0], map_location=DEV)); m.eval()

    P, L = [], []
    with torch.no_grad():
        for img, ra, da, beam, label in loader(a):
            img_b, radar_b, tgt, np_ = T.prepare_batch(img, ra, da, beam, label, a, DEV)
            s_ch = a.student_radar_channels or a.radar_channels
            r_in = radar_b if s_ch == radar_b.shape[2] else radar_b[:, :, :s_ch]
            logits, _, _ = m(r_in, np_)
            P.append(logits[:, -(a.num_pred + 1):, :].argmax(-1).cpu())
            L.append(tgt.cpu())
    P = torch.cat(P).numpy(); L = torch.cat(L).numpy()
    stored = np.load(f'{d}/eval_results.npz')['top1']
    got = (P == L).mean(0)
    ok = abs(got.mean() - stored.mean()) < 1e-3 and np.abs(got - stored).max() < 8e-3
    return (P, L, got, stored), ('MATCH' if ok else 'MISMATCH')


print(f'family {FAM}   {k}={v}   workers={NW}')
n_ok = n_bad = 0
for d in sorted(glob.glob(PATTERNS[FAM])):
    seed = json.load(open(f'{d}/args.json'))['seed']
    tag = f'{FAM}_s{seed}'
    if os.path.exists(f'{OUT}/{tag}.npz'):
        print(f'{tag:12s} cached'); n_ok += 1; continue
    if FAM in SCR_OFFSET:
        os.environ['SCRAMBLE_SEED'] = str(SCR_OFFSET[FAM] + seed)
    r, status = run(d)
    if r is None:
        print(f'{tag:12s} {status}  ({d})'); n_bad += 1; continue
    P, L, got, stored = r
    print(f'{tag:12s} {status}  n={len(P)}  ATop-1 {100*got.mean():.2f} '
          f'(stored {100*stored.mean():.2f}, max slot dev {100*np.abs(got-stored).max():.2f} pp)')
    if status == 'MATCH':
        np.savez_compressed(f'{OUT}/{tag}.npz', pred=P, label=L, seed=seed, fam=FAM)
        n_ok += 1
    else:
        n_bad += 1
print(f'LADDER_DONE {FAM}  ok={n_ok} bad={n_bad}')
