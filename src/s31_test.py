#!/usr/bin/env python3
"""Scenario 31, the eleven test passes, ten seeds per arm: the pre-registered replication of C2.

Follows s31_val.py (the go/no-go script) exactly in how a loss is scored, but on the test split,
with the ten-seed arms, and with the paper's cluster bootstrap (11 passes x seeds, seeds drawn
unpaired). Primary metric per the pre-registration: paired dL(seed) = L_RA - L_RA+RD in dB from
Scenario 31's own sweeps over all seven slots; secondary: the horizon proper (t+1..t+6) and ATop-1.
Self-check: forward ATop-1 against the eval_results.npz written at training time.
"""
import numpy as np, pandas as pd, os, sys, glob, time, json, torch
import torchvision.transforms as transf
from torch.utils.data import DataLoader
from MyFunc import set_seed
from model_cmkd import RadarOnlyStudent
from DataFeed import DataFeed

ROOT = 'dataset/scenario31'
CSV = f'{ROOT}/test_seqs_RA.csv'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEV.type == 'cpu':
    torch.set_num_threads(max(8, (os.cpu_count() or 16) // 2))
set_seed(42); torch.set_grad_enabled(False)
print('device', DEV, flush=True)
T0 = time.time()
proc = transf.Compose([transf.ToPILImage(), transf.Resize((224, 224))])
te = pd.read_csv(CSV)
PASS = te['seq_index'].to_numpy()
COLS = ['beam8'] + [f'future_beam{i}' for i in range(1, 7)]
PW = np.stack([[np.nan_to_num(np.loadtxt(os.path.join(ROOT, str(r).lstrip('./'))), nan=0.0) for r in te[c]]
               for c in COLS], axis=1)                       # (N, 7, 64)
BEST = PW.max(-1); OK = BEST > 0
LBL = PW.argmax(-1)
N = len(te)
print(f'test windows {N}, passes {len(np.unique(PASS))}, scored slots {int(OK.sum())} of {OK.size}, loaded in {time.time()-T0:.0f}s', flush=True)

_CACHE = None


def batches():
    global _CACHE
    if _CACHE is None:
        ld = DataLoader(DataFeed(ROOT, CSV, seq_len=8, transform=proc, clipped_range=128, second_map='RD'),
                        batch_size=64, shuffle=False, num_workers=8)
        t = time.time(); _CACHE = [b for b in ld]
        print(f'  cached test loader in {time.time()-t:.0f}s', flush=True)
    return _CACHE


def predict(ck, ch):
    m = RadarOnlyStudent(64, 64, 64, gru_layers=1, num_heads_mha=8, radar_h=128, radar_w=64, radar_channels=ch)
    sd = torch.load(ck, map_location='cpu')
    m.load_state_dict(sd.get('model', sd) if isinstance(sd, dict) else sd, strict=True)
    m.eval().to(DEV)
    out = []
    for img, ra, x2, beam, label in batches():
        B = ra.shape[0]
        if ch == 2:
            r = torch.cat([torch.stack([ra[:, 1:], x2[:, 1:]], 2), torch.zeros(B, 7, 2, 128, 64)], 1).to(DEV)
        else:
            r = torch.cat([ra[:, 1:].unsqueeze(2), torch.zeros(B, 7, 1, 128, 64)], 1).to(DEV)
        out.append(m(r, 7)[0][:, -7:, :].argmax(-1).cpu())
    return torch.cat(out).numpy()


def loss_arr(P):
    cap = np.take_along_axis(PW, P[:, :, None], axis=2)[:, :, 0]
    r = np.where(OK, cap / np.where(OK, BEST, 1), 1.0)
    return (-10 * np.log10(np.clip(r, 1e-6, None))).astype(np.float32)   # (N, 7); unscored slots -> 0, as in s31_val.py


def stored_top1(tag):
    f = sorted(glob.glob(f'cmkd_results/{tag}/phase0_*/eval_results.npz'))
    return 100 * float(np.mean(np.load(f[-1])['top1'])) if f else np.nan


SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
PREFIX = sys.argv[1] if len(sys.argv) > 1 else '_s31'   # '_s31fix' for the arms retrained on corrected labels
ARMS = {'RA': (PREFIX + 'ra_s{}', 1), 'RD': (PREFIX + 'rd_s{}', 2)}
L = {k: [] for k in ARMS}      # loss (N, 7) per seed
C = {k: [] for k in ARMS}      # correct-and-scored (N, 7) per seed, float32
AU = {k: [] for k in ARMS}     # unmasked ATop-1 (all slots), for the self-check
ST = {k: [] for k in ARMS}     # stored eval_results ATop-1
used = []
print(f"\n{'seed':>4s} | {'L7 RA':>7s} {'L7 RD':>7s} {'dL7':>8s} | {'L6 RA':>7s} {'L6 RD':>7s} {'dL6':>8s} | {'A7 RA':>6s} {'A7 RD':>6s} | {'stored RA / RD':>15s}")
for S in SEEDS:
    P = {}
    for k, (pat, ch) in ARMS.items():
        ck = sorted(glob.glob(f'cmkd_results/{pat.format(S)}/phase0_*/radar_baseline.pth'))
        if ck:
            P[k] = predict(ck[0], ch)
    if len(P) < 2:
        print(f'  seed {S}: checkpoint missing, skipped'); continue
    used.append(S)
    for k in ARMS:
        L[k].append(loss_arr(P[k])); C[k].append(((P[k] == LBL) & OK).astype(np.float32))
        AU[k].append(100 * (P[k] == LBL).mean()); ST[k].append(stored_top1(ARMS[k][0].format(S)))
    la, lb = L['RA'][-1], L['RD'][-1]
    print(f"{S:4d} | {la.mean():7.4f} {lb.mean():7.4f} {la.mean()-lb.mean():+8.4f} | "
          f"{la[:, 1:].mean():7.4f} {lb[:, 1:].mean():7.4f} {la[:, 1:].mean()-lb[:, 1:].mean():+8.4f} | "
          f"{AU['RA'][-1]:6.2f} {AU['RD'][-1]:6.2f} | {ST['RA'][-1]:6.2f} / {ST['RD'][-1]:6.2f}", flush=True)
LA, LB = np.stack(L['RA']), np.stack(L['RD'])            # (n, N, 7)
CA, CB = np.stack(C['RA']), np.stack(C['RD'])
n = len(used)
mm = [f'{k} s{S}: {a:.2f} vs {s:.2f}' for k in ARMS for S, a, s in zip(used, AU[k], ST[k]) if np.isfinite(s) and abs(a - s) > 0.05]
print(f'\n{n} seeds per arm, forward done in {time.time()-T0:.0f}s.  forward-vs-stored ATop-1 mismatches > 0.05 pp: {mm if mm else "none"}')

# persistence reference (oracle-fed: holds the measured beam at t), read over t+1..t+6
Pp = np.repeat(LBL[:, :1], 7, axis=1); Lp = loss_arr(Pp)
print(f'persistence: L6 {Lp[:, 1:].mean():.4f} dB, ATop-1(t+1..t+6) {100*((Pp == LBL) & OK)[:, 1:].sum()/OK[:, 1:].sum():.2f}%')

nOK7, nOK6 = OK.sum(), OK[:, 1:].sum()
A7a, A7b = 100 * CA.sum((1, 2)) / nOK7, 100 * CB.sum((1, 2)) / nOK7
A6a, A6b = 100 * CA[:, :, 1:].sum((1, 2)) / nOK6, 100 * CB[:, :, 1:].sum((1, 2)) / nOK6
dL7 = LA.mean((1, 2)) - LB.mean((1, 2)); dL6 = LA[:, :, 1:].mean((1, 2)) - LB[:, :, 1:].mean((1, 2))
print(f'\nPAIRED PER SEED (Doppler helps when positive): dL7 = L_RA - L_RD, dA = A_RD - A_RA')
print(f'  dL7 {dL7.mean():+.4f} dB, positive on {int((dL7 > 0).sum())}/{n}   per seed: ' + ', '.join(f'{x:+.4f}' for x in dL7))
print(f'  dL6 {dL6.mean():+.4f} dB, positive on {int((dL6 > 0).sum())}/{n}   per seed: ' + ', '.join(f'{x:+.4f}' for x in dL6))
print(f'  dA7 {(A7b - A7a).mean():+.2f} pp, positive on {int(((A7b - A7a) > 0).sum())}/{n};  dA6 {(A6b - A6a).mean():+.2f} pp')
print(f'  arm means: L7 RA {LA.mean():.4f} RD {LB.mean():.4f} | L6 RA {LA[:, :, 1:].mean():.4f} RD {LB[:, :, 1:].mean():.4f} | '
      f'A7 RA {A7a.mean():.2f}+-{A7a.std(ddof=1):.2f} RD {A7b.mean():.2f}+-{A7b.std(ddof=1):.2f} | A6 RA {A6a.mean():.2f} RD {A6b.mean():.2f}')

# cluster bootstrap: 11 passes with replacement (whole passes), seeds drawn unpaired per arm
rng = np.random.default_rng(42)
passes = np.unique(PASS); widx = [np.where(PASS == p)[0] for p in passes]
NB = int(os.environ.get('NBOOT', '20000'))
res = {k: np.empty(NB) for k in ('dL7', 'dL6', 'dA7', 'dA6')}
t = time.time()
for r in range(NB):
    idx = np.concatenate([widx[p] for p in rng.integers(0, len(passes), len(passes))])
    sa, sb = rng.integers(0, n, n), rng.integers(0, n, n)
    la, lb, ca, cb = LA[sa][:, idx], LB[sb][:, idx], CA[sa][:, idx], CB[sb][:, idx]
    ok7, ok6 = n * OK[idx].sum(), n * OK[idx][:, 1:].sum()
    res['dL7'][r] = la.mean() - lb.mean()
    res['dL6'][r] = la[:, :, 1:].mean() - lb[:, :, 1:].mean()
    res['dA7'][r] = 100 * (cb.sum() - ca.sum()) / ok7
    res['dA6'][r] = 100 * (cb[:, :, 1:].sum() - ca[:, :, 1:].sum()) / ok6
CI = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in res.items()}
print(f'\nCLUSTER BOOTSTRAP, {NB} replicates, {len(passes)} passes x {n} seeds (unpaired), {time.time()-t:.0f}s')
for k, pt in (('dL7', dL7.mean()), ('dL6', dL6.mean()), ('dA7', (A7b - A7a).mean()), ('dA6', (A6b - A6a).mean())):
    print(f'  {k}: {pt:+.4f}  [{CI[k][0]:+.4f}, {CI[k][1]:+.4f}]  {"dB" if k[1] == "L" else "pp"}')
np.savez_compressed(f'cmkd_results/{PREFIX}_test.npz', LA=LA, LB=LB, CA=CA, CB=CB, pass_id=PASS, ok=OK, seeds=np.array(used))
json.dump(dict(seeds=used, dL7=dL7.tolist(), dL6=dL6.tolist(), A7_RA=A7a.tolist(), A7_RD=A7b.tolist(),
               A6_RA=A6a.tolist(), A6_RD=A6b.tolist(), L7_RA=LA.mean((1, 2)).tolist(), L7_RD=LB.mean((1, 2)).tolist(),
               ci=CI, nboot=NB, stored_top1={k: ST[k] for k in ARMS}, unmasked_top1={k: AU[k] for k in ARMS},
               persistence=dict(L6=float(Lp[:, 1:].mean())), device=str(DEV)),
          open(f'cmkd_results/{PREFIX}_test_summary.json', 'w'), indent=1)
print(f'saved cmkd_results/{PREFIX}_test.npz and {PREFIX}_test_summary.json   total {time.time()-T0:.0f}s\nS31TEST_DONE')
