#!/usr/bin/env python3
"""Gamma=1 logit distillation at ten seeds, against no distillation and against the reported Gamma=3 arm
(prespecification/PREREG_KD_T1_20260922.md). Scoring and bootstrap follow exp_power_tail.py: per-window, per-slot loss
from the measured Scenario-9 sweeps over 774 test windows x 7 slots; cluster bootstrap over 28 passes x seeds,
seeds drawn unpaired per arm, 20,000 replicates. Self-check: forward ATop-1 against eval_results.npz.
Sign convention: 'recovery' = L_reference - L_arm in dB (positive = the arm loses less power)."""
import numpy as np, pandas as pd, os, glob, time, json, torch
import torchvision.transforms as transf
from torch.utils.data import DataLoader
from MyFunc import set_seed
from model_cmkd import RadarOnlyStudent
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
POW = np.stack([[np.loadtxt(os.path.join(ROOT, str(seq.iloc[i][c]).lstrip('./'))) for c in SLOT_COLS] for i in range(len(seq))])
OPT = POW.argmax(-1); BEST = np.take_along_axis(POW, OPT[..., None], -1).squeeze(-1)
N = len(seq)
print(f'{N} windows, {len(np.unique(PASS))} passes, power loaded in {time.time()-T0:.0f}s', flush=True)
proc = transf.Compose([transf.ToPILImage(), transf.Resize((224, 224))])
_CACHE = None


def batches():
    global _CACHE
    if _CACHE is None:
        ld = DataLoader(DataFeed(ROOT, f'{ROOT}/test_seqs_RA.csv', seq_len=8, transform=proc, clipped_range=128, second_map='RD'),
                        batch_size=32, shuffle=False, num_workers=8)
        t = time.time(); _CACHE = [b for b in ld]
        print(f'  cached test loader (RD) in {time.time()-t:.0f}s', flush=True)
    return _CACHE


def predictions(ckpt):
    m = RadarOnlyStudent(feature_size=64, num_classes=64, gru_hidden=64, gru_layers=1, num_heads_mha=8,
                         radar_h=128, radar_w=64, radar_channels=2)
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(sd.get('model', sd) if isinstance(sd, dict) else sd, strict=True)
    m.to(DEV).eval()
    out, tgt = [], []
    for img, ra, x2, beam, label in batches():
        B = ra.shape[0]
        r = torch.stack([ra[:, 1:], x2[:, 1:]], 2)
        radar = torch.cat([r, torch.zeros(B, 7, 2, 128, 64)], 1).to(DEV)
        out.append(m(radar, 7)[0][:, -7:].argmax(-1).cpu())
        tgt.append(torch.cat([beam[:, -1:], label[:, :6]], -1))
    return torch.cat(out).numpy(), torch.cat(tgt).numpy()


def stat(v, k):
    if k == 'mean': return float(v.mean())
    if k.startswith('p'): return float(np.percentile(v, int(k[1:])))
    if k.startswith('gt'): return float(100 * (v > float(k[2:])).mean())
    raise KeyError(k)


SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ARMS = {'no KD':   ('_v3rd_s{}',   'phase0_*/radar_baseline.pth'),
        'Gamma=3': ('_v3rdkd_s{}', 'phase3_*/student_cmkd_best.pth'),
        'Gamma=1': ('_kd_T1_s{}',  'phase3_*/student_cmkd_best.pth')}
L, C, A, ST, SD = {}, {}, {}, {}, {}
for name, (pat, leaf) in ARMS.items():
    L[name], C[name], A[name], ST[name], SD[name] = {}, {}, {}, {}, []
    for S in SEEDS:
        ck = sorted(glob.glob(f'cmkd_results/{pat.format(S)}/{leaf}'))
        ev = sorted(glob.glob(f'cmkd_results/{pat.format(S)}/*/eval_results.npz'))
        if not ck or not ev:
            continue
        pred, tgt = predictions(ck[-1])
        assert (tgt == OPT).mean() > 0.999
        cap = np.take_along_axis(POW, pred[..., None], -1).squeeze(-1) / (BEST + 1e-12)
        L[name][S] = (-10 * np.log10(np.clip(cap, 1e-6, None))).astype(np.float32)
        C[name][S] = (pred == tgt).astype(np.float32)
        A[name][S] = 100 * (pred == tgt).mean(); ST[name][S] = 100 * float(np.mean(np.load(ev[-1])['top1'])); SD[name].append(S)
    print(f'{name:8s} seeds {SD[name]}', flush=True)

KEYS = ['mean', 'gt3', 'p95', 'p99']
print(f'\n{"arm":8s} {"n":>2s} {"ATop-1 fwd":>10s} {"stored":>7s} | {"mean dB":>11s} {"P>3dB %":>8s} {"p95 dB":>7s} {"p99 dB":>7s}')
mism = []
for name in ARMS:
    if not SD[name]:
        print(f'{name:8s}  0  (no runs)'); continue
    ms = {k: np.array([stat(L[name][S].ravel(), k) for S in SD[name]]) for k in KEYS}
    af = np.array([A[name][S] for S in SD[name]]); st = np.array([ST[name][S] for S in SD[name]])
    mism += [f'{name} s{S}: {A[name][S]:.2f} vs {ST[name][S]:.2f}' for S in SD[name] if abs(A[name][S] - ST[name][S]) > 0.05]
    print(f'{name:8s} {len(SD[name]):2d} {af.mean():10.2f} {st.mean():7.2f} | {ms["mean"].mean():5.3f}+-{ms["mean"].std(ddof=1) if len(af) > 1 else 0:.3f} '
          f'{ms["gt3"].mean():8.2f} {ms["p95"].mean():7.2f} {ms["p99"].mean():7.2f}', flush=True)
print('forward-vs-stored ATop-1 mismatches > 0.05 pp:', mism if mism else 'none')

rng = np.random.default_rng(42)
passes = np.unique(PASS); widx = [np.where(PASS == p)[0] for p in passes]
NB = int(os.environ.get('NBOOT', '20000'))
RES = {}
for label, ref, arm in (('Gamma=1 vs no KD', 'no KD', 'Gamma=1'), ('Gamma=1 vs Gamma=3', 'Gamma=3', 'Gamma=1'), ('Gamma=3 vs no KD', 'no KD', 'Gamma=3')):
    common = [S for S in SD[arm] if S in SD[ref]]
    if not common:
        print(f'\n{label}: no seeds'); continue
    LA = np.stack([L[ref][S] for S in SD[ref]]); LB = np.stack([L[arm][S] for S in SD[arm]])
    CA = np.stack([C[ref][S] for S in SD[ref]]); CB = np.stack([C[arm][S] for S in SD[arm]])
    # paired point estimates on the common seeds
    pt = {'acc_pp': float(np.mean([A[arm][S] - A[ref][S] for S in common])),
          'recovery_dB': float(np.mean([L[ref][S].mean() - L[arm][S].mean() for S in common])),
          'gt3_pp': float(np.mean([stat(L[arm][S].ravel(), 'gt3') - stat(L[ref][S].ravel(), 'gt3') for S in common])),
          'p95_dB': float(np.mean([stat(L[arm][S].ravel(), 'p95') - stat(L[ref][S].ravel(), 'p95') for S in common]))}
    # accuracy arrays per seed for the bootstrap (correct / not, (N,7))
    t = time.time(); acc = {k: np.empty(NB) for k in pt}
    for r in range(NB):
        idx = np.concatenate([widx[p] for p in rng.integers(0, len(passes), len(passes))])
        sa = rng.integers(0, len(LA), len(LA)); sb = rng.integers(0, len(LB), len(LB))
        va = LA[sa][:, idx].ravel(); vb = LB[sb][:, idx].ravel()
        ca = CA[sa][:, idx].ravel(); cb = CB[sb][:, idx].ravel()
        acc['recovery_dB'][r] = va.mean() - vb.mean()
        acc['gt3_pp'][r] = stat(vb, 'gt3') - stat(va, 'gt3')
        acc['p95_dB'][r] = stat(vb, 'p95') - stat(va, 'p95')
        acc['acc_pp'][r] = 100 * (cb.mean() - ca.mean())
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in acc.items()}
    RES[label] = dict(point=pt, ci=ci, n_ref=len(LA), n_arm=len(LB), common=common)
    print(f'\n{label}  ({len(LA)} vs {len(LB)} seeds; paired on {len(common)}; {time.time()-t:.0f}s)')
    for k, unit in (('acc_pp', 'pp ATop-1 (arm minus ref)'), ('recovery_dB', 'dB recovered (ref loss minus arm loss)'),
                    ('gt3_pp', 'pp of slots above 3 dB (arm minus ref)'), ('p95_dB', 'dB at p95 (arm minus ref)')):
        print(f'   {k:12s} {pt[k]:+8.3f}  [{ci[k][0]:+8.3f}, {ci[k][1]:+8.3f}]  {unit}', flush=True)

n1 = len(SD['Gamma=1'])
if 'Gamma=1 vs no KD' in RES:
    p = RES['Gamma=1 vs no KD']['point']['recovery_dB']; lo, hi = RES['Gamma=1 vs no KD']['ci']['recovery_dB']
    if lo <= 0 <= hi and abs(p) < 0.05: branch = 'branch 1: interval includes 0 and |point| < 0.05 dB -> C3 stands; Sec. V-A wording corrected'
    elif lo > 0: branch = f'branch 2: interval excludes 0 on the positive side -> C3 rewritten (distillation recovers {p:+.3f} dB at Gamma=1)'
    else: branch = 'branch 3: report as unresolved with the interval; Sec. V-A wording corrected'
    print(f'\nPREREG READING ({n1}/10 Gamma=1 seeds{"" if n1 == 10 else " -- PARTIAL, not decision-eligible"}): {branch}')
np.savez_compressed('cmkd_results/_kd_t1_eval.npz', pass_id=PASS, **{f'{n}::s{S}': L[n][S] for n in L for S in L[n]})
json.dump(dict(results=RES, seeds={n: SD[n] for n in ARMS}, atop1_fwd={n: A[n] for n in ARMS}, atop1_stored={n: ST[n] for n in ARMS},
               nboot=NB, device=DEV), open('cmkd_results/_kd_t1_eval_summary.json', 'w'), indent=1, default=float)
print(f'saved cmkd_results/_kd_t1_eval.npz and _kd_t1_eval_summary.json   total {time.time()-T0:.0f}s\nKD_T1_EVAL_DONE')
