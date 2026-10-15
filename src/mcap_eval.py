#!/usr/bin/env python3
"""All-zero second map (matched capacity), ten seeds per arm (seeds trained on two machines).
Arms: published range-angle student (RA10), retrained one-channel RA (_mcap_ra_s*), two-channel
student with the second channel held at zero (_mcap_ra0_s*), and the reference [RA,RD] student
(_v3rd_s*). Scoring and bootstrap as in exp_power_tail.py (774 windows x 7 slots, 28 passes x
seeds, seeds unpaired, 20,000 replicates). Self-check: forward ATop-1 vs eval_results.npz."""
import numpy as np, pandas as pd, os, glob, time, json, torch
import torchvision.transforms as transf
from torch.utils.data import DataLoader
from MyFunc import set_seed
from model_cmkd import RadarOnlyStudent
from DataFeed import DataFeed

ROOT = 'dataset/scenario9'
set_seed(42); torch.set_grad_enabled(False)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEV == 'cpu': torch.set_num_threads(max(8, (os.cpu_count() or 16) // 2))
print('device', DEV, flush=True)
T0 = time.time()
seq = pd.read_csv(f'{ROOT}/test_seqs_RA.csv'); PASS = seq['seq_index'].to_numpy()
SLOT_COLS = ['beam8'] + [f'future_beam{j}' for j in range(1, 7)]
POW = np.stack([[np.loadtxt(os.path.join(ROOT, str(seq.iloc[i][c]).lstrip('./'))) for c in SLOT_COLS] for i in range(len(seq))])
OPT = POW.argmax(-1); BEST = np.take_along_axis(POW, OPT[..., None], -1).squeeze(-1)
proc = transf.Compose([transf.ToPILImage(), transf.Resize((224, 224))])
_CACHE = None
def batches():
    global _CACHE
    if _CACHE is None:
        ld = DataLoader(DataFeed(ROOT, f'{ROOT}/test_seqs_RA.csv', seq_len=8, transform=proc, clipped_range=128, second_map='RD'),
                        batch_size=32, shuffle=False, num_workers=8)
        t = time.time(); _CACHE = [b for b in ld]; print(f'  cached loader in {time.time()-t:.0f}s', flush=True)
    return _CACHE
def predictions(ckpt, chan):
    m = RadarOnlyStudent(feature_size=64, num_classes=64, gru_hidden=64, gru_layers=1, num_heads_mha=8, radar_h=128, radar_w=64,
                         radar_channels=1 if chan == 'first' else 2)
    sd = torch.load(ckpt, map_location='cpu'); m.load_state_dict(sd.get('model', sd) if isinstance(sd, dict) else sd, strict=True); m.to(DEV).eval()
    out, tgt = [], []
    for img, ra, x2, beam, label in batches():
        B = ra.shape[0]
        if chan == 'first': r = ra[:, 1:].unsqueeze(2)
        elif chan == 'zero': r = torch.stack([ra[:, 1:], torch.zeros_like(ra[:, 1:])], 2)
        else: r = torch.stack([ra[:, 1:], x2[:, 1:]], 2)
        radar = torch.cat([r, torch.zeros(B, 7, r.shape[2], 128, 64)], 1).to(DEV)
        out.append(m(radar, 7)[0][:, -7:].argmax(-1).cpu()); tgt.append(torch.cat([beam[:, -1:], label[:, :6]], -1))
    return torch.cat(out).numpy(), torch.cat(tgt).numpy()
def stat(v, k):
    if k == 'mean': return float(v.mean())
    if k.startswith('p'): return float(np.percentile(v, int(k[1:])))
    if k.startswith('gt'): return float(100 * (v > float(k[2:])).mean())
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
RA10 = {42: '_v3_noKD_RA', 1: '_v3s1_noKD_RA', 2: '_v3s2_noKD_RA'} | {s: f'_c1ra_s{s}' for s in range(3, 10)}
ARMS = {'RA published': (lambda s: RA10[s], 'first'), 'RA retrained': (lambda s: f'_mcap_ra_s{s}', 'first'),
        'RA zero-padded 2ch': (lambda s: f'_mcap_ra0_s{s}', 'zero'), 'RA+RD reference': (lambda s: f'_v3rd_s{s}', 'both')}
L, C, A, ST, SD = {}, {}, {}, {}, {}
for name, (fn, chan) in ARMS.items():
    L[name], C[name], A[name], ST[name], SD[name] = {}, {}, {}, {}, []
    for S in SEEDS:
        d = fn(S); ck = sorted(glob.glob(f'cmkd_results/{d}/phase0_*/radar_baseline.pth')); ev = sorted(glob.glob(f'cmkd_results/{d}/*/eval_results.npz'))
        if not ck or not ev: continue
        pred, tgt = predictions(ck[-1], chan); assert (tgt == OPT).mean() > 0.999
        cap = np.take_along_axis(POW, pred[..., None], -1).squeeze(-1) / (BEST + 1e-12)
        L[name][S] = (-10 * np.log10(np.clip(cap, 1e-6, None))).astype(np.float32); C[name][S] = (pred == tgt).astype(np.float32)
        A[name][S] = 100 * (pred == tgt).mean(); ST[name][S] = 100 * float(np.mean(np.load(ev[-1])['top1'])); SD[name].append(S)
    print(f'{name:20s} seeds {SD[name]}', flush=True)
print(f'\n{"arm":20s} {"n":>2s} {"ATop-1 fwd":>10s} {"stored":>7s} | {"mean dB":>12s} {"P>3dB %":>8s} {"p95 dB":>7s}')
mism = []
for name in ARMS:
    if not SD[name]: continue
    ms = {k: np.array([stat(L[name][S].ravel(), k) for S in SD[name]]) for k in ('mean', 'gt3', 'p95')}
    af = np.array([A[name][S] for S in SD[name]]); st = np.array([ST[name][S] for S in SD[name]])
    mism += [f'{name} s{S}: {A[name][S]:.2f} vs {ST[name][S]:.2f}' for S in SD[name] if abs(A[name][S] - ST[name][S]) > 0.05]
    print(f'{name:20s} {len(af):2d} {af.mean():10.2f} {st.mean():7.2f} | {ms["mean"].mean():5.3f}+-{ms["mean"].std(ddof=1) if len(af)>1 else 0:.3f} {ms["gt3"].mean():8.2f} {ms["p95"].mean():7.2f}', flush=True)
    print('   ATop-1 per seed:', {S: round(A[name][S], 2) for S in SD[name]})
print('forward-vs-stored mismatches > 0.05 pp:', mism if mism else 'none')
rng = np.random.default_rng(42); passes = np.unique(PASS); widx = [np.where(PASS == p)[0] for p in passes]
NB = int(os.environ.get('NBOOT', '20000')); RES = {}
for label, ref, arm in (('zero-padded minus published RA', 'RA published', 'RA zero-padded 2ch'),
                        ('RA+RD minus zero-padded', 'RA zero-padded 2ch', 'RA+RD reference'),
                        ('retrained RA minus published RA', 'RA published', 'RA retrained'),
                        ('RA+RD minus published RA', 'RA published', 'RA+RD reference')):
    common = [S for S in SD[arm] if S in SD[ref]]
    if not common: print(f'\n{label}: no seeds'); continue
    LA = np.stack([L[ref][S] for S in SD[ref]]); LB = np.stack([L[arm][S] for S in SD[arm]]); CA = np.stack([C[ref][S] for S in SD[ref]]); CB = np.stack([C[arm][S] for S in SD[arm]])
    pt = {'acc_pp': float(np.mean([A[arm][S] - A[ref][S] for S in common])), 'recovery_dB': float(np.mean([L[ref][S].mean() - L[arm][S].mean() for S in common]))}
    acc = {k: np.empty(NB) for k in pt}; t = time.time()
    for r in range(NB):
        idx = np.concatenate([widx[p] for p in rng.integers(0, len(passes), len(passes))])
        sa = rng.integers(0, len(LA), len(LA)); sb = rng.integers(0, len(LB), len(LB))
        acc['recovery_dB'][r] = LA[sa][:, idx].mean() - LB[sb][:, idx].mean()
        acc['acc_pp'][r] = 100 * (CB[sb][:, idx].mean() - CA[sa][:, idx].mean())
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in acc.items()}
    RES[label] = dict(point=pt, ci=ci, n_ref=len(LA), n_arm=len(LB), common=common)
    print(f'\n{label}  ({len(LA)} vs {len(LB)} seeds; paired on {len(common)}; {time.time()-t:.0f}s)')
    for k, unit in (('acc_pp', 'pp ATop-1 (arm minus ref)'), ('recovery_dB', 'dB recovered (ref loss minus arm loss)')):
        print(f'   {k:12s} {pt[k]:+8.3f}  [{ci[k][0]:+8.3f}, {ci[k][1]:+8.3f}]  {unit}', flush=True)
json.dump(dict(results=RES, seeds={n: SD[n] for n in ARMS}, atop1_fwd={n: A[n] for n in ARMS}, atop1_stored={n: ST[n] for n in ARMS},
               mean_dB={n: {S: float(L[n][S].mean()) for S in SD[n]} for n in ARMS}, nboot=NB), open('cmkd_results/_mcap_eval_summary.json', 'w'), indent=1, default=float)
print(f'saved cmkd_results/_mcap_eval_summary.json   total {time.time()-T0:.0f}s\nMCAP_EVAL_DONE')
