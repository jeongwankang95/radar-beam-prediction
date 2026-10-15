"""Generic test-set prediction dump for trained arms (run from the repository root with the arm's
environment variables set, because DataFeed reads MAP_TRANSFORM / LAST_FRAME_ONLY / RA_VARIANT at import).

  python src/w2_pred.py --name twostem --tags "_twostem_rd_s*" --leaf "phase0_*/radar_baseline.pth" \
      --chan both --second_map RD --model student --kwargs '{"two_stem": true}' --out cmkd_results/_w2pred_twostem.npz

Writes '<name>::<run_dir>' int16 arrays of shape (N_test, 7) plus OPT (arg max of the recorded sweep) and PASS.
The forward path is that of src/exp_power_tail.py."""
import argparse, json, glob, os, sys, time
import numpy as np, pandas as pd, torch
import torchvision.transforms as transf
from torch.utils.data import DataLoader
from MyFunc import set_seed
from model_cmkd import RadarOnlyStudent, MultimodalTeacher
from model_vision_only import VisionOnlyStudent
from DataFeed import DataFeed

p = argparse.ArgumentParser()
p.add_argument('--name', required=True); p.add_argument('--tags', required=True, help='glob under cmkd_results/; several globs separated by commas keep their order')
p.add_argument('--leaf', default='phase0_*/radar_baseline.pth')
p.add_argument('--chan', default='both', choices=['first', 'second', 'both', 'three'])
p.add_argument('--second_map', default='RD'); p.add_argument('--extra_map', default='')
p.add_argument('--model', default='student', choices=['student', 'vision', 'teacher'])
p.add_argument('--kwargs', default='{}', help='json overrides for the model constructor')
p.add_argument('--data_root', default='dataset/scenario9'); p.add_argument('--test_csv', default='test_seqs_RA.csv')
p.add_argument('--out', required=True); p.add_argument('--num_workers', type=int, default=8)
a = p.parse_args()
set_seed(42); torch.set_grad_enabled(False)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = a.data_root
seq = pd.read_csv(f'{ROOT}/{a.test_csv}')
PASS = seq['seq_index'].to_numpy()
SLOT_COLS = ['beam8'] + [f'future_beam{j}' for j in range(1, 7)]
def loadpow(path):
    v = np.loadtxt(os.path.join(ROOT, str(path).lstrip('./')))
    return v
POW = np.stack([[loadpow(seq.iloc[i][c]) for c in SLOT_COLS] for i in range(len(seq))])   # (N,7,64), may hold NaN (S31)
OPT = np.where(np.isnan(POW).all(-1), -1, np.nanargmax(np.where(np.isnan(POW), -np.inf, POW), -1))
OK = ~np.isnan(POW).any(-1)
proc = transf.Compose([transf.ToPILImage(), transf.Resize((224, 224))])
ld = DataLoader(DataFeed(ROOT, f'{ROOT}/{a.test_csv}', seq_len=8, transform=proc, clipped_range=128,
                         second_map=a.second_map, extra_map=a.extra_map),
                batch_size=32, shuffle=False, num_workers=a.num_workers)
t0 = time.time(); BATCHES = [b for b in ld]; print(f'loader cached in {time.time()-t0:.0f}s ({len(seq)} windows)', flush=True)

base_S = dict(feature_size=64, num_classes=64, gru_hidden=64, gru_layers=1, num_heads_mha=8, radar_h=128, radar_w=64)
base_V = dict(feature_size=64, num_classes=64, gru_hidden=64, gru_layers=1, num_heads_mha=8)
base_T = dict(feature_size=64, num_classes=64, gru_hidden=64, gru_layers=2, num_heads_ca=4, num_heads_mha=8, radar_h=128, radar_w=64)
kw_over = json.loads(a.kwargs)
if a.model == 'student':
    C = {'first': 1, 'second': 1, 'both': 2, 'three': 3}[a.chan]
    kw = base_S | dict(radar_channels=C) | kw_over; cls = RadarOnlyStudent
elif a.model == 'vision':
    kw = base_V | kw_over; cls = VisionOnlyStudent
else:
    kw = base_T | dict(radar_channels=2) | kw_over; cls = MultimodalTeacher

def predictions(ckpt):
    m = cls(**kw)
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(sd.get('model', sd) if isinstance(sd, dict) else sd, strict=True)
    m.to(DEV).eval()
    out, tgt = [], []
    for img, ra, x2, beam, label in BATCHES:
        B = ra.shape[0]
        if a.chan == 'both':
            r = torch.stack([ra[:, 1:], x2[:, 1:]], 2)
        elif a.chan == 'second':
            r = x2[:, 1:].unsqueeze(2)
        elif a.chan == 'three':
            r = torch.cat([ra[:, 1:].unsqueeze(2), x2[:, 1:]], 2)
        else:
            r = ra[:, 1:].unsqueeze(2)
        Cc = r.shape[2]
        radar = torch.cat([r, torch.zeros(B, 7, Cc, 128, 64)], 1).to(DEV)
        vis = torch.cat([img.unsqueeze(2), torch.zeros(B, 7, 1, 224, 224)], 1).to(DEV)
        if a.model == 'vision':
            lg = m(vis, 7)[0]
        elif a.model == 'teacher':
            lg = m(vis, radar, 7)[0]
        else:
            lg = m(radar, 7)[0]
        out.append(lg[:, -7:].argmax(-1).cpu())
        tgt.append(torch.cat([beam[:, -1:], label[:, :6]], -1))
    return torch.cat(out).numpy(), torch.cat(tgt).numpy()

out = {'OPT': OPT.astype(np.int16), 'PASS': PASS.astype(np.int32), 'OK': OK}
if os.path.exists(a.out):                       # resumable: keep arrays already computed
    old = np.load(a.out); out.update({k: old[k] for k in old.files if '::' in k})
dirs = []
for t in a.tags.split(','):                    # each glob sorted, globs in the given order (seed order matters for the bootstrap)
    dirs += [d for d in sorted(glob.glob(f'cmkd_results/{t.strip()}')) if d not in dirs]
n = 0
for d in dirs:
    tag = os.path.basename(d); key = f'{a.name}::{tag}'
    if key in out: n += 1; continue
    ck = sorted(glob.glob(os.path.join(d, a.leaf)))
    if not ck: print(f'  no checkpoint yet: {tag}', flush=True); continue
    if not glob.glob(os.path.join(d, '*', 'eval_results.npz')) and not glob.glob(os.path.join(d, 'eval_results.npz')):
        print(f'  not finished: {tag}', flush=True); continue
    pred, tgt = predictions(ck[-1])
    agree = (tgt[OK] == OPT[OK]).mean()
    assert agree > 0.99, f'labels != sweep argmax ({agree:.3f}) for {tag}'
    out[key] = pred.astype(np.int16); n += 1
    acc = 100 * (pred[OK] == OPT[OK]).mean()
    print(f'  {tag}: ATop-1 {acc:.2f}%', flush=True)
np.savez(a.out, **out)
print(f'{a.name}: {n} runs saved -> {a.out}')
