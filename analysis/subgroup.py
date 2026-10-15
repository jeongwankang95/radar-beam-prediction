"""Is the Doppler gain a property of this site's geometry, or does it hold across the
conditions the site contains?

Supplementary S12. Whether the effect is uniform across the variation this site contains:
direction of travel, vehicle speed, how fast the beam moves, and position in the horizon.

Every interval is the paper's own cluster bootstrap, resampling the passes of the
subgroup with replacement alongside the seeds. Subgroups hold roughly a third to a half
of twenty-eight passes each, so the intervals are wide by construction; that width is
part of the answer.

Inputs: the range-angle-only and RA+RD predictions in results/_predcache.npz (families ra and rd, ten seeds each), and
the Scenario 9 CSVs, power vectors and calibrated GPS under dataset/scenario9 (CMKD_DATA overrides).
"""
import os
import numpy as np, pandas as pd
from paths import DATA, result, pow_test

R, B = str(DATA), 20000
FRAME_S = 0.111
t = pd.read_csv(f'{R}/test_seqs_RA.csv')
passes_id = t['seq_index'].values
uniq = list(pd.unique(passes_id))
PW = pow_test(); BEST = PW.max(-1); OK = BEST > 0
Z = np.load(result("_predcache.npz"))

def arms(fam):
    out = []
    for k in sorted((k for k in Z.files if k.startswith(fam + "::")), key=lambda k: k.split("::")[1] + ".npz"):
        P = Z[k]                                  # seeds in the order the original cache files were globbed
        cap = np.take_along_axis(PW, P[:, :, None], axis=2)[:, :, 0]
        db = -10*np.log10(np.clip(np.where(OK, cap/np.where(OK, BEST, 1), 1.0), 1e-6, None))
        out.append(dict(correct=(P == Z["LABEL"]).astype(float), db=db))
    return out
RD, RA = arms('rd'), arms('ra')
LB = Z["LABEL"]

# ---- per-pass covariates -----------------------------------------------------------
base = pd.read_csv(f'{R}/scenario9.csv')
E = 6371000.0
def latlon(f):
    p = os.path.join(R, str(f).lstrip('./'))
    if not os.path.exists(p): return None
    return [float(x) for x in open(p).read().split()[:2]]
speed_by_pass, dir_by_pass, disp_by_pass = {}, {}, {}
for p, g in base.groupby('seq_index'):
    g = g.sort_values('index')
    bi = [int(np.argmax(np.loadtxt(os.path.join(R, str(x).lstrip('./')))))
          for x in g['unit1_pwr_60ghz']]
    dir_by_pass[p] = np.sign(bi[-1] - bi[0])
    disp_by_pass[p] = np.mean(np.abs(np.diff(bi))) if len(bi) > 1 else 0.0
    ll = [latlon(f) for f in g['unit2_loc_cal']]
    v = []
    for i in range(1, len(ll)):
        a, b = ll[i-1], ll[i]
        if a is None or b is None: continue
        dx = E*np.cos(np.radians(a[0]))*np.radians(b[1]-a[1]); dy = E*np.radians(b[0]-a[0])
        v.append(np.hypot(dx, dy)/FRAME_S)
    speed_by_pass[p] = np.mean(v) if v else np.nan

rng = np.random.default_rng(0)
def effect(mask_pass, slots=slice(None), label=''):
    """RA+RD minus RA-only on the passes in mask_pass, bootstrap over those passes."""
    sel = [p for p in uniq if mask_pass(p)]
    if len(sel) < 3:
        print(f'  {label:34s} n={len(sel):2d}  (too few passes to resample)')
        return None
    idx = {p: np.where(passes_id == p)[0] for p in sel}
    rows = np.concatenate([idx[p] for p in sel])
    def stat(r, seeds_rd, seeds_ra, key, scale):
        a = np.mean([RD[j][key][r][:, slots].mean() for j in seeds_rd])
        b = np.mean([RA[j][key][r][:, slots].mean() for j in seeds_ra])
        return scale*(a - b)
    out = {}
    for key, scale, unit in (('correct', 100.0, 'pp'), ('db', -1.0, 'dB')):
        pt = stat(rows, range(len(RD)), range(len(RA)), key, scale)
        v = np.empty(B)
        for i in range(B):
            dr = np.concatenate([idx[sel[j]] for j in rng.integers(0, len(sel), len(sel))])
            v[i] = stat(dr, rng.integers(0, len(RD), len(RD)),
                        rng.integers(0, len(RA), len(RA)), key, scale)
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[unit] = (pt, lo, hi)
    print(f'  {label:34s} n={len(sel):2d}  '
          f'{out["pp"][0]:+5.2f} pp [{out["pp"][1]:+5.2f}, {out["pp"][2]:+5.2f}]   '
          f'{out["dB"][0]:+.3f} dB [{out["dB"][1]:+.3f}, {out["dB"][2]:+.3f}]')
    return out

net = {p: None for p in uniq}
for p, g in base.groupby('seq_index'):
    if p not in net: continue
    g = g.sort_values('index')
    bi = [int(np.argmax(np.loadtxt(os.path.join(R, str(x).lstrip('./')))))
          for x in g['unit1_pwr_60ghz']]
    net[p] = bi[-1] - bi[0]
nv = np.array([v for v in net.values() if v is not None])
print(f'  {len(uniq)} test passes; net beam displacement per pass: '
      f'min {nv.min():+d}, median {int(np.median(nv)):+d}, max {nv.max():+d}')
print(f'  increasing on {(nv>0).sum()}, decreasing on {(nv<0).sum()}, flat on {(nv==0).sum()}')
allnet=[]
for p, g in base.groupby('seq_index'):
    g=g.sort_values('index')
    bi=[int(np.argmax(np.loadtxt(os.path.join(R, str(x).lstrip('./'))))) for x in g['unit1_pwr_60ghz']]
    allnet.append(bi[-1]-bi[0])
allnet=np.array(allnet)
print(f'  over all {len(allnet)} passes in the scenario: increasing {(allnet>0).sum()}, '
      f'decreasing {(allnet<0).sum()}, flat {(allnet==0).sum()}')
print('  Doppler effect = RA+RD minus RA-only\n')
effect(lambda p: True, label='all passes')

print('\n  by direction of travel (net beam-index displacement)')
for d, nm in ((1, 'increasing beam index'), (-1, 'decreasing beam index')):
    effect(lambda p, d=d: dir_by_pass.get(p, 0) == d, label=nm)

sp = np.array([speed_by_pass.get(p, np.nan) for p in uniq]); q = np.nanpercentile(sp, [50])
print(f'\n  by mean vehicle speed (median {q[0]:.2f} m/s)')
effect(lambda p: speed_by_pass.get(p, np.nan) < q[0], label=f'slower than {q[0]:.2f} m/s')
effect(lambda p: speed_by_pass.get(p, np.nan) >= q[0], label=f'faster than {q[0]:.2f} m/s')

dp = np.array([disp_by_pass.get(p, 0) for p in uniq]); m = np.median(dp)
print(f'\n  by beam speed (median {m:.2f} bins per frame)')
effect(lambda p: disp_by_pass.get(p, 0) < m, label=f'beam slower than {m:.2f} bins')
effect(lambda p: disp_by_pass.get(p, 0) >= m, label=f'beam faster than {m:.2f} bins')

print('\n  by position in the horizon')
for j in range(7):
    effect(lambda p: True, slots=slice(j, j+1), label=f'slot t+{j}' if j else 'slot t')
print('SUBGROUP_DONE')
