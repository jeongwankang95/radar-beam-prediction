"""Doppler-centroid test (Doppler as a bearing proxy) + codebook angular-grid check (Scenario 9, local).

Pre-specified in prespecification/MECHANISM_PRESPEC_2026-09-27.md. Reads only local files:
  dataset/scenario9/scenario9_RA.csv   (paths, seq_index, timestamps, GPS, power files)
  dataset/scenario9/unit1/radar_DA/*.npy (128 x 64 Doppler-angle maps, log2(1+.) compressed)
  dataset/scenario9/unit1/mmWave_data/*.txt (64-beam linear power vectors)
  dataset/scenario9/unit2/GPS_data_calibrated/*.txt (lat, lon)
Outputs a JSON summary and an NPZ with per-frame statistics.
"""
import csv, os, json, sys, math
import numpy as np

from paths import DATA, OUT
ROOT = str(DATA)
os.makedirs(OUT, exist_ok=True)
DV = 0.243  # m/s per Doppler bin (750 MHz, 15 MHz/us, 128 chirps; v_max 56 km/h, [7])
N_ARR = 16  # communication ULA elements
C = 64

def P(rel):
    return os.path.join(ROOT, rel.lstrip("./").lstrip("/"))

rows = list(csv.DictReader(open(os.path.join(ROOT, "scenario9_RA.csv"))))
n = len(rows)
print("samples", n)

pass_id = np.array([int(r["seq_index"]) for r in rows])
def ts(r):
    s = r["time_stamp[UTC]"].strip("[]'\" ")
    h, m, sec, ms = s.split("-")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0
t = np.array([ts(r) for r in rows])

# ---- labels and power profiles
POW = np.zeros((n, C), np.float64)
for i, r in enumerate(rows):
    POW[i] = np.loadtxt(P(r["unit1_pwr_60ghz"]))
b = POW.argmax(1)
csv_b = np.array([int(r["unit1_beam_index"]) for r in rows])
print("label check: argmax == csv-1 on", np.mean(b == csv_b - 1), "of samples; argmax == csv on", np.mean(b == csv_b))

# ---- GPS speed (calibrated)
lat = np.zeros(n); lon = np.zeros(n)
for i, r in enumerate(rows):
    g = np.loadtxt(P(r["unit2_loc_cal"]))
    lat[i], lon[i] = g[0], g[1]
def hav(la1, lo1, la2, lo2):
    R = 6371000.0
    p1, p2 = np.radians(la1), np.radians(la2)
    dphi = p2 - p1; dl = np.radians(lo2 - lo1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
speed = np.full(n, np.nan)
for i in range(1, n):
    if pass_id[i] == pass_id[i - 1] and 0 < (t[i] - t[i - 1]) < 1.0:
        speed[i] = hav(lat[i - 1], lon[i - 1], lat[i], lon[i]) / (t[i] - t[i - 1])

# ---- Doppler-angle maps: per-frame statistics
cen_band = np.zeros(n); std_band = np.zeros(n); cen_glob = np.zeros(n); apk = np.zeros(n, int)
argmax_row = np.zeros(n, int); e_peak = np.zeros(n)
for i, r in enumerate(rows):
    m = np.load(P(r["unit1_radar"]).replace("_RA", "_DA")).astype(np.float64)  # 128 x 64
    w = np.maximum(m - np.median(m, axis=0, keepdims=True), 0.0)
    e = w.sum(0)
    a = int(e.argmax()); apk[i] = a; e_peak[i] = e[a]
    lo, hi = max(0, a - 2), min(C, a + 3)
    wb = w[:, lo:hi]
    d = np.arange(128)[:, None]
    tot = wb.sum()
    if tot > 0:
        c = (wb * d).sum() / tot
        cen_band[i] = c
        std_band[i] = math.sqrt(max(((wb * (d - c) ** 2).sum() / tot), 0.0))
    else:
        cen_band[i] = np.nan; std_band[i] = np.nan
    cen_glob[i] = (w * d).sum() / max(w.sum(), 1e-12)
    argmax_row[i] = np.unravel_index(w.argmax(), w.shape)[0]

# ---- split by pass order (paper: first 91 train, next 17 val, last 28 test)
passes = np.unique(pass_id)
order = passes  # seq_index is chronological
train_p = set(order[:91]); val_p = set(order[91:108]); test_p = set(order[108:])
tr = np.array([p in train_p for p in pass_id]); te = np.array([p in test_p for p in pass_id])
print("passes", len(passes), "train/val/test frames", tr.sum(), (~tr & ~te).sum(), te.sum())

def linfit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return coef
def rms(v): return float(np.sqrt(np.mean(v ** 2)))

ok = np.isfinite(cen_band)
res = {}
# P1: b from Doppler centroid
co = linfit(cen_band[tr & ok], b[tr & ok]); pred = co[0] * cen_band[te & ok] + co[1]
res["P1_centroid_reader"] = dict(slope_beam_per_bin=float(co[0]), intercept=float(co[1]),
    test_rms_resid_beams=rms(pred - b[te & ok]), test_r=float(np.corrcoef(cen_band[te & ok], b[te & ok])[0, 1]),
    train_r=float(np.corrcoef(cen_band[tr & ok], b[tr & ok])[0, 1]))
# angle-peak reader (reproduce Table S11 calibration)
co_a = linfit(apk[tr].astype(float), b[tr]); pred_a = co_a[0] * apk[te] + co_a[1]
res["angle_peak_reader"] = dict(slope=float(co_a[0]), intercept=float(co_a[1]),
    test_rms_resid_beams=rms(pred_a - b[te]), test_r=float(np.corrcoef(apk[te], b[te])[0, 1]),
    train_r=float(np.corrcoef(apk[tr], b[tr])[0, 1]))
# argmax-row reader
co_m = linfit(argmax_row[tr].astype(float), b[tr]); pred_m = co_m[0] * argmax_row[te] + co_m[1]
res["argmax_row_reader"] = dict(slope=float(co_m[0]), test_rms_resid_beams=rms(pred_m - b[te]),
    test_r=float(np.corrcoef(argmax_row[te], b[te])[0, 1]))
# P3: joint reader
X = np.vstack([apk[tr & ok], cen_band[tr & ok], np.ones((tr & ok).sum())]).T
cj = np.linalg.lstsq(X, b[tr & ok], rcond=None)[0]
pj = cj[0] * apk[te & ok] + cj[1] * cen_band[te & ok] + cj[2]
res["P3_joint_reader"] = dict(coef=[float(v) for v in cj], test_rms_resid_beams=rms(pj - b[te & ok]))
# residual as a function of |b-32| (broadside vs edges) for the centroid reader
edges = np.abs(b[te & ok] - 31.5)
for name, mask in [("near_broadside(|b-32|<8)", edges < 8), ("mid(8..20)", (edges >= 8) & (edges < 20)), ("edge(>=20)", edges >= 20)]:
    res["P1_centroid_reader"]["resid_" + name] = rms((pred - b[te & ok])[mask]) if mask.any() else None

# P2: per-pass slope vs speed
per = []
for p in passes:
    m = (pass_id == p) & ok
    if m.sum() < 8: continue
    sl = linfit(b[m].astype(float), cen_band[m])[0]  # bins per beam
    sp = np.nanmedian(speed[pass_id == p])
    r = np.corrcoef(b[m], cen_band[m])[0, 1]
    per.append((int(p), float(sl), float(sp), float(r), int(m.sum()), float(np.nanstd(speed[pass_id == p]) / sp)))
per = np.array(per)
sl, sp = per[:, 1], per[:, 2]
k = (sl * sp).sum() / (sp * sp).sum()  # through-origin fit slope = k * speed
r2 = 1 - ((sl - k * sp) ** 2).sum() / ((sl - sl.mean()) ** 2).sum()
ds = sl * DV / sp  # implied delta(sin theta) per beam
res["P2_speed_scaling"] = dict(n_passes=int(len(per)), same_sign=bool((np.sign(sl) == np.sign(sl[0])).all()),
    k_bins_per_beam_per_mps=float(k), R2_through_origin=float(r2), corr_slope_speed=float(np.corrcoef(sl, sp)[0, 1]),
    median_within_pass_r=float(np.median(per[:, 3])), speed_mps_min_med_max=[float(np.min(sp)), float(np.median(sp)), float(np.max(sp))],
    implied_dsin_per_beam_median=float(np.median(ds)), implied_dsin_iqr=[float(np.percentile(ds, 25)), float(np.percentile(ds, 75))],
    within_pass_speed_cv_median=float(np.median(per[:, 5])))

# ---- codebook grid check (independent of H_proxy): single-ray 16-element ULA fit of measured profiles
def pattern(N, sin_t, sin_c):
    d = np.pi * (sin_t[:, None] - sin_c[None, :]) / 2.0
    num = np.sin(N * d); den = N * np.sin(d)
    g = np.where(np.abs(den) < 1e-9, 1.0, num / np.where(np.abs(den) < 1e-9, 1.0, den))
    return g ** 2
cands = {
    "sin_uniform_pm90 (paper)": 2.0 * np.arange(C) / (C - 1) - 1.0,
    "sin_uniform_pm45": np.sin(np.radians(45)) * (2.0 * np.arange(C) / (C - 1) - 1.0),
    "angle_uniform_pm45": np.sin(np.radians(np.linspace(-45, 45, C))),
    "sin_uniform_pm37(profile-reader calib)": 0.60 * (2.0 * np.arange(C) / (C - 1) - 1.0),
}
Pn = POW / POW.max(1, keepdims=True)
grid = np.linspace(-1, 1, 4001)
fitres = {}
for name, sc in cands.items():
    G = pattern(N_ARR, grid, sc)  # (T, C)
    Gn = G / np.maximum(G.max(1, keepdims=True), 1e-12)
    best = np.full(n, np.inf);
    for s in range(0, n, 512):
        Pb = Pn[s:s + 512]
        num = Pb @ Gn.T                      # (B,T)
        den = (Gn ** 2).sum(1)[None, :]      # (1,T)
        g = num / den
        resid = (Pb ** 2).sum(1)[:, None] - 2 * g * num + g ** 2 * den  # ||p - g m||^2
        best[s:s + 512] = np.minimum(best[s:s + 512], resid.min(1) / (Pb ** 2).sum(1))
    fitres[name] = dict(median_rel_resid=float(np.median(best)), mean_rel_resid=float(np.mean(best)))
# measured -3 dB width in beams (interpolated) around argmax
def width3db(p):
    j = p.argmax(); h = p[j] / 2.0
    l = j
    while l > 0 and p[l - 1] >= h: l -= 1
    r = j
    while r < C - 1 and p[r + 1] >= h: r += 1
    lf = l - (p[l] - h) / (p[l] - p[l - 1]) if l > 0 and p[l] > p[l - 1] else l
    rf = r + (p[r] - h) / (p[r] - p[r + 1]) if r < C - 1 and p[r] > p[r + 1] else r
    return rf - lf
W = np.array([width3db(p) for p in POW])
def model_width(sc):
    # -3 dB width (in beam units) of an N-element ULA main lobe steered at beam 32, sampled on this grid
    G = pattern(N_ARR, np.array([sc[32]]), sc)[0]
    return width3db(G / G.max())
res["grid_check"] = dict(single_ray_fit=fitres, measured_3dB_width_beams_median=float(np.median(W)),
    measured_3dB_width_beams_mean=float(np.mean(W)),
    model_3dB_width_beams={k: float(model_width(v)) for k, v in cands.items()},
    profile_reader_slope_radar_bins_per_beam=float(1.0 / co_a[0]) if co_a[0] != 0 else None,
    implied_dsin_from_profile_reader=float((2.0 / 64) * co_a[0]) )  # radar bin = 2/64 in sin; beams per radar bin = co_a[0]

json.dump(res, open(os.path.join(OUT, "e4_summary.json"), "w"), indent=2)
np.savez(os.path.join(OUT, "e4_frames.npz"), b=b, cen_band=cen_band, std_band=std_band, cen_glob=cen_glob, apk=apk,
         argmax_row=argmax_row, pass_id=pass_id, speed=speed, per_pass=per, width3db=W, e_peak=e_peak)
print(json.dumps(res, indent=1))
