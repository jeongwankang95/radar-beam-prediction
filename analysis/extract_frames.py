"""Per-frame radar measurements for the classical tracker (Scenario 9, all 5,964 frames, local files only).
For every frame: target range r [m] and angle bin a from the range-angle map (argmax after floor removal, then a
local energy-weighted centroid), signed Doppler centroid c [bins] from the Doppler-angle map in a +-2 column band
around the detected angle, detection strength, plus pass id, timestamp, measured beam (sweep arg max), GPS.
Writes frames.npz."""
import csv, os, math, numpy as np
from paths import DATA, CACHE
ROOT = str(DATA)
OUT = str(CACHE)
DR = 45.0 / 256          # m per range bin (256 complex samples, 45 m maximum range, [7])
P = lambda rel: os.path.join(ROOT, rel.lstrip("./"))
rows = list(csv.DictReader(open(os.path.join(ROOT, "scenario9_RA.csv"))))
n = len(rows)
def ts(r):
    h, m, s, ms = r["time_stamp[UTC]"].strip("[]'\" ").split("-"); return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000
out = {k: np.full(n, np.nan) for k in ["r", "a", "c", "strength", "c_strength", "t", "lat", "lon", "a_da", "c_thr", "c_pk"]}
pid = np.zeros(n, int); beam = np.zeros(n, int); path = []
for i, rw in enumerate(rows):
    pid[i] = int(rw["seq_index"]); out["t"][i] = ts(rw); path.append(rw["unit1_radar"])
    beam[i] = int(np.argmax(np.loadtxt(P(rw["unit1_pwr_60ghz"]))))
    g = np.loadtxt(P(rw["unit2_loc_cal"])); out["lat"][i], out["lon"][i] = g[0], g[1]
    RA = np.load(P(rw["unit1_radar"])).astype(np.float64)[:128]          # near 128 range bins (22.5 m)
    W = np.maximum(RA - np.median(RA), 0.0)
    ri, ai = np.unravel_index(W.argmax(), W.shape)
    r0, r1, a0, a1 = max(0, ri-2), min(128, ri+3), max(0, ai-2), min(64, ai+3)
    w = W[r0:r1, a0:a1]; tot = w.sum()
    out["r"][i] = DR * ((w.sum(1) * np.arange(r0, r1)).sum() / tot)
    out["a"][i] = (w.sum(0) * np.arange(a0, a1)).sum() / tot
    out["strength"][i] = W[ri, ai]
    DA = np.load(P(rw["unit1_radar"]).replace("_RA", "_DA")).astype(np.float64)   # 128 Doppler x 64 angle
    Wd = np.maximum(DA - np.median(DA, axis=0, keepdims=True), 0.0)
    ac = int(round(out["a"][i])); b0, b1 = max(0, ac-2), min(64, ac+3)
    wb = Wd[:, b0:b1].sum(1); tb = wb.sum()
    if tb > 0:
        out["c"][i] = (wb * np.arange(128)).sum() / tb; out["c_strength"][i] = tb
        m_ = wb >= 0.5 * wb.max(); out["c_thr"][i] = (wb[m_] * np.arange(128)[m_]).sum() / wb[m_].sum()
        out["c_pk"][i] = float(np.argmax(wb))
    e = Wd.sum(0); ap = int(e.argmax()); q0, q1 = max(0, ap-2), min(64, ap+3)
    out["a_da"][i] = (e[q0:q1] * np.arange(q0, q1)).sum() / max(e[q0:q1].sum(), 1e-12)
np.savez(os.path.join(OUT, "frames.npz"), pid=pid, beam=beam, path=np.array(path), **out)
print("frames", n, "passes", len(np.unique(pid)))
for k in ["r", "a", "c", "strength"]:
    v = out[k]; print(f"{k:9s} p5 {np.nanpercentile(v,5):7.2f}  median {np.nanmedian(v):7.2f}  p95 {np.nanpercentile(v,95):7.2f}")
print("corr(a, beam) %.3f   corr(c, beam) %.3f" % (np.corrcoef(out["a"], beam)[0,1], np.corrcoef(np.nan_to_num(out["c"]), beam)[0,1]))
