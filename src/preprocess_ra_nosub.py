"""Range-angle maps WITHOUT the slow-time mean subtraction (Scenario 9; supplementary S5), from the raw
.mat radar files. Same chain as preprocess_radar.py otherwise (range FFT 256, Doppler FFT 128,
angle FFT 64, fftshift on angle/Doppler, mean of |cube| over Doppler, log2(1+.)), full 256x64,
saved as unit1/radar_RAnosub/<base>_RAnosub.npy (float32). DataFeed clips to 128 range bins."""
import os, sys, numpy as np, csv
from concurrent.futures import ProcessPoolExecutor
from scipy.io import loadmat
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "scenario9")
OUT = os.path.join(R, "unit1", "radar_RAnosub"); os.makedirs(OUT, exist_ok=True)

def cube_nosub(raw):
    r = np.fft.fft(raw, n=256, axis=1)                 # range, NO mean removal
    d = np.fft.fft(r, n=128, axis=2)                   # Doppler
    a = np.fft.fft(d, n=64, axis=0)                    # angle
    return np.fft.fftshift(a, axes=(0, 2))

def one(rel):
    base = os.path.splitext(os.path.basename(rel))[0]
    out = os.path.join(OUT, f"{base}_RAnosub.npy")
    if os.path.exists(out):
        return "ok"
    mat = loadmat(os.path.join(R, rel.lstrip("./")))
    key = [k for k in mat if not k.startswith("__")][0]
    raw = mat[key]
    if raw.ndim == 2:
        raw = raw.reshape(4, 256, 128)
    if raw.shape != (4, 256, 128):
        return f"SHAPE {raw.shape}"
    ra = np.log2(1 + np.mean(np.abs(cube_nosub(raw)), axis=2).T).astype(np.float32)  # (256, 64)
    np.save(out, ra)
    return "ok"

if __name__ == "__main__":
    rows = list(csv.DictReader(open(os.path.join(R, "scenario9.csv"))))
    rels = [r["unit1_radar"] for r in rows]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(one, rels, chunksize=32))
    bad = [x for x in res if x != "ok"]
    print(f"written/present: {len(res) - len(bad)} / {len(res)}; problems: {bad[:5]}")
    # sanity: compare one map with the mean-subtracted version
    b = os.path.splitext(os.path.basename(rels[0]))[0]
    a = np.load(os.path.join(OUT, f"{b}_RAnosub.npy")); s = np.load(os.path.join(R, "unit1", "radar_RA", f"{b}_RA.npy"))
    print("shapes", a.shape, s.shape, "nosub range", float(a.min()), float(a.max()), "sub range", float(s.min()), float(s.max()))
