"""Classical radar tracker in road coordinates (Sec. V-C; supplementary S8), Scenario 9, local files only.

Model. The lane is a straight line at perpendicular distance d from the radar; the vehicle's along-road position is
s (s = 0 at closest approach), its along-road speed v. Then
    range            r   = sqrt(s^2 + d^2)
    bearing          phi = atan2(s, d)              (0 at broadside to the road)
    radial velocity  v_r = v * s / r = v * sin(phi)
and the bearing rate is d(sin phi)/dt = v cos^2(phi) / r, which a fixed k * v_r / r model cannot represent.

Measurements per frame (extract_frames.py): range centroid r [m] of the range-angle detection; angle a [bins], the
centroid of the Doppler-summed per-angle moving-energy profile around its peak (the most precise angle reading we
found: 2.2 beam indices of residual against 4.2 for the range-angle detection); Doppler centroid c [bins] from the
Doppler-angle map in a +-2 column band around the detected angle (plain or thresholded at half the band maximum).
Calibration (training passes only; beam labels and pass-median GPS speed): bearing reference from the measured beam,
phi_b = (b - b0) * dphi with the 90-degree, 64-beam codebook (dphi = 90/64 degrees) and b0 the beam at which the
Doppler centroid crosses zero; d from r = d / cos(phi_b); a = alpha + beta * sin(phi_b);
(c - 64) * 0.243 = kappa * v_gps * sin(phi_b); noise levels from robust residuals.
Filter: extended Kalman filter, state [s, v], constant velocity with white-acceleration process noise q, innovation
gating, prior v0 = training mean pass speed with standard deviation sv0.
Position-to-beam mapping: either the codebook geometry b = b0 + atan(s/d)/dphi, or an isotonic regression of the
measured beam at slot t on the filtered s at slot t over the training windows (the site geometry a learned model
absorbs from its training passes).
Variants: full (range, angle, Doppler), no Doppler (range, angle), position held (no extrapolation); each also
anchored at the last measured beam (oracle-fed), b(t+j) = b_meas(t) + B(s(t+j)) - B(s(t)).
Every setting (q, gate, sv0, Doppler estimator, mapping) is chosen per variant on the training windows by ATop-1
over t+1..t+6. Scored over the 774 Scenario 9 test windows with the measured-sweep loss of the paper."""
import csv, os, math, json, itertools, numpy as np

from paths import DATA, CACHE, OUT, result, pow_test
HERE = str(CACHE)      # frames.npz from extract_frames.py
ROOT = str(DATA)
DV, DT, C = 0.243, 1.0 / 9.0, 64
DPHI = math.radians(90.0 / 64)
Z = np.load(os.path.join(HERE, "frames.npz"))
FIDX = {p: i for i, p in enumerate(Z["path"])}
R_, S_, T_, PID, BEAM = Z["r"], Z["strength"], Z["t"], Z["pid"], Z["beam"]
A_ = Z["a_da"]
DOP = {"centroid": Z["c"], "thresholded": Z["c_thr"]}
LAT, LON = Z["lat"], Z["lon"]

_pw = {}
def argmax_pow(rel):
    if rel not in _pw:
        _pw[rel] = int(np.argmax(np.loadtxt(os.path.join(ROOT, rel.lstrip("./")))))
    return _pw[rel]
def load_windows(name):
    rows = list(csv.DictReader(open(os.path.join(ROOT, name))))
    F = np.array([[FIDX[r[f"radar{k}"]] for k in range(1, 9)] for r in rows])
    Y = np.array([[argmax_pow(r["beam8"])] + [argmax_pow(r[f"future_beam{j}"]) for j in range(1, 7)] for r in rows])
    P = np.array([int(r["seq_index"]) for r in rows])
    return F, Y, P
Ftr, Ytr, Ptr = load_windows("train_seqs_RA_tr.csv")
Fte, Yte, Pte = load_windows("test_seqs_RA.csv")
train_passes = set(Ptr.tolist())
trf = np.array([p in train_passes for p in PID])

# ---------------- calibration on training frames ----------------
def hav(la1, lo1, la2, lo2):
    Rr = 6371000.0; p1, p2 = np.radians(la1), np.radians(la2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2) ** 2
    return 2 * Rr * np.arcsin(np.sqrt(a))
spd = np.full(len(PID), np.nan)
same = (PID[1:] == PID[:-1]) & (np.diff(T_) > 0) & (np.diff(T_) < 1.0)
spd[1:][same] = hav(LAT[:-1], LON[:-1], LAT[1:], LON[1:])[same] / np.diff(T_)[same]
pass_speed = {p: np.nanmedian(spd[PID == p]) for p in np.unique(PID)}
VGPS = np.array([pass_speed[p] for p in PID])
mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))

def calibrate(dop):
    Cd = DOP[dop]
    good = trf & (BEAM >= 2) & (BEAM <= 61) & (R_ > 5.0) & (R_ < 12.0) & np.isfinite(Cd)
    mid = good & (BEAM >= 20) & (BEAM <= 46)
    k1, k0 = np.polyfit(BEAM[mid].astype(float), Cd[mid] - 64.0, 1)
    b0 = -k0 / k1
    phib = (BEAM - b0) * DPHI
    d = float(np.sum(R_[good] / np.cos(phib[good])) / np.sum(1 / np.cos(phib[good]) ** 2))
    X = np.vstack([np.ones(good.sum()), np.sin(phib[good])]).T
    alpha, beta = np.linalg.lstsq(X, A_[good], rcond=None)[0]
    vd = (Cd - 64.0) * DV; g2 = good & np.isfinite(VGPS)
    kappa = float(np.sum(vd[g2] * VGPS[g2] * np.sin(phib[g2])) / np.sum((VGPS[g2] * np.sin(phib[g2])) ** 2))
    return dict(dop=dop, b0=float(b0), d=d, alpha=float(alpha), beta=float(beta), kappa=kappa,
                sig_r=float(mad(R_[good] - d / np.cos(phib[good]))), sig_a=float(mad(A_[good] - (alpha + beta * np.sin(phib[good])))),
                sig_d=float(mad(vd[g2] - kappa * VGPS[g2] * np.sin(phib[g2]))),
                v0=float(np.nanmean([pass_speed[p] for p in train_passes])), strength_min=float(np.percentile(S_[trf], 5)),
                n_frames=int(good.sum()))
CAL = {dop: calibrate(dop) for dop in DOP}
for dop, c in CAL.items():
    print("calibration", dop, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}))

# ---------------- EKF ----------------
def run_window(frames, cal, q, gate, sv0, use, dscale=1.0):
    Cd = DOP[cal["dop"]]; D, AL, BE, KA = cal["d"], cal["alpha"], cal["beta"], cal["kappa"]
    use = list(use); Rm = np.diag(np.array([cal["sig_r"], cal["sig_a"], cal["sig_d"] * dscale])[use] ** 2)
    x = None; Pm = None; tprev = None
    for f in frames:
        ok = (S_[f] >= cal["strength_min"]) and (5.0 < R_[f] < 12.0) and np.isfinite(Cd[f]) and np.isfinite(A_[f])
        if x is None:
            if not ok:
                continue
            sp = float(np.clip((A_[f] - AL) / BE, -0.95, 0.95))
            x = np.array([D * math.tan(math.asin(sp)), cal["v0"]]); Pm = np.diag([2.0 ** 2, sv0 ** 2]); tprev = T_[f]
            continue
        dt = T_[f] - tprev; tprev = T_[f]
        Fm = np.array([[1.0, dt], [0.0, 1.0]]); Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        x = Fm @ x; Pm = Fm @ Pm @ Fm.T + Q
        if not ok:
            continue
        s, v = x; r = math.hypot(s, D); qq = D * D / r ** 3
        hx = np.array([r, AL + BE * s / r, KA * v * s / r])
        Hm = np.array([[s / r, 0.0], [BE * qq, 0.0], [KA * v * qq, KA * s / r]])[use]
        z = np.array([R_[f], A_[f], (Cd[f] - 64.0) * DV])
        y = (z - hx)[use]; Sm = Hm @ Pm @ Hm.T + Rm
        if y @ np.linalg.solve(Sm, y) > gate:
            continue
        K = Pm @ Hm.T @ np.linalg.inv(Sm); x = x + K @ y; Pm = (np.eye(2) - K @ Hm) @ Pm
    if x is None:
        return np.nan, np.nan
    return x[0] + x[1] * (T_[frames[-1]] - tprev), x[1]      # estimate at the time of slot t

_cache = {}
def states(split, key):
    k = (split,) + key
    if k not in _cache:
        dop, q, gate, sv0, use, ds = key; F = Ftr if split == "tr" else Fte
        _cache[k] = np.array([run_window(fr, CAL[dop], q, gate, sv0, use, ds) for fr in F])
    return _cache[k]

def pav(x, y):
    o = np.argsort(x); xs, ys = x[o], y[o].astype(float)
    val, wt, xr = [], [], []
    for xi, yi in zip(xs, ys):
        val.append(yi); wt.append(1.0); xr.append(xi)
        while len(val) > 1 and val[-2] > val[-1]:
            w = wt[-2] + wt[-1]; v_ = (val[-2] * wt[-2] + val[-1] * wt[-1]) / w
            val[-2:] = [v_]; wt[-2:] = [w]; xr[-2:] = [xr[-1]]
    xr, val = np.array(xr), np.array(val)
    return lambda s: np.interp(s, xr, val)

def mapping(kind, cal, Str):
    if kind == "geometry":
        return lambda s: cal["b0"] + np.arctan2(s, cal["d"]) / DPHI
    ok = np.isfinite(Str[:, 0])
    return pav(Str[ok, 0], Ytr[ok, 0])

def predict(Sx, Bmap, v0, anchor=None, hold=False):
    s, v = Sx[:, 0].copy(), Sx[:, 1].copy()
    bad = ~np.isfinite(s); s[bad] = 0.0; v[bad] = v0
    j = np.arange(7)[None, :]
    sj = s[:, None] + (0.0 if hold else 1.0) * v[:, None] * j * DT
    Bj = Bmap(sj)
    if anchor is not None:
        Bj = anchor[:, None] + Bj - Bj[:, :1]
    return np.clip(np.rint(Bj), 0, C - 1).astype(int)

acc = lambda P, Y: 100 * (P[:, 1:] == Y[:, 1:]).mean()

POWTE = pow_test()
assert (POWTE.argmax(-1) == Yte).all()
BEST = POWTE.max(-1)
def loss_db(pred):
    return -10 * np.log10(np.clip(np.take_along_axis(POWTE, pred[..., None], -1)[..., 0] / BEST, 1e-6, None))

VARIANTS = {"full": ((0, 1, 2), False), "no_doppler": ((0, 1), False), "held": ((0, 1, 2), True)}
GRID = dict(dop=list(DOP), q=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], gate=[9.0, 16.0, 1e9], sv0=[0.1, 0.25, 0.5, 1.0, 1.5], mapping=["geometry", "isotonic"], dscale=[1.0, 2.0, 4.0, 8.0, 16.0])
res, preds, chosen = {}, {}, {}
for vname, (use, hold) in VARIANTS.items():
    for anch in [False, True]:
        tag = vname + ("_anchored" if anch else "")
        best = None
        for dop, q, gate, sv0, mp, ds in itertools.product(*GRID.values()):
            if vname == "no_doppler" and (dop != "centroid" or ds != 1.0):
                continue                                     # Doppler settings are irrelevant without Doppler
            key = (dop, q, gate, sv0, use, ds)
            Str = states("tr", key); Bm = mapping(mp, CAL[dop], Str)
            a = acc(predict(Str, Bm, CAL[dop]["v0"], anchor=Ytr[:, 0] if anch else None, hold=hold), Ytr)
            if best is None or a > best[0]:
                best = (a, dop, q, gate, sv0, mp, ds)
        a_tr, dop, q, gate, sv0, mp, ds = best
        key = (dop, q, gate, sv0, use, ds)
        Bm = mapping(mp, CAL[dop], states("tr", key)); Ste = states("te", key)
        pr = predict(Ste, Bm, CAL[dop]["v0"], anchor=Yte[:, 0] if anch else None, hold=hold)
        L = loss_db(pr)
        res[tag] = dict(train_atop1_h=a_tr, atop1_h=acc(pr, Yte), atop1_7=float(100 * (pr == Yte).mean()),
                        db_h=float(L[:, 1:].mean()), db_7=float(L.mean()), gt3_h=float(100 * (L[:, 1:] > 3).mean()),
                        v_median=float(np.nanmedian(Ste[:, 1])), valid=int(np.isfinite(Ste[:, 0]).sum()))
        chosen[tag] = dict(dop=dop, q=q, gate=gate, sv0=sv0, mapping=mp, dscale=ds)
        preds[tag] = pr
        print(f"{tag:20s} train {a_tr:5.1f}% | test ATop-1 t+1..t+6 {res[tag]['atop1_h']:5.1f}% (t..t+6 {res[tag]['atop1_7']:5.1f}%) "
              f"loss {res[tag]['db_h']:.2f} dB  >3dB {res[tag]['gt3_h']:4.1f}%  v~{res[tag]['v_median']:.2f}  [{dop}, q={q}, gate={gate:g}, sv0={sv0}, {mp}, dscale={ds}]")
print("test pass-median GPS speed:", round(float(np.median([pass_speed[p] for p in set(Pte.tolist())])), 2), "m/s")

ref = np.load(result("_pred_dump.npz"))
assert (ref["OPT"].astype(int) == Yte).all()
np.savez(os.path.join(OUT, "_trk_road_ekf.npz"), OPT=ref["OPT"], PASS=ref["PASS"],
         **{f"{k}::ekf": v.astype(np.int16) for k, v in preds.items()})
json.dump(dict(calibration=CAL, grid=GRID, chosen=chosen, results=res), open(os.path.join(OUT, "road_ekf_summary.json"), "w"), indent=1)
