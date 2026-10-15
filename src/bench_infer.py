#!/usr/bin/env python3
"""Single-sample inference latency of the radar-only and vision-only students: the measurement behind the latencies the
paper cites from its companion paper (Sec. V-B), with that paper's inputs.
Arms: VisionOnlyStudent(64,64,64,gru_layers=1,num_heads_mha=8) on motion masks
(input 1x14x1x224x224, num_past=7) and RadarOnlyStudent(...,radar_h=128,radar_w=64,radar_channels=2) on RA+RD maps
(input 1x14x2x128x64, num_past=7). Latency depends on architecture and input size, not on weights, so the weights are
random (seed 0). Timed: forward pass only under inference_mode (no capture, preprocessing or I/O).
Protocol: eval mode; 50 warm-up calls; 300 timed calls, each timed separately with perf_counter; CPU at 1 thread and at
the default thread count; Apple GPU via MPS with torch.mps.synchronize() before and after every call.
Usage, from the repository root: python src/bench_infer.py [tag]   (writes cmkd_results/bench_infer_<date>[_tag].json)"""
import os, sys, json, time, platform, subprocess, datetime
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from model_cmkd import RadarOnlyStudent
from model_vision_only import VisionOnlyStudent

WARM, N = 50, 300
torch.manual_seed(0)
arms = {
    'vision': (VisionOnlyStudent(64, 64, 64, gru_layers=1, num_heads_mha=8), (1, 14, 1, 224, 224)),
    'radar':  (RadarOnlyStudent(64, 64, 64, gru_layers=1, num_heads_mha=8, radar_h=128, radar_w=64, radar_channels=2),
               (1, 14, 2, 128, 64)),
}

def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception: return ''

env = dict(date=datetime.datetime.now().isoformat(timespec='seconds'), torch=torch.__version__,
           platform=platform.platform(), cpu=sh('sysctl -n machdep.cpu.brand_string'),
           perf_cores=sh('sysctl -n hw.perflevel0.physicalcpu'), eff_cores=sh('sysctl -n hw.perflevel1.physicalcpu'),
           mem_bytes=sh('sysctl -n hw.memsize'), default_threads=torch.get_num_threads(),
           mps=torch.backends.mps.is_available(), warmup=WARM, timed_calls=N)

def stats(ts):
    ts = np.array(ts) * 1e3
    return dict(median_ms=float(np.median(ts)), mean_ms=float(ts.mean()), std_ms=float(ts.std()),
                p05_ms=float(np.percentile(ts, 5)), p95_ms=float(np.percentile(ts, 95)))

def time_cpu(model, x, threads):
    torch.set_num_threads(threads)
    with torch.inference_mode():
        for _ in range(WARM): model(x, 7)
        ts = []
        for _ in range(N):
            t0 = time.perf_counter(); model(x, 7); ts.append(time.perf_counter() - t0)
    return stats(ts)

def time_mps(model, x):
    dev = torch.device('mps'); m = model.to(dev); xd = x.to(dev)
    with torch.inference_mode():
        for _ in range(WARM): m(xd, 7)
        torch.mps.synchronize()
        ts = []
        for _ in range(N):
            torch.mps.synchronize(); t0 = time.perf_counter(); m(xd, 7); torch.mps.synchronize()
            ts.append(time.perf_counter() - t0)
    model.to('cpu')
    return stats(ts)

def flops(model, x):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.inference_mode(), FlopCounterMode(display=False) as fc:
            model(x, 7)
        return int(fc.get_total_flops())
    except Exception as e:
        return None

res = dict(env=env, arms={})
default_threads = env['default_threads']
for name, (model, shape) in arms.items():
    model.eval(); x = torch.randn(*shape)
    r = dict(input=list(shape), params=int(sum(p.numel() for p in model.parameters())), flops=flops(model, x))
    r['cpu_1thread'] = time_cpu(model, x, 1)
    r['cpu_default_threads'] = time_cpu(model, x, default_threads)
    torch.set_num_threads(default_threads)
    r['mps'] = time_mps(model, x) if env['mps'] else None
    res['arms'][name] = r

print(json.dumps(env, indent=1))
print('%-7s %10s %9s | %-26s | %-26s | %-26s' % ('arm', 'params', 'GFLOP', 'CPU 1 thread median (p05-p95)', 'CPU %d threads median (p05-p95)' % default_threads, 'MPS median (p05-p95)'))
for name, r in res['arms'].items():
    f = lambda d: '%.2f ms (%.2f-%.2f)' % (d['median_ms'], d['p05_ms'], d['p95_ms']) if d else 'n/a'
    print('%-7s %10d %9s | %-26s | %-26s | %-26s' % (name, r['params'], ('%.3f' % (r['flops'] / 1e9)) if r['flops'] else 'n/a',
                                                      f(r['cpu_1thread']), f(r['cpu_default_threads']), f(r['mps'])))
tag = ('_' + sys.argv[1]) if len(sys.argv) > 1 else ''
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cmkd_results', 'bench_infer_%s%s.json' % (datetime.date.today().strftime('%Y%m%d'), tag))
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, 'w'), indent=1)
print('saved', out)
