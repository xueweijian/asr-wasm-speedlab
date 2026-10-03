#!/usr/bin/env python3
"""Tolerance calibration for matmul_int8_dq (#154 family).

Recomputes the DQL -> MMI -> Cast -> Mul chain in float64 from raw tensors,
compares against the ORT reference, and measures the noise introduced by
+-1 integer quantization wobble (what a 1-ulp scale difference between GPU
qstat and ORT would produce). Outputs a statistically grounded tolerance.
"""
import json
import struct
import sys
import numpy as np

BASE = "/var/minis/workspace/asr-wasm-speedlab"
KJ = json.load(open(f"{BASE}/d1/out/kernels.json"))
WMAP = {w["name"]: w for w in KJ["weights"]}
WB = open(f"{BASE}/d1/out/weights.bin", "rb").read()
MAN = json.load(open(f"{BASE}/d1/runtime/ref/chunk0.json"))
RB = open(f"{BASE}/d1/runtime/ref/chunk0.bin", "rb").read()


def from_weights(name):
    m = WMAP[name]
    off, n = m["offset"], m["bytes"]
    dt = m["dtype"]
    if dt == "int8":
        return np.frombuffer(WB, dtype=np.int8, count=n, offset=off)
    if dt == "float32":
        return np.frombuffer(WB, dtype=np.float32, count=n // 4, offset=off)
    raise ValueError(dt)


def from_ref(name):
    m = MAN[name]
    off, n = m["offset"], m["bytes"]
    dt = m["dtype"]
    if dt == "float32":
        return np.frombuffer(RB, dtype=np.float32, count=n // 4, offset=off).astype(np.float64)
    if dt == "int8":
        return np.frombuffer(RB, dtype=np.int8, count=n, offset=off)
    raise ValueError(dt + " " + name)


def dql_per_tensor(x):
    """ORT DynamicQuantizeLinear per-tensor semantics."""
    mn = min(float(x.min()), 0.0)
    mx = max(float(x.max()), 0.0)
    scale = (mx - mn) / 255.0 if (mx - mn) != 0 else 1.0
    zp = int(np.clip(round(-mn / scale), 0, 255))  # uint8 zp
    q = np.clip(np.round(x / scale) + zp, 0, 255).astype(np.int32)
    return q, scale, zp


def wshape_full(wname):
    return tuple(WMAP[wname]["shape"])


def mmi_dq(x, wname, wzpname, out_ref_name, perturb=None):
    """Full chain: DQL(x) -> MMI with int8 w -> f32 -> result.
    perturb: None | 'zp' | 'rand1' — perturbation modes for noise estimation."""
    w = from_weights(wname).astype(np.int32).reshape(wshape_full(wname))
    wscale = float(from_weights(wname.replace("_quantized", "_scale"))[0])
    wzp = int(from_ref(wzpname)[0]) if wzpname in MAN else int(from_weights(wzpname)[0])
    q, scale, zp = dql_per_tensor(x)
    if perturb == "zp":
        zp2 = zp + 1 if zp < 255 else zp - 1
        q = np.clip(np.round(x / scale) + zp2, 0, 255).astype(np.int32)
    elif perturb == "rand1":
        # simulate ~0.1% of quantized ints differing by +-1
        rng = np.random.default_rng(7)
        mask = rng.random(q.shape) < 0.001
        delta = rng.integers(-1, 2, q.shape)
        q = np.clip(q + mask * delta, 0, 255).astype(np.int32)
    acc = (q - zp).astype(np.float64) @ (w - wzp).astype(np.float64)
    return acc * scale * wscale, w.shape


def main():
    x_name = sys.argv[1] if len(sys.argv) > 1 else "/feed_forward2/out_proj/Sub_2_output_0"
    w_name = sys.argv[2] if len(sys.argv) > 2 else "onnx::MatMul_9225_quantized"
    wzp_name = sys.argv[3] if len(sys.argv) > 3 else "onnx::MatMul_9146_zero_point"
    out_name = sys.argv[4] if len(sys.argv) > 4 else "/feed_forward2/out_proj/MatMul_output_0"

    xflat = from_ref(x_name)
    xlast = MAN[x_name]["shape"][-1]
    x = xflat.astype(np.float64).reshape(-1, xlast)
    ref = from_ref(out_name)
    mine, wshape = mmi_dq(x, w_name, wzp_name, out_name)
    K = x.shape[-1]

    ref2 = ref.reshape(mine.shape)
    d = np.abs(mine - ref2)
    print(f"== {out_name}  K={K} x{x.shape} w{wshape}")
    print(f"f64-recompute vs ORT-ref : max={d.max():.3e} p99.9={np.quantile(d, 0.999):.3e} "
          f"mean={d.mean():.3e} nBad(1e-4)={(d > 1e-4).sum()}/{d.size}")
    mm = mine.reshape(ref.shape)

    for mode in ("zp", "rand1"):
        m2, _ = mmi_dq(x, w_name, wzp_name, out_name, perturb=mode)
        d2 = np.abs(m2 - ref2)
        print(f"perturb[{mode}] vs ORT-ref : max={d2.max():.3e} p99.9={np.quantile(d2, 0.999):.3e} "
              f"nBad(1e-4)={(d2 > 1e-4).sum()}/{d2.size}")

    # |ref| stats for relative tolerance design
    a = np.abs(ref2)
    print(f"|ref|: median={np.median(a):.3e} p99={np.quantile(a, 0.99):.3e} max={a.max():.3e}")
    # suggested gate: TOL + REL*|ref| covering 3x observed f64-vs-ORT noise
    base = np.quantile(d, 0.9999)
    print(f"suggested gate: TOL={3 * base:.1e} + REL={(3 * base / max(np.median(a), 1e-6)):.1e}*|ref|")


if __name__ == "__main__":
    main()
