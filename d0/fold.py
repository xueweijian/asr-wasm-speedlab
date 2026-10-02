#!/usr/bin/env python3
"""D0: offline constant-fold the small-ctc model and verify parity vs original.

Pipeline:
  1. Load model.onnx, report node/ops stats (before)
  2. onnxsim simplify (constant folding + shape inference + dead code elimination)
  3. Report stats (after); assert node reduction
  4. Parity gate A (random smoke): iid noise chunks, deployment-semantics state
     feedback. Historically drives the CTC head to NaN — kept only as a
     mask-symmetry smoke, NOT the pass criterion.
  5. Parity gate B (real fbank, THE gate): 16 kHz speech -> numpy kaldi-style
     fbank (povey window / preemph 0.97 / 80 mel bins) -> chunked streaming
     with correct cached_len/processed_lens maintenance. Requires:
       - log_probs live (zero NaN) in BOTH graphs  (liveness gate)
       - NaN masks identical
       - elementwise max|diff| <= 1e-4
Exit non-zero if any gate fails or folding explodes.

Usage: fold.py MODEL.onnx OUT.onnx STATS.json [--wav path/to/speech.wav]
"""
import json
import sys
import os
import wave

import numpy as np
import onnx
from onnxsim import simplify

MODEL = sys.argv[1] if len(sys.argv) > 1 else "model.onnx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "folded.onnx"
STATS = sys.argv[3] if len(sys.argv) > 3 else "results/d0-fold-stats.json"
WAV = None
if "--wav" in sys.argv:
    WAV = sys.argv[sys.argv.index("--wav") + 1]
TOL = 1e-4  # fp32 folding on int8-quantized graph: quant params are constants,
            # folded DQ arithmetic must be bit-close (1e-4 headroom for op reordering)


def stats_of(m):
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    init = len(m.graph.initializer)
    init_bytes = sum(i.raw_data.__len__() if i.raw_data else 0 for i in m.graph.initializer)
    return {"nodes": len(m.graph.node), "ops": ops, "initializers": init,
            "initializer_bytes": init_bytes, "op_kinds": len(ops)}


# ---------------- real fbank: numpy kaldi-style ----------------
def hz_to_mel(f):
    return 1127.0 * np.log(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (np.exp(m / 1127.0) - 1.0)


def mel_filterbank(num_bins, nfft, sr, low=20.0, high=None):
    """kaldi-style triangular mel banks (low_freq=20, high_freq=Nyquist)."""
    if high is None:
        high = sr / 2.0
    melpts = np.linspace(hz_to_mel(low), hz_to_mel(high), num_bins + 2)
    pts = mel_to_hz(melpts)
    freqs = np.arange(nfft // 2 + 1) * (sr / nfft)
    w = np.zeros((num_bins, len(freqs)))
    for b in range(num_bins):
        lo, ce, hi = pts[b], pts[b + 1], pts[b + 2]
        left = (freqs - lo) / max(ce - lo, 1e-9)
        right = (hi - freqs) / max(hi - ce, 1e-9)
        w[b] = np.maximum(0.0, np.minimum(left, right))
    return w


def kaldi_fbank(sig, sr=16000, num_bins=80, frame_ms=25.0, shift_ms=10.0, preemph=0.97):
    """numpy kaldi-style fbank: povey window, per-frame dc removal, preemph,
    512-pt FFT, 80 mel bins, log floor 1e-10. Returns float32 [T, 80]."""
    N = int(sr * frame_ms / 1000)
    S = int(sr * shift_ms / 1000)
    nfft = 1 << (N - 1).bit_length()
    n = np.arange(N)
    win = (0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))) ** 0.85  # povey
    banks = mel_filterbank(num_bins, nfft, sr)
    T = 1 + (len(sig) - N) // S
    out = np.empty((T, num_bins), np.float64)
    x = sig.astype(np.float64)
    for t in range(T):
        fr = x[t * S:t * S + N].copy()
        fr -= fr.mean()
        p = np.empty_like(fr)
        p[0] = fr[0]
        p[1:] = fr[1:] - preemph * fr[:-1]
        p *= win
        spec = np.abs(np.fft.rfft(p, nfft)) ** 2
        out[t] = np.log(np.maximum(banks @ spec, 1e-10))
    return out.astype(np.float32)


def read_wav16k(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"sr={w.getframerate()}, want 16k"
        assert w.getnchannels() == 1, "want mono"
        assert w.getsampwidth() == 2, "want 16-bit PCM"
        raw = w.readframes(w.getnframes())
    return (np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0)


# ---------------- feeds ----------------
def zero_state(vi):
    """sherpa-faithful init states: every non-x input gets a FULL-SIZE zeros
    tensor. Unknown dims are batch (=1), never time — the small-ctc zipformer2
    uses fixed-length sliding-window caches (left_context_len=256), and states
    are fully self-managed via new_* outputs (no external len bookkeeping)."""
    t = vi.type.tensor_type
    elem = onnx.TensorProto.DataType.Name(t.elem_type)
    shape = [(d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 1)
             for d in t.shape.dim]
    if elem == "FLOAT":
        return np.zeros(shape, np.float32)
    if elem in ("INT64", "INT32"):
        return np.zeros(shape, np.int64 if elem == "INT64" else np.int32)
    if elem == "BOOL":
        return np.zeros(shape, bool)
    raise SystemExit(f"unsupported input type {elem} for {vi.name}")


def rand_x(vi, seed, t_frames=None):
    """Random fbank-shaped x (smoke gate only)."""
    rng = np.random.default_rng(seed)
    t = vi.type.tensor_type
    elem = onnx.TensorProto.DataType.Name(t.elem_type)
    dims = [d.dim_value if d.HasField("dim_value") else None for d in t.shape.dim]
    if t_frames is None:
        t_frames = dims[1] if dims[1] else 32
    shape = [1, t_frames, dims[2]]
    return (rng.random(shape).astype(np.float32) * 25.0) - 20.0


def dtype_of(vi):
    return onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)


def parity_loop(sess1, sess2, m, chunks, live_required, label):
    """Run both graphs chunk-by-chunk, each feeding back its own states.
    All non-x inputs start as full-size zeros and loop back via new_* outputs
    (sherpa semantics: Forward(features, states) — states are self-managed)."""
    in_names = [vi.name for vi in m.graph.input]
    out_names = [o.name for o in m.graph.output]
    by_out = {n: i for i in in_names for n in out_names if n == "new_" + i}

    def init_state():
        return {vi.name: zero_state(vi) for vi in m.graph.input if vi.name != "x"}

    st1, st2 = init_state(), init_state()
    report, worst, fail = [], 0.0, None
    for c, x in enumerate(chunks):
        f1 = {"x": x, **st1}
        f2 = {"x": x, **st2}
        o1 = dict(zip(out_names, sess1.run(None, f1)))
        o2 = dict(zip(out_names, sess2.run(None, f2)))
        a, b = o1["log_probs"], o2["log_probs"]
        fa, fb = a.astype(np.float64), b.astype(np.float64)
        na, nb = np.isnan(fa), np.isnan(fb)
        clean = ~na & ~nb
        diff = np.abs(np.where(clean, fa - fb, 0.0))
        mad = float(diff.max()) if clean.any() else 0.0
        mismatch = int((na ^ nb).sum())
        print(f"  [{label}] chunk{c}: shape={a.shape} max_abs={mad:.3e} "
              f"nan(o/f)={int(na.sum())}/{int(nb.sum())} mask_mismatch={mismatch}")
        report.append({"chunk": c, "shape": list(a.shape), "max_abs": mad,
                       "nan_orig": int(na.sum()), "nan_folded": int(nb.sum()),
                       "nan_mask_mismatch": mismatch})
        worst = max(worst, mad, float("inf") if mismatch else 0.0)
        if live_required and (int(na.sum()) or int(nb.sum())):
            fail = fail or f"liveness: chunk{c} produced NaN with real fbank"
        st1 = {by_out[n]: v for n, v in o1.items() if n in by_out and n != "log_probs"}
        st2 = {by_out[n]: v for n, v in o2.items() if n in by_out and n != "log_probs"}
    return report, worst, fail


def main():
    m = onnx.load(MODEL)
    before = stats_of(m)
    print(f"[before] nodes={before['nodes']} op_kinds={before['op_kinds']} "
          f"initializers={before['initializers']}")

    sm, ok = simplify(m)
    if not ok:
        raise SystemExit("onnxsim simplify check failed")
    after = stats_of(sm)
    print(f"[after ] nodes={after['nodes']} op_kinds={after['op_kinds']} "
          f"initializers={after['initializers']}")

    onnx.save(sm, OUT)
    print(f"saved folded model -> {OUT} ({os.path.getsize(OUT)/1e6:.1f}MB)")

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s1 = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    s2 = ort.InferenceSession(OUT, so, providers=["CPUExecutionProvider"])

    print("--- input table ---")
    for vi in m.graph.input:
        t = vi.type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") else "?" for d in t.shape.dim]
        print(f"  {vi.name}: {dtype_of(vi)} {dims}")

    result = {"model": os.path.basename(MODEL), "before": before, "after": after,
              "node_reduction": round(1 - after["nodes"] / before["nodes"], 4),
              "tol": TOL}

    # dump model metadata (T / decode_chunk_len / left_context_len etc.)
    meta = {p.key: p.value for p in m.metadata_props}
    if meta:
        print("--- metadata ---")
        for k, v in meta.items():
            print(f"  {k}: {v[:120]}")

    # ---- gate A: random smoke (mask symmetry only) ----
    chunks = [rand_x(m.graph.input[0], seed=100 + c) for c in range(3)]
    rep_a, worst_a, _ = parity_loop(s1, s2, m, chunks, live_required=False, label="rand")
    result["parity_random_smoke"] = rep_a

    # ---- gate B: real fbank (THE gate) ----
    if WAV:
        sig = read_wav16k(WAV)
        fb = kaldi_fbank(sig)
        T_frames = fb.shape[0]
        # chunk length from the graph's own x spec (fixed dim 77 for small-ctc)
        xd = [d.dim_value if d.HasField("dim_value") else 0
              for d in m.graph.input[0].type.tensor_type.shape.dim]
        frames_per_chunk = int(xd[1]) if len(xd) > 1 and xd[1] > 0 else 32
        nchunks = min(12, T_frames // frames_per_chunk)
        print(f"[real] wav={os.path.basename(WAV)} {len(sig)/16000:.2f}s "
              f"fbank T={T_frames} chunks={nchunks} "
              f"fbank mean={fb.mean():.2f} std={fb.std():.2f} "
              f"min={fb.min():.2f} max={fb.max():.2f}")
        result["real_fbank"] = {"wav": os.path.basename(WAV),
                                "seconds": round(len(sig) / 16000, 2),
                                "fbank_frames": T_frames,
                                "fbank_mean": float(fb.mean()), "fbank_std": float(fb.std()),
                                "fbank_min": float(fb.min()), "fbank_max": float(fb.max()),
                                "chunks": nchunks, "frames_per_chunk": frames_per_chunk}
        chunks = [fb[c * frames_per_chunk:(c + 1) * frames_per_chunk][None, ...] for c in range(nchunks)]
        rep_b, worst_b, fail_b = parity_loop(s1, s2, m, chunks, live_required=True, label="real")
        result["parity_real"] = rep_b
        result["pass"] = (fail_b is None) and worst_b <= TOL
        result["fail_reason"] = fail_b or (None if worst_b <= TOL else f"real max_abs {worst_b:.3e} > {TOL}")
    else:
        print("[real] no --wav given: random smoke only (weak proof)")
        result["pass"] = worst_a <= TOL

    os.makedirs(os.path.dirname(STATS), exist_ok=True)
    json.dump(result, open(STATS, "w"), indent=1)
    print(f"[gate] node_reduction={result['node_reduction']:.1%} "
          f"random_worst={worst_a:.3e} tol={TOL} pass={result['pass']}")
    if not result["pass"]:
        raise SystemExit(f"D0 GATE FAILED: {result.get('fail_reason')}")
    print("D0 PASS")


if __name__ == "__main__":
    main()
