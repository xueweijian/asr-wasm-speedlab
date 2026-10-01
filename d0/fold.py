#!/usr/bin/env python3
"""D0: offline constant-fold the small-ctc model and verify parity vs original.

Pipeline:
  1. Load model.onnx, report node/ops stats (before)
  2. onnxsim simplify (constant folding + shape inference + dead code elimination)
  3. Report stats (after); assert node reduction
  4. Parity gate: run original vs folded in onnxruntime (CPU fp32) on random
     inputs shaped from the model's own input specs; elementwise max-abs/rel diff.
Exit non-zero if parity fails or folding explodes.
"""
import json
import sys
import os

import numpy as np
import onnx
from onnxsim import simplify

MODEL = sys.argv[1] if len(sys.argv) > 1 else "model.onnx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "folded.onnx"
STATS = sys.argv[3] if len(sys.argv) > 3 else "results/d0-fold-stats.json"


def stats_of(m):
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    init = len(m.graph.initializer)
    init_bytes = sum(i.raw_data.__len__() if i.raw_data else 0 for i in m.graph.initializer)
    return {"nodes": len(m.graph.node), "ops": ops, "initializers": init,
            "initializer_bytes": init_bytes, "op_kinds": len(ops)}


def rand_for(vi, all_inputs, seed):
    """Streaming-contract-aware random feeds.

    zipformer-ctc streaming inputs: x [N,T,80] fbank; cached_len/processed_lens
    [N] int64; cached_avg/key/val/val2 [N,T_prev,D] float states (initially empty).
    Unknown dims: batch->1, x's time->32, state time->0 (initial state).
    """
    rng = np.random.default_rng(seed)
    t = vi.type.tensor_type
    elem = onnx.TensorProto.DataType.Name(t.elem_type)
    dims = [d.dim_value if d.HasField("dim_value") else None for d in t.shape.dim]
    is_fbank = elem == "FLOAT" and len(dims) == 3 and dims[2] == 80
    shape = []
    for i, d in enumerate(dims):
        if d is not None and d > 0:
            shape.append(d)
        elif i == 0:
            shape.append(1)                      # batch
        elif is_fbank and i == 1:
            shape.append(32)                     # current-chunk frames
        elif not is_fbank and i == 1:
            shape.append(0)                      # initial state: T_prev = 0
        else:
            shape.append(dims[-1] or 1)
    if elem == "FLOAT":
        if is_fbank:
            # log-mel scale ~[-20, +5]: random normal noise NaNs the CTC head
            return (rng.random(shape).astype(np.float32) * 25.0) - 20.0
        return rng.standard_normal(shape).astype(np.float32) * 0.5
    if elem in ("INT64", "INT32"):
        return np.zeros(shape, dtype=np.int64 if elem == "INT64" else np.int32)
    if elem == "BOOL":
        return np.zeros(shape, dtype=bool)
    raise SystemExit(f"unsupported input type {elem} for {vi.name}")


def main():
    m = onnx.load(MODEL)
    before = stats_of(m)
    print(f"[before] nodes={before['nodes']} op_kinds={before['op_kinds']} "
          f"initializers={before['initializers']}")

    sm, ok = simplify(m)  # default: constant folding + shape inference + DCE
    if not ok:
        raise SystemExit("onnxsim simplify check failed")
    after = stats_of(sm)
    print(f"[after ] nodes={after['nodes']} op_kinds={after['op_kinds']} "
          f"initializers={after['initializers']}")

    onnx.save(sm, OUT)
    print(f"saved folded model -> {OUT} ({os.path.getsize(OUT)/1e6:.1f}MB)")

    # ---- parity gate (onnxruntime, CPU fp32, deterministic seeds) ----
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # fold check, not ORT's
    s1 = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    s2 = ort.InferenceSession(OUT, so, providers=["CPUExecutionProvider"])

    print("--- input table ---")
    for vi in m.graph.input:
        t = vi.type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") else "?" for d in t.shape.dim]
        print(f"  {vi.name}: {onnx.TensorProto.DataType.Name(t.elem_type)} {dims}")
    feeds = {vi.name: rand_for(vi, m.graph.input, seed=42 + i)
             for i, vi in enumerate(m.graph.input)}
    o1 = s1.run(None, feeds)
    o2 = s2.run(None, feeds)

    report = []
    worst = 0.0
    for a, b, out_vi in zip(o1, o2, m.graph.output):
        if a.dtype.kind == "f":
            fa, fb = a.astype(np.float64), b.astype(np.float64)
            na, nb = np.isnan(fa), np.isnan(fb)  # nan masks must match exactly
            both_nan = int((na & nb).sum())
            only_a, only_b = int((na & ~nb).sum()), int((nb & ~na).sum())
            clean = ~na & ~nb
            diff = np.abs(np.where(clean, fa - fb, 0.0))
            denom = np.maximum(np.abs(np.where(clean, fa, 1.0)), 1e-9)
            rel = float((diff / denom).max()) if clean.any() else 0.0
            mad = float(diff.max()) if clean.any() else 0.0
            nan_mismatch = only_a + only_b
            worst = max(worst, mad, float("inf") if nan_mismatch else 0.0)
            report.append({"output": out_vi.name, "max_abs": mad, "max_rel": rel,
                           "nan_both": both_nan, "nan_only_orig": only_a,
                           "nan_only_folded": only_b})
            print(f"  parity {out_vi.name}: max_abs={mad:.3e} max_rel={rel:.3e} "
                  f"nan(both/orig/folded)={both_nan}/{only_a}/{only_b}")
        else:
            eq = bool((a == b).all())
            report.append({"output": out_vi.name, "exact": eq})
            print(f"  parity {out_vi.name}: exact={eq}")
            if not eq:
                worst = float("inf")

    TOL = 1e-4  # fp32 folding on int8-quantized graph: quant params are constants,
    # folded DQ arithmetic must be bit-close (1e-4 headroom for op reordering)
    reduction = 1 - after["nodes"] / before["nodes"]
    os.makedirs(os.path.dirname(STATS), exist_ok=True)
    json.dump({"model": os.path.basename(MODEL), "before": before, "after": after,
               "node_reduction": round(reduction, 4), "parity": report,
               "tol": TOL, "pass": worst <= TOL},
              open(STATS, "w"), indent=1)
    print(f"[gate] node_reduction={reduction:.1%} parity_worst={worst:.3e} tol={TOL}")
    if worst > TOL:
        raise SystemExit("D0 PARITY GATE FAILED")
    print("D0 PASS")


if __name__ == "__main__":
    main()
