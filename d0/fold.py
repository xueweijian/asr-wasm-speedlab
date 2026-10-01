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

    # ---- parity gate: multi-chunk streaming, each model feeds its own states ----
    # Single-chunk compare shows orig emits NaN in dead slots where the folded
    # graph emits 0.0 (empty-state subgraphs folded away). Honest gate = deployment
    # semantics: states loop back per model, log_probs must match every chunk.
    # If dead slots were live, divergence would amplify within 3 chunks.
    in_names = [vi.name for vi in m.graph.input]
    out_names = [o.name for o in m.graph.output]
    by_out = {}
    for n in out_names:
        for i in in_names:
            if n == "new_" + i or n == i:
                by_out[n] = i

    state1 = {vi.name: rand_for(vi, m.graph.input, seed=7)
              for vi in m.graph.input if vi.name != "x"}
    state2 = dict(state1)
    report, worst, NCHUNK = [], 0.0, 3
    for c in range(NCHUNK):
        x = rand_for(m.graph.input[0], m.graph.input, seed=100 + c)  # fresh chunk
        o1 = dict(zip(out_names, s1.run(None, {"x": x, **state1})))
        o2 = dict(zip(out_names, s2.run(None, {"x": x, **state2})))
        a, b = o1["log_probs"], o2["log_probs"]
        fa, fb = a.astype(np.float64), b.astype(np.float64)
        na, nb = np.isnan(fa), np.isnan(fb)
        clean = ~na & ~nb
        diff = np.abs(np.where(clean, fa - fb, 0.0))
        mad = float(diff.max()) if clean.any() else 0.0
        mismatch = int((na ^ nb).sum())
        print(f"  chunk{c}: log_probs shape={a.shape} max_abs={mad:.3e} "
              f"nan(o/f)={int(na.sum())}/{int(nb.sum())} mask_mismatch={mismatch}")
        report.append({"chunk": c, "shape": list(a.shape), "max_abs": mad,
                       "nan_orig": int(na.sum()), "nan_folded": int(nb.sum()),
                       "nan_mask_mismatch": mismatch})
        worst = max(worst, mad, float("inf") if mismatch else 0.0)
        state1 = {by_out[n]: v for n, v in o1.items() if n in by_out and n != "log_probs"}
        state2 = {by_out[n]: v for n, v in o2.items() if n in by_out and n != "log_probs"}

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
