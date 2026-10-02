#!/usr/bin/env python3
"""D1 ref: ground-truth generator for the WGSL runtime parity gate.

Runs the STATIC-FOLDED model (the exact graph kgen compiles) under ORT with
all optimizations disabled and dumps:

  ref/feats.json    fbank chunks fed to the model  [n,1,77,80]
  ref/chunk0.bin    raw bytes of EVERY tensor after chunk 0 (states=0)
  ref/chunk0.json   manifest: name -> {dtype, shape, offset, bytes}
  ref/e2e.json      per-chunk log_probs + final states (the end-to-end gate)

The runtime (d1/runtime) must reproduce chunk0.bin tensor-for-tensor
(kernel-level anchor) and e2e.json within tolerance.

Usage: ref.py FOLDED.onnx WAV OUT_REF_DIR
"""
import json
import os
import sys

import numpy as np
import onnx
from onnx import shape_inference

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# fold.py / kgen.py may live in /root/job (Modal) or d0/ d1/ (CI); accept either
for _p in (os.path.dirname(os.path.abspath(__file__)),
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "d0"),
           "/root/job"):
    if os.path.exists(os.path.join(_p, "fold.py")):
        sys.path.insert(0, os.path.abspath(_p)); break
import fold  # noqa: E402  (kaldi_fbank, read_wav16k, zero_state)
from kgen import static_shape_pass  # noqa: E402

MODEL, WAV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NP_DT = {"FLOAT": np.float32, "UINT8": np.uint8, "INT8": np.int8,
         "INT32": np.int32, "INT64": np.int64, "BOOL": bool}


def expose_intermediates(m):
    """Append every value_info tensor to graph.output so ORT yields all."""
    have = {o.name for o in m.graph.output}
    mi = shape_inference.infer_shapes(m, strict_mode=False)
    for vi in mi.graph.value_info:
        if vi.name not in have:
            m.graph.output.append(vi)
            have.add(vi.name)
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    m = onnx.load(MODEL)
    m = static_shape_pass(m)          # identical transformation to kgen's
    m_all = expose_intermediates(onnx.load_from_string(m.SerializeToString()))
    # keep the plain model too for the streaming loop (graph outputs only)
    m_e2e = m

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s_all = ort.InferenceSession(m_all.SerializeToString(), so,
                                 providers=["CPUExecutionProvider"])
    s_e2e = ort.InferenceSession(m_e2e.SerializeToString(), so,
                                 providers=["CPUExecutionProvider"])

    in_names = [vi.name for vi in m.graph.input]
    out_names = [o.name for o in m.graph.output]
    by_out = {n: i for i in in_names for n in out_names if n == "new_" + i}

    sig = fold.read_wav16k(WAV)
    fb = fold.kaldi_fbank(sig)
    xd = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    fpc = int(xd[1])
    n = min(12, fb.shape[0] // fpc)
    chunks = [fb[c * fpc:(c + 1) * fpc][None].astype(np.float32) for c in range(n)]
    json.dump([c.tolist() for c in chunks], open(os.path.join(OUT, "feats.json"), "w"))
    print(f"[ref] {len(sig)/16000:.2f}s -> {n} chunks x {fpc} frames")

    def zero_states():
        return {vi.name: fold.zero_state(vi) for vi in m.graph.input if vi.name != "x"}

    # ---- chunk 0: all intermediates (kernel-level anchor) ----
    feeds = {"x": chunks[0], **zero_states()}
    outs = s_all.run(None, feeds)
    manifest, blob, off = {}, [], 0
    for name, arr in zip([o.name for o in m_all.graph.output], outs):
        a = np.ascontiguousarray(arr)
        b = a.tobytes()
        manifest[name] = {"dtype": str(a.dtype), "shape": list(a.shape),
                          "offset": off, "bytes": len(b)}
        blob.append(b); off += len(b)
    with open(os.path.join(OUT, "chunk0.bin"), "wb") as f:
        f.write(b"".join(blob))
    json.dump(manifest, open(os.path.join(OUT, "chunk0.json"), "w"))
    print(f"[ref] chunk0: {len(manifest)} tensors {off/1e6:.1f}MB")

    # ---- e2e streaming loop (the gate) ----
    st = zero_states()
    e2e = {"log_probs": [], "states_final": {}, "shapes": {}}
    worst = 0.0
    for c, x in enumerate(chunks):
        o = dict(zip(out_names, s_e2e.run(None, {"x": x, **st})))
        lp = o["log_probs"]
        e2e["log_probs"].append(lp.astype(float).tolist())
        if c == 0:
            e2e["shapes"]["log_probs"] = list(lp.shape)
        st = {by_out[k]: v for k, v in o.items() if k in by_out}
        print(f"[ref] chunk{c}: log_probs{list(lp.shape)} "
              f"range[{lp.min():.2f},{lp.max():.2f}]")
    for k, v in st.items():
        e2e["states_final"][k] = v.astype(float).tolist()
        e2e["shapes"][k] = list(v.shape)
    json.dump(e2e, open(os.path.join(OUT, "e2e.json"), "w"))
    print("[ref] e2e gate data written")
    print("REF PASS")


if __name__ == "__main__":
    main()
