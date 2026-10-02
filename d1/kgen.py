#!/usr/bin/env python3
"""D1 kgen: folded.onnx -> kernels.json + weights.bin + kgen-stats.json.

Compiler front-end for the WGSL runtime:
  * fuses DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul*2 chains into
    single `matmul_int8_dq` kernels (the dot4I8Packed target)
  * classifies every node into kernel kinds the runtime pool must cover
  * separates compile-time layout ops (Reshape/Transpose/Slice/...) from
    real compute, so the dispatch budget is honest
  * packs initializers into weights.bin with a manifest (offset/length)

Usage: kgen.py FOLDED.onnx OUT_DIR [--max-kernels-json N]
"""
import json
import sys
import os

import numpy as np
import onnx
from onnx import numpy_helper

MODEL = sys.argv[1]
OUT_DIR = sys.argv[2]
MAX_JSON = int(sys.argv[sys.argv.index("--max-kernels-json") + 1]) if "--max-kernels-json" in sys.argv else 4000

ELEMENTWISE = {"Add", "Sub", "Mul", "Div", "Max", "Min", "Exp", "Log", "Tanh",
               "Sigmoid", "Pow", "Abs", "Neg", "Sqrt", "Relu", "Gelu", "Erf"}
REDUCE = {"ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd"}
SOFTMAX = {"Softmax", "LogSoftmax"}
LAYOUT = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Slice",
          "Concat", "Split", "Pad", "Gather", "GatherElements", "GatherND",
          "Scatter", "ScatterElements", "Tile", "Expand", "BroadcastGradientArgs",
          "Shape", "Size", "ConstantOfShape", "Range", "Identity"}
CONTROL = {"Cast", "Equal", "NotEqual", "Less", "LessOrEqual", "Greater",
           "GreaterOrEqual", "Where", "Clip", "LeakyRelu", "HardSigmoid"}


def attrs_of(n):
    a = {}
    for at in n.attribute:
        if at.type == onnx.AttributeProto.INT:
            a[at.name] = at.i
        elif at.type == onnx.AttributeProto.INTS:
            a[at.name] = list(at.ints)
        elif at.type == onnx.AttributeProto.FLOAT:
            a[at.name] = at.f
        elif at.type == onnx.AttributeProto.FLOATS:
            a[at.name] = list(at.floats)
        elif at.type == onnx.AttributeProto.TENSOR:
            a[at.name] = "<tensor>"
    return a


def main():
    m = onnx.load(MODEL)
    g = m.graph

    init = {i.name: i for i in g.initializer}
    producer = {}          # tensor name -> producing node index
    for idx, n in enumerate(g.node):
        for o in n.output:
            producer[o] = idx
    consumers = {}         # tensor name -> [node indices]
    for idx, n in enumerate(g.node):
        for i in n.input:
            consumers.setdefault(i, []).append(idx)

    kernels = []           # logical kernels
    consumed = set()       # node indices folded into a fused kernel
    weight_refs = set()    # initializer names referenced by compute kernels

    def in0_is_dql(n):
        """MatMulInteger(x, W): is x produced by DynamicQuantizeLinear?"""
        if not n.input or n.input[0] not in producer:
            return None
        p = g.node[producer[n.input[0]]]
        return p if p.op_type == "DynamicQuantizeLinear" else None

    def dq_chain(n):
        """Find Cast->fp32 then Mul(x_scale)[-> Mul(w_scale) | Mul(w_scale) then Mul(x_scale)] consumers."""
        out = n.output[0]
        chain = []
        cur = out
        for _ in range(3):
            cs = [c for c in consumers.get(cur, []) if c not in consumed]
            if len(cs) != 1:
                break
            node = g.node[cs[0]]
            if node.op_type == "Cast" and cur == node.input[0]:
                chain.append(("cast", cs[0])); cur = node.output[0]; continue
            if node.op_type == "Mul" and len(node.input) == 2:
                other = node.input[1] if node.input[0] == cur else node.input[0]
                chain.append(("mul", cs[0], other)); cur = node.output[0]; continue
            break
        return chain, cur

    for idx, n in enumerate(g.node):
        if idx in consumed:
            continue
        op = n.op_type
        if op == "MatMulInteger":
            dql = in0_is_dql(n)
            if dql is not None and dql.output[0] == n.input[0]:
                chain, last = dq_chain(n)
                if chain and any(k == "mul" for k, *_ in chain):
                    consumed.update({producer[n.input[0]], idx,
                                     *[c for _, c, *_ in chain]})
                    scale_init = [r[0] for k, c, *r in chain if k == "mul" and r and r[0] in init]
                    for i in n.input[1:] + tuple(scale_init) + tuple(dql.input[1:]):
                        if i in init:
                            weight_refs.add(i)
                    kernels.append({
                        "kind": "matmul_int8_dq", "op": op,
                        "inputs": [dql.input[0], n.input[1]],
                        "outputs": [last],
                        "attrs": attrs_of(n),
                        "fused": ["DynamicQuantizeLinear", "MatMulInteger"] +
                                 [g.node[c].op_type for _, c, *_ in chain],
                    })
                    continue
            # plain int8 matmul (x already quantized)
            for i in n.input:
                if i in init:
                    weight_refs.add(i)
            kernels.append({"kind": "matmul_int8", "op": op,
                            "inputs": list(n.input), "outputs": list(n.output),
                            "attrs": attrs_of(n)})
            continue
        if op in ("MatMul", "Gemm"):
            for i in n.input:
                if i in init:
                    weight_refs.add(i)
            kernels.append({"kind": "matmul_fp" if op == "MatMul" else "gemm_fp",
                            "op": op, "inputs": list(n.input),
                            "outputs": list(n.output), "attrs": attrs_of(n)})
            continue
        if op == "Conv":
            for i in n.input:
                if i in init:
                    weight_refs.add(i)
            kernels.append({"kind": "conv", "op": op, "inputs": list(n.input),
                            "outputs": list(n.output), "attrs": attrs_of(n)})
            continue
        if op in ELEMENTWISE:
            kernels.append({"kind": "elementwise", "op": op,
                            "inputs": list(n.input), "outputs": list(n.output),
                            "attrs": attrs_of(n)})
            continue
        if op in REDUCE:
            kernels.append({"kind": "reduce", "op": op,
                            "inputs": list(n.input), "outputs": list(n.output),
                            "attrs": attrs_of(n)})
            continue
        if op in SOFTMAX:
            kernels.append({"kind": "softmax", "op": op,
                            "inputs": list(n.input), "outputs": list(n.output),
                            "attrs": attrs_of(n)})
            continue
        if op in CONTROL:
            kernels.append({"kind": "control", "op": op,
                            "inputs": list(n.input), "outputs": list(n.output),
                            "attrs": attrs_of(n)})
            continue
        if op in LAYOUT:
            kernels.append({"kind": "layout", "op": op,
                            "inputs": list(n.input), "outputs": list(n.output),
                            "attrs": attrs_of(n)})
            continue
        kernels.append({"kind": "UNSUPPORTED", "op": op,
                        "inputs": list(n.input), "outputs": list(n.output),
                        "attrs": attrs_of(n)})

    # ---- weights packing ----
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest, blob, off = [], [], 0
    for name in weight_refs:
        t = init[name]
        arr = numpy_helper.to_array(t)
        b = arr.tobytes()
        manifest.append({"name": name, "dtype": str(arr.dtype),
                         "shape": list(arr.shape), "offset": off, "bytes": len(b)})
        blob.append(b)
        off += len(b)
    with open(os.path.join(OUT_DIR, "weights.bin"), "wb") as f:
        f.write(b"".join(blob))

    # ---- stats ----
    by_kind = {}
    for k in kernels:
        key = k["kind"] + (f"[{k['op']}]" if k["kind"] in ("elementwise", "control", "reduce", "layout") else "")
        by_kind[key] = by_kind.get(key, 0) + 1
    kind_disp = {}
    for k in kernels:
        kind_disp[k["kind"]] = kind_disp.get(k["kind"], 0) + 1
    dt_bytes = {}
    for w in manifest:
        dt_bytes[w["dtype"]] = dt_bytes.get(w["dtype"], 0) + w["bytes"]
    stats = {
        "model": os.path.basename(MODEL),
        "nodes": len(g.node),
        "kernels": len(kernels),
        "dispatch_budget_note": "compute = matmul*/conv/elementwise/reduce/softmax/control; layout targeted for compile-time elimination",
        "kernels_by_kind": kind_disp,
        "kinds_detail": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
        "unsupported": [k["op"] for k in kernels if k["kind"] == "UNSUPPORTED"],
        "weights": {"count": len(manifest), "total_bytes": off,
                    "bytes_by_dtype": dt_bytes},
        "fused_matmul_int8_dq": kind_disp.get("matmul_int8_dq", 0),
        "standalone_matmul_int8": kind_disp.get("matmul_int8", 0),
    }
    stats["compute_dispatches"] = sum(v for k, v in kind_disp.items()
                                      if k != "layout")
    stats["layout_ops"] = kind_disp.get("layout", 0)

    # ---- kernels.json ----
    doc = {"meta": {"model": stats["model"], "nodes": stats["nodes"],
                    "kernels": stats["kernels"],
                    "inputs": [i.name for i in g.input],
                    "outputs": [o.name for o in g.output]},
           "weights": manifest,
           "kernels": kernels[:MAX_JSON]}
    with open(os.path.join(OUT_DIR, "kernels.json"), "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    with open(os.path.join(OUT_DIR, "kgen-stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

    print(f"[kgen] nodes={stats['nodes']} -> kernels={stats['kernels']} "
          f"(compute={stats['compute_dispatches']} layout={stats['layout_ops']})")
    print(f"[kgen] fused matmul_int8_dq={stats['fused_matmul_int8_dq']} "
          f"standalone={stats['standalone_matmul_int8']}")
    print(f"[kgen] weights: {len(manifest)} tensors {off/1e6:.1f}MB "
          f"{ {k: round(v/1e6,1) for k,v in dt_bytes.items()} }")
    if stats["unsupported"]:
        print(f"[kgen] UNSUPPORTED ops: {sorted(set(stats['unsupported']))}")
        raise SystemExit("kgen: unsupported ops present")
    print("KGEN PASS")


if __name__ == "__main__":
    main()
