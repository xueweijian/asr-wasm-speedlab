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
from onnxsim import simplify
from onnx import numpy_helper


def _args():
    MODEL = sys.argv[1]
    OUT_DIR = sys.argv[2]
    MAX_JSON = int(sys.argv[sys.argv.index("--max-kernels-json") + 1]) if "--max-kernels-json" in sys.argv else 4000
    STATIC_SHAPES = "--dynamic" not in sys.argv
    return MODEL, OUT_DIR, MAX_JSON, STATIC_SHAPES


# importable module (ref.py uses static_shape_pass); argv parsed when run as script
MODEL, OUT_DIR, MAX_JSON, STATIC_SHAPES = _args() if __name__ == "__main__" else (None, None, 4000, True)


def static_shape_pass(m):
    """Re-simplify under the deployment contract: batch=1, chunk=77, fixed
    state shapes (D0-verified). Kills the dynamic-shape plumbing that the
    fold pass had to keep. Returns (simplified, parity_report|None)."""
    import os
    os.environ.setdefault("ONNXSIM_FIXED_POINT_ITERS", "300")
    overwrite = {}
    for vi in m.graph.input:
        dims = [d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 1
                for d in vi.type.tensor_type.shape.dim]
        overwrite[vi.name] = dims
    print(f"[static] overwrite_input_shapes: {list(overwrite.items())[:3]} ...")
    sm, ok = simplify(m, overwrite_input_shapes=overwrite)
    if not ok:
        raise SystemExit("static-shape simplify check failed")
    # numerical equivalence: onnxsim's internal check here + the D1 runtime
    # parity gate (WGSL vs ORT-orig) is the authoritative guard.
    return sm

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


def chain_fuse(kernels):
    """Greedy fusion of consecutive single-consumer elementwise/control/reduce
    kernels into one `fuseq` kernel (straight-line op sequence for runtime
    codegen). Intermediates must have exactly one consumer to be absorbable."""
    FUSABLE = {"elementwise", "control", "reduce"}
    cons = {}
    for i, k in enumerate(kernels):
        for t in k["inputs"]:
            cons.setdefault(t, []).append(i)

    def single_consumer(i):
        outs = kernels[i]["outputs"]
        cs = [c for t in outs for c in cons.get(t, []) if c != i]
        return cs[0] if len(cs) == 1 and len(outs) == 1 else None

    member = [False] * len(kernels)
    out = []
    for i, k in enumerate(kernels):
        if member[i]:
            continue
        if k["kind"] not in FUSABLE:
            out.append(k)
            continue
        chain = [i]
        member[i] = True
        cur = i
        while True:
            nxt = single_consumer(cur)
            if (nxt is not None and not member[nxt] and kernels[nxt]["kind"] in FUSABLE
                    and nxt > cur):
                chain.append(nxt)
                member[nxt] = True
                cur = nxt
            else:
                break
        if len(chain) == 1:
            out.append(kernels[i])
            continue
        internal = {t for j in chain[:-1] for t in kernels[j]["outputs"]}
        inputs = []
        for j in chain:
            for t in kernels[j]["inputs"]:
                if t not in internal and t not in inputs:
                    inputs.append(t)
        # per-step wiring: runtime codegen needs each step's actual in/out
        # tensor names to rebuild dataflow (intermediates stay in-kernel)
        seq = [{"op": kernels[j]["op"], "attrs": kernels[j]["attrs"],
                "inputs": list(kernels[j]["inputs"]),
                "outputs": list(kernels[j]["outputs"])}
               for j in chain]
        out.append({"kind": "fuseq", "op": "fuseq", "seq": seq,
                    "inputs": inputs, "outputs": kernels[chain[-1]]["outputs"],
                    "fused_from": len(chain)})
    n_fused = sum(k.get("fused_from", 0) for k in out if k["kind"] == "fuseq")
    n_chains = sum(1 for k in out if k["kind"] == "fuseq")
    print(f"[fuseq] {n_fused} nodes fused into {n_chains} chain kernels "
          f"(avg {n_fused/max(n_chains,1):.1f})")
    return out


DT_NAMES = {1: "float32", 2: "uint8", 3: "int8", 4: "int16", 5: "int32",
            6: "int64", 7: "bool", 9: "bool", 10: "float16", 11: "float64"}


def tensor_shape_table(m):
    """Shapes + dtypes for every tensor (graph io + intermediates) under the
    static contract — the runtime's buffer-allocation ground truth."""
    from onnx import shape_inference
    mi = shape_inference.infer_shapes(m, check_type=True, strict_mode=False)
    table = {}
    for coll in (mi.graph.value_info, mi.graph.input, mi.graph.output):
        for vi in coll:
            tt = vi.type.tensor_type
            dims = [d.dim_value for d in tt.shape.dim]
            table[vi.name] = {"shape": dims, "dtype": DT_NAMES.get(tt.elem_type, f"t{tt.elem_type}")}
    for init_ in mi.graph.initializer:
        table[init_.name] = {"shape": list(init_.dims),
                             "dtype": DT_NAMES.get(init_.data_type, f"t{init_.data_type}")}
    missing = 0
    return table


def main():
    m = onnx.load(MODEL)
    if STATIC_SHAPES:
        m = static_shape_pass(m)
    g = m.graph
    tensor_shapes = tensor_shape_table(m)

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
    fusions = {}           # mmi node idx -> fusion spec

    def in0_is_dql(n):
        """MatMulInteger(x, W): is x produced by DynamicQuantizeLinear?"""
        if not n.input or n.input[0] not in producer:
            return None
        p = g.node[producer[n.input[0]]]
        return p if p.op_type == "DynamicQuantizeLinear" else None

    def dq_chain(n):
        """Find Cast->fp32 then Mul(x_scale)[-> Mul(w_scale)] consumers."""
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

    # ---- pass 1: collect DQ+MMI fusions (retroactive, must precede emission) ----
    for idx, n in enumerate(g.node):
        if n.op_type != "MatMulInteger":
            continue
        dql = in0_is_dql(n)
        if dql is None or dql.output[0] != n.input[0]:
            continue
        chain, last = dq_chain(n)
        if chain and any(k == "mul" for k, *_ in chain):
            dql_idx = producer[n.input[0]]
            eaten = {dql_idx, idx} | {c for _, c, *_ in chain}
            scale_init = [r[0] for k, c, *r in chain if k == "mul" and r and r[0] in init]
            refs = [i for i in list(n.input[1:]) + list(scale_init) if i in init]
            w_name = n.input[1]
            extras = [r for r in refs if r != w_name]  # w_zp / w_scale inits
            fusions[idx] = {"dql": dql_idx, "inputs": [dql.input[0], n.input[1]] + extras,
                            "outputs": [last],
                            "fused": ["DynamicQuantizeLinear", "MatMulInteger"] +
                                     [g.node[c].op_type for _, c, *_ in chain],
                            "refs": refs}
            consumed |= eaten

    # ---- pass 2: emit kernels in topo order ----
    for idx, n in enumerate(g.node):
        if idx in consumed and idx not in fusions:
            continue
        op = n.op_type
        if idx in fusions:
            f = fusions[idx]
            weight_refs.update(f["refs"])
            kernels.append({"kind": "matmul_int8_dq", "op": op,
                            "inputs": f["inputs"], "outputs": f["outputs"],
                            "attrs": attrs_of(n), "fused": f["fused"]})
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
            for i in n.input:
                if i in init:
                    weight_refs.add(i)  # Pow exp / scalar operands etc.
            kernels.append({"kind": "elementwise", "op": op,
                            "inputs": list(n.input), "outputs": list(n.output),
                            "attrs": attrs_of(n)})
            continue
        if op in REDUCE:
            for i in n.input[1:]:  # axes tensor (opset<13) is an initializer
                if i in init:
                    weight_refs.add(i)
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
            for i in n.input:
                if i in init:
                    weight_refs.add(i)
            kernels.append({"kind": "control", "op": op,
                            "inputs": list(n.input), "outputs": list(n.output),
                            "attrs": attrs_of(n)})
            continue
        if op in LAYOUT:
            # layout param tensors (Slice starts/ends/axes/steps, Reshape
            # shape, Unsqueeze axes, Expand shape, Concat axis if input) are
            # initializers — pack them so the runtime resolves params host-side
            for i in n.input[1:]:
                if i in init:
                    weight_refs.add(i)
            kernels.append({"kind": "layout", "op": op,
                            "inputs": list(n.input), "outputs": list(n.output),
                            "attrs": attrs_of(n)})
            continue
        kernels.append({"kind": "UNSUPPORTED", "op": op,
                        "inputs": list(n.input), "outputs": list(n.output),
                        "attrs": attrs_of(n)})

    # ---- pass 3: elementwise/control/reduce chain fusion (fuseq) ----
    kernels = chain_fuse(kernels)

    # ---- weights packing (sorted for cross-run determinism; 4-byte aligned:
    # WGSL storage buffers have no 8-bit types — the arena is read as u32) ----
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest, blob, off = [], [], 0
    for name in sorted(weight_refs):
        t = init[name]
        arr = numpy_helper.to_array(t)
        b = arr.tobytes()
        pad = (-off) & 3
        if pad:
            blob.append(b"\x00" * pad)
            off += pad
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
    if len(kernels) > MAX_JSON:
        raise SystemExit(f"kgen: {len(kernels)} kernels > MAX_JSON {MAX_JSON}; "
                         "raise the cap instead of truncating the IR")
    doc = {"meta": {"model": stats["model"], "nodes": stats["nodes"],
                    "kernels": stats["kernels"],
                    "inputs": [i.name for i in g.input],
                    "outputs": [o.name for o in g.output]},
           "tensors": tensor_shapes,
           "weights": manifest,
           "kernels": kernels}
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
