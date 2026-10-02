#!/usr/bin/env python3
"""Probe: exact ORT semantics of DQL -> MatMulInteger on one real tensor.
Reads chunk0 x from the volume cache, builds a mini ONNX graph, runs ORT
with optimizations disabled, prints y_scale/y_zero_point and the int acc row.
"""
import json
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

BIN = "/cache/ref/chunk0.bin"
MAN = "/cache/ref/chunk0.json"
WEIGHTS = "/cache/out/weights.bin"
KJ = "/cache/out/kernels.json"

man = json.load(open(MAN))
bin_ = open(BIN, "rb").read()


def refarr(name):
    m = man[name]
    dt = {"float32": np.float32, "bool": np.bool_, "int64": np.int64, "int32": np.int32}[m["dtype"]]
    return np.frombuffer(bin_[m["offset"]:m["offset"] + m["bytes"]], dtype=dt).reshape(m["shape"])


d = json.load(open(KJ))
k = next(kk for kk in d["kernels"] if kk.get("scale_tensor"))
x = refarr(k["inputs[0]" if False else "inputs"][0]).astype(np.float32)
wb = open(WEIGHTS, "rb").read()
wm = next(m for m in d["weights"] if m["name"] == k["inputs"][1])
W = np.frombuffer(wb[wm["offset"]:wm["offset"] + wm["bytes"]], dtype=np.int8).reshape(wm["shape"])
ref = refarr(k["outputs"][0])
lazy = float(refarr(k["scale_tensor"]).flatten()[0])
print("x", x.shape, "W", W.shape, "lazy", lazy)

g = helper.make_graph(
    [
        helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "ys", "yz"]),
        helper.make_node("MatMulInteger", ["y", "W", "yz", "wz"], ["acc"]),
        helper.make_node("Cast", ["acc"], ["accf"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["accf", "lazy"], ["out"]),
    ],
    "probe",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)],
    [helper.make_tensor_value_info("out", TensorProto.FLOAT, ref.shape),
     helper.make_tensor_value_info("ys", TensorProto.FLOAT, []),
     helper.make_tensor_value_info("yz", TensorProto.UINT8, []),
     helper.make_tensor_value_info("acc", TensorProto.INT32, ref.shape),
     helper.make_tensor_value_info("y", TensorProto.UINT8, x.shape)],
    [helper.make_tensor("W", TensorProto.INT8, W.shape, W.tobytes(), raw=True),
     helper.make_tensor("wz", TensorProto.INT8, [], bytes([0]), raw=True),
     helper.make_tensor("lazy", TensorProto.FLOAT, [], np.float32(lazy).tobytes(), raw=True)],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
outs = sess.run(None, {"x": x})
out, ys, yz, acc, y = outs
print("y_scale:", ys, "y_zp:", yz)
print("acc[0,0,:4] :", acc.flatten()[:4])
print("out[0,0,:4] :", out.flatten()[:4])
print("ref[0,0,:4] :", ref.flatten()[:4])
print("probe-vs-ref maxAbs:", np.abs(out - ref).max())
# my semantics for contrast
mn, mx = x.min(), x.max()
s = (mx - mn) / 255.0
zp = np.clip(np.round(mn / s), 0, 255)
q = np.clip(np.round(x / s) + zp, 0, 255).astype(np.int32) - zp
mine = q @ W.astype(np.int32)
print("my acc[0,0,:4] :", mine.flatten()[:4])
np.save("/cache/y_probe.npy", y)
mn_, mx_ = x.min(), x.max(); s_ = (mx_-mn_)/255.0
zp_ = np.clip(np.round(-mn_/s_), 0, 255)
q_ = np.clip(np.round(x/s_) + zp_, 0, 255).astype(np.uint8)
diff = (y != q_)
print("y-vs-mine quant mismatches:", int(diff.sum()), "/", y.size)
idx = np.argwhere(diff)[:5]
for i in idx:
    print("  at", i, "x=", float(x[tuple(i)]), "ort_y=", int(y[tuple(i)]), "mine=", int(q_[tuple(i)]))
