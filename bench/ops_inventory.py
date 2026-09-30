"""模型资产清点：算子集合 + 权重体积分解 —— C 轨瘦身的前置测量
产物: results/ops-inventory.json
"""
import json
import os
import subprocess
import sys
from collections import Counter

MODELS = {
    "zipformer-14m": [
        ("encoder.onnx", "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2",
         "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/encoder-epoch-99-avg-1.int8.onnx"),
        ("decoder.onnx", None, "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/decoder-epoch-99-avg-1.int8.onnx"),
        ("joiner.onnx", None, "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/joiner-epoch-99-avg-1.int8.onnx"),
    ],
    "small-ctc": [
        ("model.onnx", "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01.tar.bz2",
         "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/model.int8.onnx"),
    ],
}


def main():
    import onnx
    out = {}
    for family, files in MODELS.items():
        tar = None
        fam_out = {"files": {}}
        all_ops = Counter()
        for name, url, member in files:
            if url:
                tar = url.rsplit("/", 1)[-1]
                if not os.path.exists(tar):
                    subprocess.run(["curl", "-sSL", "-o", tar, url], check=True)
                subprocess.run(["tar", "xf", tar], check=True)
            m = onnx.load(member, load_external_data=False)
            ops = Counter(n.op_type for n in m.graph.node)
            all_ops.update(ops)
            init_bytes = sum(len(t.raw_data) for t in m.graph.initializer if t.raw_data)
            fam_out["files"][name] = {
                "size_bytes": os.path.getsize(member),
                "nodes": len(m.graph.node),
                "initializers": len(m.graph.initializer),
                "initializer_bytes": init_bytes,
                "ops": dict(sorted(ops.items(), key=lambda x: -x[1])),
                "opset": [f"{o.domain or 'ai.onnx'}:{o.version}" for o in m.opset_import],
            }
        fam_out["op_union"] = sorted(all_ops)
        fam_out["op_union_count"] = len(all_ops)
        out[family] = fam_out

    os.makedirs("results", exist_ok=True)
    with open("results/ops-inventory.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for fam, d in out.items():
        print(f"== {fam}: {d['op_union_count']} unique ops")
        print("  ", d["op_union"])
        for name, fd in d["files"].items():
            print(f"  {name}: {fd['size_bytes']/1e6:.1f}MB, {fd['nodes']} nodes, init {fd['initializer_bytes']/1e6:.1f}MB")


if __name__ == "__main__":
    sys.exit(main())
