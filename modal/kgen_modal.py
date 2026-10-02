"""kgen fast-iteration farm on Modal CPU.

CI round-trip is ~4min/push; this runs the same pipeline in ~20s once the
model/fold cache is warm (Volume kgen-cache).

Usage:
  modal run modal/kgen_modal.py               # first run: fetch model + fold + kgen
  modal run modal/kgen_modal.py --skip-fold   # fast loop: kgen only (fresh local code every run)

Outputs land in Volume /cache/out and are mirrored back to d1/out locally.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

app = modal.App("kgen-iter")
CACHE = modal.Volume.from_name("kgen-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("onnx", "onnxsim", "onnxruntime", "numpy")
    .add_local_file(str(ROOT / "d0" / "fold.py"), "/root/job/fold.py")
    .add_local_file(str(ROOT / "d1" / "kgen.py"), "/root/job/kgen.py")
    .add_local_file(str(ROOT / "d1" / "ref.py"), "/root/job/ref.py")
)

MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
             "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01.tar.bz2")


@app.function(image=image, volumes={"/cache": CACHE}, cpu=4, memory=4096, timeout=900)
def run(skip_fold: bool = False) -> dict:
    import glob
    import json
    import os
    import shutil
    import subprocess
    import tarfile
    import urllib.request

    model, wav, folded = "/cache/model.onnx", "/cache/test.wav", "/cache/folded.onnx"

    if not os.path.exists(model):
        tb = "/cache/ctc.tar.bz2"
        if not os.path.exists(tb):
            print("[farm] downloading model tarball...")
            urllib.request.urlretrieve(MODEL_URL, tb)
        with tarfile.open(tb) as t:
            names = t.getnames()
            t.extract([n for n in names if n.endswith("model.int8.onnx")][0], "/cache/x")
            t.extract(sorted(n for n in names if n.endswith(".wav"))[0], "/cache/x")
        shutil.move(glob.glob("/cache/x/**/model.int8.onnx", recursive=True)[0], model)
        shutil.move(glob.glob("/cache/x/**/*.wav", recursive=True)[0], wav)
        os.remove(tb)
        print(f"[farm] cached {os.path.getsize(model)/1e6:.1f}MB model + wav")

    if skip_fold and os.path.exists(folded):
        print("[iter] reuse cached folded.onnx")
    else:
        print("[farm] folding (D0 gate included)...")
        r = subprocess.run(["python", "/root/job/fold.py", model, folded,
                            "/cache/d0stats.json", "--wav", wav])
        if r.returncode:
            raise RuntimeError("fold/D0 gate failed")

    print("[iter] kgen (fresh local code)...")
    r = subprocess.run(["python", "/root/job/kgen.py", folded, "/cache/out"])
    if r.returncode:
        raise RuntimeError("kgen failed")

    stats = json.load(open("/cache/out/kgen-stats.json"))
    kernels_raw = open("/cache/out/kgen-stats.json").read()
    kj = json.load(open("/cache/out/kernels.json"))
    CACHE.commit()
    return {"stats": stats, "kernels": kj, "kernels_json_bytes": len(kernels_raw)}


@app.function(image=image, volumes={"/cache": CACHE}, cpu=4, memory=4096, timeout=900)
def make_ref():
    """Generate parity ground truth from the cached folded.onnx."""
    import subprocess
    r = subprocess.run(["python", "/root/job/ref.py", "/cache/folded.onnx",
                        "/cache/test.wav", "/cache/ref"])
    if r.returncode:
        raise RuntimeError("ref generation failed")
    import os
    CACHE.commit()
    return {f: os.path.getsize(f"/cache/ref/{f}") for f in
            sorted(os.listdir("/cache/ref"))}


@app.function(image=image, volumes={"/cache": CACHE}, cpu=2, memory=2048, timeout=300)
def pull_artifacts() -> dict:
    import base64
    out = {}
    for f in ("kernels.json", "kgen-stats.json", "weights.bin"):
        out[f] = base64.b64encode(open(f"/cache/out/{f}", "rb").read()).decode()
    return out


@app.local_entrypoint()
def main(skip_fold: bool = False, ref: bool = False, pull: bool = False):
    if pull:
        import base64
        res = pull_artifacts.remote()
        outdir = ROOT / "d1" / "out"
        outdir.mkdir(exist_ok=True)
        for f, b64 in res.items():
            (outdir / f).write_bytes(base64.b64decode(b64))
            print(f"[local] {f} {len(b64) * 3 // 4 / 1e6:.1f}MB")
        return
    if ref:
        print(make_ref.remote())
        return
    res = run.remote(skip_fold=skip_fold)
    s = res["stats"]
    keys = ("nodes", "kernels", "compute_dispatches", "layout_ops",
            "fused_matmul_int8_dq", "standalone_matmul_int8", "unsupported")
    print("\n=== kgen verdict ===")
    for k in keys:
        print(f"  {k}: {s.get(k)}")
    out = ROOT / "d1" / "out"
    out.mkdir(exist_ok=True)
    import json as _j
    (out / "kgen-stats.json").write_text(_j.dumps(s, indent=1))
    (out / "kernels.json").write_text(_j.dumps(res["kernels"], separators=(",", ":")))
    print("[local] mirrored -> d1/out/kgen-stats.json, kernels.json")
