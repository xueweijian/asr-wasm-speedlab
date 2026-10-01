"""Modal 诊断第 2 轮：成员级取证。
Q: 合并档丢成员的机制是什么？
手段:
  A. emar t 成员表对比：合并档 vs 原始档逐档 (llvm-ar t)，输出丢失成员清单
  B. 对涉事原始档做单档 emar x 提取实验（提取是否本身失败）
  C. 精确 mangled 符号核对（C1/C2 ctor 区分）
用法: modal run modal/diag2.py
"""
import pathlib
import subprocess

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "unzip", "curl", "binutils")
    .run_commands(
        "git clone --depth 1 https://github.com/emscripten-core/emsdk /opt/emsdk",
        "cd /opt/emsdk && ./emsdk install 4.0.23 && ./emsdk activate 4.0.23",
    )
    .env({"EMSDK": "/opt/emsdk",
          "PATH": "/opt/emsdk:/opt/emsdk/upstream/emscripten:/opt/emsdk/upstream/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
)

vol = modal.Volume.from_name("ort-wasm-cache", create_if_missing=True)
app = modal.App("asr-ort-diag2")
ZIP_NAME = "onnxruntime-wasm-static_lib-simd-1.28.2.zip"


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr)


@app.function(image=image, volumes={"/cache": vol}, timeout=1200)
def diag() -> str:
    sh(f"rm -rf /tmp/pkg && mkdir -p /tmp/pkg && cd /tmp/pkg && unzip -q /cache/dist/{ZIP_NAME}")
    merged = "/tmp/pkg/lib/libonnxruntime.a"
    print("=== [A] member-table diff ===")
    merged_members = set(sh(f"emar t {merged}").split())
    print(f"merged members: {len(merged_members)}")
    archives = sorted(sh("find /cache/onnxruntime/build -name '*.a'").split())
    orig_total, lost, dup_dir = 0, [], []
    seen_dirs = {}
    for a in archives:
        base = pathlib.Path(a).stem
        members = sh(f"emar t {a}").split()
        orig_total += len(members)
        d = seen_dirs.setdefault(base, a)
        if d != a:
            dup_dir.append((d, a))
        for m in members:
            key = f"{base}__{m}"
            if key not in merged_members:
                lost.append((base, m))
    print(f"original members total: {orig_total}; lost in merge: {len(lost)}")
    from collections import Counter
    c = Counter(b for b, _ in lost)
    print("lost by archive (top 12):")
    for b, n in c.most_common(12):
        print(f"  {b}: {n}")
    print("same-basename archive collisions (extraction dir reuse bug):")
    for x in dup_dir[:10]:
        print(f"  {x[0]}  <->  {x[1]}")

    print("=== [B] single-archive extraction test ===")
    for probe in ["libonnxruntime_common.a", "libonnx.a", "libonnxruntime_optimizer.a"]:
        p = sh(f"find /cache/onnxruntime/build -name '{probe}' | head -1").strip()
        if not p:
            continue
        before = sh(f"emar t {p} | wc -l").strip()
        sh(f"rm -rf /tmp/ex && mkdir -p /tmp/ex && cd /tmp/ex && emar x {p}")
        got = sh("ls /tmp/ex | wc -l").strip()
        got_o = sh("ls /tmp/ex/*.o 2>/dev/null | wc -l").strip()
        print(f"  {probe}: members={before} extracted_all={got} extracted_o={got_o}")

    print("=== [C] exact mangled symbols ===")
    for pat in ["EnvTimeC1Ev", "EnvTimeC2Ev", "GetErrnoInfoEv", "GetAllNodeUnits", "GetOpSchemaINS_12GlobalLpPool"]:
        orig = sh(f"/opt/emsdk/upstream/bin/llvm-nm --defined-only /cache/onnxruntime/build/Linux/Release/libonnxruntime_common.a /cache/onnxruntime/build/Linux/Release/_deps/onnx-build/libonnx.a /cache/onnxruntime/build/Linux/Release/libonnxruntime_optimizer.a 2>/dev/null | grep -c '{pat}'").strip()
        mg = sh(f"/opt/emsdk/upstream/bin/llvm-nm --defined-only {merged} 2>/dev/null | grep -c '{pat}'").strip()
        print(f"  {pat}: orig3libs={orig} merged={mg}")

    print("=== [D] is the defining member present? ===")
    for m in merged_members:
        if m.endswith(("__env.cc.o", "__env_time.cc.o", "__old.cc.o", "__utils.cc.o", "__defs.cc.o")):
            print(f"  present: {m}")
    vol.commit()
    return "diag2-done"


@app.local_entrypoint()
def main():
    print(diag.remote())
