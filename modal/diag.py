"""Modal 诊断：c2 链接缺符号审计。
问题：sherpa 链接报 undefined：EnvTime::EnvTime / onnx::GetOpSchema<*> / QDQ::GetAllNodeUnits / GetErrnoInfo
手段：llvm-nm 审合并档 + 全 build 树反查定义所在档，输出证据表。
用法: modal run modal/diag.py
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

app = modal.App("asr-ort-diag")

SYMS = ["EnvTime", "GetOpSchema", "GetAllNodeUnits", "GetErrnoInfo"]
ZIP_NAME = "onnxruntime-wasm-static_lib-simd-1.28.2.zip"
NM = "/opt/emsdk/upstream/bin/llvm-nm"


def sh(cmd: str, check=False) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"$ {cmd[:160]}")
    if r.returncode != 0 and check:
        raise RuntimeError(r.stderr[-2000:])
    return r.stdout + r.stderr


@app.function(image=image, volumes={"/cache": vol}, timeout=1200)
def diag() -> str:
    # 1) 合并档里有没有这些符号（定义 T/t/W）
    sh(f"rm -rf /tmp/pkg && mkdir -p /tmp/pkg && cd /tmp/pkg && unzip -q /cache/dist/{ZIP_NAME}")
    merged = "/tmp/pkg/lib/libonnxruntime.a"
    print("=== [1] merged archive symbol audit ===")
    for s in SYMS:
        out = sh(f"{NM} --defined-only {merged} 2>/dev/null | grep -c '{s}'")
        print(f"  defined '{s}': {out.strip()}")

    # 2) 全 build 树反查：哪些原始档定义了它们
    print("=== [2] original archive audit (where definitions SHOULD live) ===")
    archives = sh("find /cache/onnxruntime/build -name '*.a' | sort").split()
    print(f"  total archives: {len(archives)}")
    hits = {s: [] for s in SYMS}
    for a in archives:
        out = sh(f"{NM} --defined-only {a} 2>/dev/null | grep -oE '({('|'.join(SYMS))})' | sort -u")
        for s in set(out.split()):
            if s in hits:
                hits[s].append(a.replace("/cache/onnxruntime/build/", ""))
    for s in SYMS:
        print(f"  '{s}' defined in {len(hits[s])} archives:")
        for a in hits[s][:8]:
            print(f"    - {a}")

    # 3) 合并档成员数 vs 原始档成员总数（拷贝丢失检测）
    print("=== [3] member count: merged vs originals ===")
    m = sh(f"{NM} -A {merged} 2>/dev/null | grep -c '\\.o:' || true").strip()
    o = 0
    for a in archives:
        c = sh(f"{NM} -A {a} 2>/dev/null | grep -c '\\.o:' || true").strip() or "0"
        o += int(c or 0)
    print(f"  merged members: {m} / original total: {o}")

    # 4) 关键反例：EnvTime/GetOpSchema 在原始档的具体成员
    print("=== [4] sample defining members ===")
    for s, lst in hits.items():
        if lst:
            a = "/cache/onnxruntime/build/" + lst[0]
            out = sh(f"{NM} -A --defined-only {a} 2>/dev/null | grep '{s}' | head -4")
            print(f"  [{s}] in {lst[0]}:\n{out}")
    vol.commit()
    return "diag-done"


@app.local_entrypoint()
def main():
    print(diag.remote())
