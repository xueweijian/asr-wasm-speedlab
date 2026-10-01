"""Modal 诊断第 3 轮：成员内容取证 + MRI 合并修复原型。
事实链: 成员在合并档里但符号没了 → 提取/重归档过程损坏了成员内容
修复假设: llvm-ar MRI 脚本 (ADDLIB) 合并可逐字节保留成员与符号表
用法: modal run modal/diag3.py
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
app = modal.App("asr-ort-diag3")
ZIP_NAME = "onnxruntime-wasm-static_lib-simd-1.28.2.zip"
NM = "/opt/emsdk/upstream/bin/llvm-nm"
AR = "/opt/emsdk/upstream/bin/llvm-ar"


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr)


@app.function(image=image, volumes={"/cache": vol}, timeout=1800)
def diag() -> str:
    sh(f"rm -rf /tmp/pkg && mkdir -p /tmp/pkg && cd /tmp/pkg && unzip -q /cache/dist/{ZIP_NAME}")
    merged = "/tmp/pkg/lib/libonnxruntime.a"

    print("=== [A] member content forensics ===")
    for m in ["libonnxruntime_common__env.cc.o", "libonnxruntime_common__env_time.cc.o",
              "libonnxruntime_optimizer__utils.cc.o"]:
        sh(f"rm -rf /tmp/one && mkdir -p /tmp/one && cd /tmp/one && emar x {merged} {m} 2>&1 | head -2")
        sz = sh(f"stat -c %s /tmp/one/{m} 2>/dev/null").strip()
        syms = sh(f"{NM} --defined-only /tmp/one/{m} 2>&1 | grep -cE 'GetErrnoInfo|EnvTime|GetAllNodeUnits'").strip()
        raw = sh(f"{NM} /tmp/one/{m} 2>&1 | head -3").strip()
        print(f"  {m}: size={sz} key_syms={syms} nm_head={raw[:100]}")

    print("=== [B] fresh original member for comparison ===")
    sh("rm -rf /tmp/orig && mkdir -p /tmp/orig && cd /tmp/orig && "
       "emar x /cache/onnxruntime/build/Linux/Release/libonnxruntime_common.a env.cc.o env_time.cc.o 2>&1 | head -2")
    for m in ["env.cc.o", "env_time.cc.o"]:
        sz = sh(f"stat -c %s /tmp/orig/{m} 2>/dev/null").strip()
        syms = sh(f"{NM} --defined-only /tmp/orig/{m} 2>&1 | grep -cE 'GetErrnoInfo|EnvTime'").strip()
        print(f"  orig {m}: size={sz} key_syms={syms}")

    print("=== [C] MRI ADDLIB merge prototype (the fix) ===")
    archives = sorted(sh("find /cache/onnxruntime/build -name '*.a'").split())
    print(f"merging {len(archives)} archives via MRI script")
    script_lines = ["CREATE /tmp/mri-merged.a"]
    for a in archives:
        script_lines.append(f"ADDLIB {a}")
    script_lines += ["SAVE", "END"]
    pathlib.Path("/tmp/mri.txt").write_text("\n".join(script_lines) + "\n")
    r = subprocess.run([AR, "-M"], input="\n".join(script_lines), capture_output=True, text=True)
    print(f"  llvm-ar -M rc={r.returncode} err={r.stderr[-300:]}")
    info = sh(f"stat -c %s /tmp/mri-merged.a 2>/dev/null").strip()
    cnt = sh(f"{AR} t /tmp/mri-merged.a 2>/dev/null | wc -l").strip()
    print(f"  mri-merged.a: {info} bytes, {cnt} members")
    for pat in ["EnvTimeC2Ev", "GetErrnoInfoEv", "GetAllNodeUnits", "GetOpSchema", "EnvC2Ev"]:
        n = sh(f"{NM} --defined-only /tmp/mri-merged.a 2>/dev/null | grep -c '{pat}'").strip()
        print(f"  '{pat}' defined: {n}")

    if pathlib.Path("/tmp/mri-merged.a").exists() and int(info or 0) > 10_000_000:
        sh("cp /tmp/mri-merged.a /cache/dist/libonnxruntime-mri-prototype.a")
        print("  prototype saved to volume: /cache/dist/libonnxruntime-mri-prototype.a")
    vol.commit()
    return "diag3-done"


@app.local_entrypoint()
def main():
    print(diag.remote())
