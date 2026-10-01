"""Modal 编译农场：ORT wasm 最小算子库 构建+打包+push+触发 site 终审 一条龙。

用法:
  modal secret create gh-token GITHUB_TOKEN=<repo-write-token>   # 一次性
  modal run modal/build_ort.py                       # 全链（ORT + sherpa c2 产物）
  modal run modal/build_ort.py --package-only        # 只重打包（源码/构建已缓存，秒级）
  modal run modal/build_ort.py --sherpa-only         # 只构建 sherpa c2（zip 取 volume 或仓库 main）

设计要点:
  - 源码 + build 目录放 Modal Volume → ninja 增量，改打包脚本不重编
  - 构建 -j16 + 32GB；晚败容忍（测试目标挂但 .a>=80 就继续，同 CI 救援路径）
  - 打包 = 全 .a 闭包 emar 合并 + env-ctor shim（对齐 ort-minimal.yml 最新版）
  - sherpa 构建 = c2 变体最终链接（原在 site.yml 内做，日志被 Actions 锁死，2026-10-01
    第 12 轮 c2 失败产物缺失且不可归因 → 链接环节整体迁入本农场，stdout 全可见）
  - sherpa 产物只提交变体特有三件套（.wasm + 两个 .js ≈13MB），
    24MB 模型 .data 不重复入库（site.yml 从 a0 构建产物补齐，同输入同产物）
  - 容器网络干净：git push + api.github.com 直连，不经沙箱
"""
import hashlib
import os
import pathlib
import subprocess

import modal

ORT_REF = "v1.28.2"
SHERPA_REF = "v1.13.8"  # 对齐 site.yml pinned（≥v1.13.0 导出 HEAPF32）
REPO = "xueweijian/asr-wasm-speedlab"
ZIP_NAME = "onnxruntime-wasm-static_lib-simd-1.28.2.zip"
MODEL_TARBALL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
                 "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2")

app = modal.App("asr-ort-builder")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "unzip", "zip", "curl", "ca-certificates", "xz-utils", "ninja-build", "build-essential", "python3-pip")
    .run_commands(
        "curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz | tar -xJ -C /usr/local --strip-components=1",
        "node --version && npm --version",
    )
    .pip_install("cmake>=3.28")  # Debian 仓库 cmake=3.25 太老，ORT v1.28.2 要 ≥3.28
    .run_commands(
        "git clone --depth 1 https://github.com/emscripten-core/emsdk /opt/emsdk",
        "cd /opt/emsdk && ./emsdk install 4.0.23 && ./emsdk activate 4.0.23",
    )
    .env({"EMSDK": "/opt/emsdk",
          "PATH": "/opt/emsdk:/opt/emsdk/upstream/emscripten:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
)

vol = modal.Volume.from_name("ort-wasm-cache", create_if_missing=True)


def sh(cmd: str, cwd=None, check=True, quiet=False) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=dict(os.environ), capture_output=True, text=True)
    if not quiet:
        print(f"$ {cmd}")
    if r.returncode != 0:
        print(r.stdout[-30000:])
        print(r.stderr[-30000:])
        if check:
            raise RuntimeError(f"FAILED ({r.returncode}): {cmd}")
    return r


def git_push_files(files: dict, msg: str, dispatch_site: bool):
    """files: {repo-relative-path: container-abs-path}；有变化才 commit+push，成功后可触发 site。"""
    tok = os.environ["GITHUB_TOKEN"]
    sh(f"rm -rf /tmp/repo && git clone --depth 1 https://x-access-token:{tok}@github.com/{REPO} /tmp/repo", quiet=True)
    sh("rm -f .git/index.lock && git config user.name bench-bot && git config user.email bot@users.noreply.github.com",
       cwd="/tmp/repo")
    for rel, src in files.items():
        sh(f"mkdir -p $(dirname /tmp/repo/{rel}) && cp {src} /tmp/repo/{rel}")
    sh("git add -A", cwd="/tmp/repo")
    r = sh('git diff --cached --quiet || git commit -m "' + msg + '"', cwd="/tmp/repo", check=False)
    if r.returncode != 0:  # 无变化
        print("[push] no changes, skip")
        return False
    sh("git pull --rebase -X ours", cwd="/tmp/repo", check=False)
    sh("git push origin main", cwd="/tmp/repo")
    if dispatch_site:
        sh(f'curl -s -X POST -H "Authorization: token {tok}" '
           f'-H "Accept: application/vnd.github+json" '
           f'"https://api.github.com/repos/{REPO}/actions/workflows/site.yml/dispatches" '
           f"-d '{{\"ref\":\"main\"}}' -w 'dispatch=%{{http_code}}\\n'")
    return True


@app.function(image=image, volumes={"/cache": vol}, secrets=[modal.Secret.from_name("gh-token")],
              cpu=16, memory=32768, timeout=1800)
def build(package_only: bool = False, push: bool = True, dispatch_site: bool = True, clean: bool = False) -> str:
    ort = pathlib.Path("/cache/onnxruntime")
    build_dir = ort / "build/Linux/Release"

    if not package_only and not (ort / ".git").exists():
        sh(f"git clone --depth 1 --branch {ORT_REF} --recurse-submodules --shallow-submodules --jobs 8 "
           f"https://github.com/microsoft/onnxruntime {ort}")

    if clean:
        sh("rm -rf /cache/onnxruntime/build")
        print("[clean] build dir wiped (CMakeCache 持久化 NOTFOUND 的坑)")

    # ---- build (incremental on volume) ----
    nlib = 0
    if not package_only:
        os.chdir(ort)
        sh("pip install -q -r requirements.txt")
        sh(f"mkdir -p {build_dir}/testdata && echo ok > {build_dir}/testdata/dummy.txt")
        r = sh("./build.sh --build_wasm --enable_wasm_simd --parallel 16 "
               "--disable_ml_ops --disable_contrib_ops --skip_tests --config Release "
               "--allow_running_as_root",  # modal 容器默认 root，ORT 检查需显式放行
               check=False)
        pathlib.Path("/cache/build-stdout.log").write_text(r.stdout[-200000:])
        pathlib.Path("/cache/build-stderr.log").write_text(r.stderr[-200000:])
        nlib = int(sh("find build -name '*.a' | wc -l", quiet=True).stdout.strip())
        print(f"[build] rc={r.returncode} libs={nlib}")
        if nlib < 80:
            raise RuntimeError(f"early build failure (libs={nlib})")

    # ---- package: closure merge + shim ----
    if nlib == 0:
        nlib = int(sh(f"find {ort}/build -name '*.a' | wc -l", quiet=True).stdout.strip())
    print(f"[package] merging {nlib} archives")
    sh("rm -rf /tmp/objs /tmp/pkg && mkdir -p /tmp/objs /tmp/pkg/include /tmp/pkg/lib")
    sh(f"""for a in $(find {ort}/build -name '*.a' | sort); do
      d="/tmp/m$(basename $a .a)"; mkdir -p "$d"; (cd "$d" && emar x "$a" 2>/dev/null || ar x "$a")
      for o in "$d"/*.o; do [ -e "$o" ] || continue; cp "$o" "/tmp/objs/$(basename $a .a)__$(basename $o)"; done
    done""")
    n = int(sh("ls /tmp/objs | wc -l", quiet=True).stdout.strip())
    print(f"[package] {n} objects")
    if n < 500:
        raise RuntimeError(f"too few objects ({n})")

    # env-ctor shim: LTO 不发射库内无引用的 defaulted ctor，空体与 = default 等价（NSDMI 照跑）
    gsl_inc = sh(f"find {ort}/build -type d -path '*gsl-src/include' | head -1", quiet=True).stdout.strip()
    print(f"[shim] gsl include: {gsl_inc}")
    sh("""printf '#include "core/platform/env.h"\\nnamespace onnxruntime { Env::Env() {} }\\n' > /tmp/env_shim.cc
      emcc -c /tmp/env_shim.cc -I""" + str(ort) + """/include/onnxruntime -I""" + str(ort) + """/onnxruntime -I""" + gsl_inc + """ -flto -o /tmp/objs/libshim__env_ctor.o""")

    sh("cd /tmp/objs && emar rc /tmp/pkg/lib/libonnxruntime.a *.o && emranlib /tmp/pkg/lib/libonnxruntime.a")
    members = sh('emar t /tmp/pkg/lib/libonnxruntime.a | wc -l', quiet=True).stdout.strip()
    print(f"[package] merged members: {members}")

    sh(f"cp {ort}/include/onnxruntime/core/session/*.h /tmp/pkg/include/")
    sh(f"mkdir -p /cache/dist && rm -f /cache/dist/{ZIP_NAME} /tmp/{ZIP_NAME}")
    sh(f"cd /tmp/pkg && zip -qr /tmp/{ZIP_NAME} . && cp /tmp/{ZIP_NAME} /cache/dist/")

    digest = hashlib.sha256(pathlib.Path(f"/tmp/{ZIP_NAME}").read_bytes()).hexdigest()
    size = pathlib.Path(f"/tmp/{ZIP_NAME}").stat().st_size
    print(f"[package] {ZIP_NAME}: {size/1e6:.1f}MB sha256={digest[:16]}…")

    if push:
        git_push_files({f"wasm/third_party/{ZIP_NAME}": f"/tmp/{ZIP_NAME}"},
                       "build(modal): ort wasm minimal-op closure lib", dispatch_site)
    vol.commit()
    return digest


@app.function(image=image, volumes={"/cache": vol}, secrets=[modal.Secret.from_name("gh-token")],
              cpu=16, memory=32768, timeout=1800)
def sherpa(push_assets: bool = True, dispatch_site: bool = True, jobs: int = 16) -> str:
    """c2 最终链接：sherpa-onnx wasm against 闭包 ORT 库。全程 stdout 可见。"""
    sherpa_dir = pathlib.Path("/cache/sherpa-onnx")
    if not (sherpa_dir / ".git").exists():
        sh(f"git clone --depth 1 --branch {SHERPA_REF} https://github.com/k2-fsa/sherpa-onnx {sherpa_dir}")
    else:
        sh("git checkout -- . && git clean -fd", cwd=sherpa_dir, check=False)  # 清上轮 sed 残留

    # ---- 模型资产（volume 缓存，file_packager 构建期打入 .data）----
    assets = sherpa_dir / "wasm/asr/assets"
    if not (assets / "encoder.onnx").exists():
        sh(f"mkdir -p {assets} && cd {assets} && curl -sSL -o m.tar.bz2 {MODEL_TARBALL} && tar xf m.tar.bz2 && "
           "D=sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 && "
           "mv $D/encoder-epoch-99-avg-1.int8.onnx encoder.onnx && "
           "mv $D/decoder-epoch-99-avg-1.int8.onnx decoder.onnx && "
           "mv $D/joiner-epoch-99-avg-1.int8.onnx joiner.onnx && "
           "mv $D/tokens.txt . && rm -rf $D m.tar.bz2")
    sh(f"ls -la {assets}")

    # ---- 闭包库 zip：volume 优先，缺则仓库 main ----
    z = pathlib.Path(f"/cache/dist/{ZIP_NAME}")
    if not z.exists():
        sh(f"mkdir -p /cache/dist && curl -sSL -o {z} https://raw.githubusercontent.com/{REPO}/main/wasm/third_party/{ZIP_NAME}")
    digest = hashlib.sha256(z.read_bytes()).hexdigest()
    print(f"[zip] {z.stat().st_size/1e6:.1f}MB sha256={digest[:16]}…")
    sh(f"cp {z} {sherpa_dir}/{ZIP_NAME}")
    sh(f"""cd {sherpa_dir} && sed -i "s/set(onnxruntime_HASH .*)/set(onnxruntime_HASH SHA256={digest})/" cmake/onnxruntime-wasm-simd.cmake""")
    sh(f"grep -n onnxruntime_HASH {sherpa_dir}/cmake/onnxruntime-wasm-simd.cmake | head -2")

    # ---- 构建（对齐 site.yml c2 步骤的 sed 序列）----
    sh(f"cd {sherpa_dir} && sed -i 's/build-wasm-simd-asr/build-wasm-minort/g' build-wasm-simd-asr.sh")
    sh(f"cd {sherpa_dir} && sed -i 's/make -j2/make -j{jobs}/' build-wasm-simd-asr.sh")
    r = sh(f"cd {sherpa_dir} && ./build-wasm-simd-asr.sh > /cache/sherpa-build.log 2>&1", check=False)
    log = pathlib.Path("/cache/sherpa-build.log").read_text(errors="replace")
    print(f"[sherpa] build rc={r.returncode}, log {len(log)} chars; tail:")
    print(log[-6000:])
    if r.returncode != 0:
        raise RuntimeError(f"sherpa build failed rc={r.returncode} (full log in volume /cache/sherpa-build.log)")

    install = sherpa_dir / "build-wasm-minort/install/bin/wasm/asr"
    ls = sh(f"ls -la {install}", check=False, quiet=True).stdout
    print(f"[sherpa] install dir:\n{ls}")
    if not (install / "sherpa-onnx-wasm-main-asr.wasm").exists():
        raise RuntimeError(f"c2 products missing in {install}")

    if push_assets:
        files = {f"site-assets/c2-minort/{f}": str(install / f)
                 for f in ["sherpa-onnx-asr.js", "sherpa-onnx-wasm-main-asr.js", "sherpa-onnx-wasm-main-asr.wasm"]}
        git_push_files(files, "build(modal): c2 sherpa wasm (closure ort link, farm-built)", dispatch_site)
    vol.commit()
    return digest


@app.local_entrypoint()
def main(package_only: bool = False, sherpa_only: bool = False, push: bool = True,
         no_dispatch: bool = False, clean: bool = False):
    if sherpa_only:
        d = sherpa.remote(push_assets=push, dispatch_site=not no_dispatch)
    else:
        d = build.remote(package_only=package_only, push=push, dispatch_site=not no_dispatch, clean=clean)
        if not package_only:
            sherpa.remote(push_assets=push, dispatch_site=not no_dispatch)
    print(f"DONE sha256={d}")
