"""官方 sherpa-onnx 原生推理基线 — 供 GitHub Actions arm64 runner 使用
自动发现 models/ 下的 zipformer-14m 与 zipformer-small-ctc，流式解码计 RTF。
输出: JSON 结果 + markdown 表格，作为 speedlab 所有浏览器实验的对比锚点。
"""
import argparse
import glob
import json
import os
import platform
import time
import wave

import numpy as np
import sherpa_onnx

CHUNK_S = 0.32  # 流式喂入块大小（模拟浏览器 mic 场景）
RUNS = 3


def load_wav(path):
    with wave.open(path, "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        data = w.readframes(n)
    a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != 16000:  # 8k 电话信道考卷：线性插值升到 16k
        n = int(len(a) / sr * 16000)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
        sr = 16000
    return a, sr


def pick(patterns):
    for p in patterns:
        m = sorted(glob.glob(p))
        if m:
            return m[0]
    return None


def find_models(root):
    """在 models/ 下自动发现两个目标模型目录"""
    m14 = mctc = None
    for d in sorted(glob.glob(os.path.join(root, "*") + "/")):
        if glob.glob(os.path.join(d, "encoder*int8.onnx")) or glob.glob(os.path.join(d, "encoder*.onnx")):
            m14 = d
        if glob.glob(os.path.join(d, "model.int8.onnx")):
            mctc = d
    return m14, mctc


def make_recognizer(model_id, d, threads):
    t0 = time.perf_counter()
    if model_id == "zipformer-14m":
        enc = pick([d + "encoder*int8.onnx", d + "encoder*.onnx"])
        dec = pick([d + "decoder*int8.onnx", d + "decoder*.onnx"])
        join = pick([d + "joiner*int8.onnx", d + "joiner*.onnx"])
        r = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=d + "tokens.txt", encoder=enc, decoder=dec, joiner=join,
            num_threads=threads, provider="cpu")
    else:
        mdl = pick([d + "model.int8.onnx", d + "model.onnx"])
        r = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=d + "tokens.txt", model=mdl, num_threads=threads, provider="cpu")
    return r, (time.perf_counter() - t0) * 1000


def run_stream(rec, samples, sr):
    chunk = int(CHUNK_S * sr)
    stream = rec.create_stream()
    dms = 0.0
    for i in range(0, len(samples), chunk):
        t0 = time.perf_counter()
        stream.accept_waveform(sr, samples[i:i + chunk])
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        dms += (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    text = rec.get_result(stream)
    dms += (time.perf_counter() - t0) * 1000
    return dms, text


def cpuinfo():
    model = "?"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name") or line.startswith("CPU part"):
                model = line.split(":", 1)[1].strip()
                if "model name" in line:
                    break
    except Exception:
        pass
    return {"arch": platform.machine(), "cpu": model, "cores": os.cpu_count(),
            "runner": os.environ.get("RUNNER_NAME", "local"),
            "sha": os.environ.get("GITHUB_SHA", "")[:7],
            "sherpa_onnx": sherpa_onnx.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--out", default="results/native-baseline.json")
    ap.add_argument("--max-wavs", type=int, default=4)
    args = ap.parse_args()

    m14, mctc = find_models(args.models_root)
    print(f"discovered: 14m={m14} ctc={mctc}")
    assert m14 or mctc, "no model found under models-root"

    # 考卷：优先各模型自带 test_wavs，缺则互备
    w14 = sorted(glob.glob(m14 + "test_wavs/*.wav"))[: args.max_wavs] if m14 else []
    wctc = sorted(glob.glob(mctc + "test_wavs/*.wav"))[: args.max_wavs] if mctc else []
    pool = list(dict.fromkeys(w14 + wctc)) or sorted(glob.glob(args.models_root + "/*/test_wavs/*.wav"))[: args.max_wavs]
    print("fixtures:", [os.path.basename(w) for w in pool])

    audio = {p: load_wav(p) for p in pool}
    results, init_ms_map = [], {}

    for model_id, d in [("zipformer-14m", m14), ("zipformer-small-ctc", mctc)]:
        if not d:
            continue
        for threads in [1, 2, 4]:
            try:
                rec, init_ms = make_recognizer(model_id, d, threads)
                init_ms_map[f"{model_id}/t{threads}"] = round(init_ms)
                run_stream(rec, audio[pool[0]][0], 16000)  # warmup
                for p in pool:
                    samples, sr = audio[p]
                    dur = len(samples) / sr
                    best = None
                    for _ in range(RUNS):
                        dms, text = run_stream(rec, samples, sr)
                        if best is None or dms < best[0]:
                            best = (dms, text)
                    dms, text = best
                    rtf = dms / 1000.0 / dur
                    results.append(dict(
                        model=model_id, threads=threads, wav=os.path.basename(p),
                        audio_s=round(dur, 2), decode_ms=round(dms, 1),
                        rtf=round(rtf, 4), x_realtime=round(1 / rtf, 1), text=text))
                    print(f"{model_id:20s} thr={threads} {os.path.basename(p):10s} "
                          f"{dur:5.1f}s {dms:7.0f}ms RTF={rtf:.3f} {1/rtf:6.1f}x  {text[:24]}")
                del rec
            except Exception as e:
                print(f"[WARN] {model_id} threads={threads} failed: {e}")

    out = {"meta": dict(env=cpuinfo(), chunk_s=CHUNK_S, runs=RUNS, mode="streaming",
                        note="官方sherpa-onnx原生CPU基线(非wasm)", init_ms=init_ms_map),
           "results": results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
