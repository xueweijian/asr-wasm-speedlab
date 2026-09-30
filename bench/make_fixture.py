"""生成浏览器考台混音考卷：语音 + 静音交替（VAD 门控实验需要静音段）
从 small-ctc 模型包 test_wavs 取真实语音，拼接 [语音 1.5s静音 语音 2s静音]，
输出 Chrome --use-file-for-fake-audio-capture 兼容的 16bit PCM WAV。
"""
import argparse
import glob
import os
import wave

import numpy as np


def load_wav(path):
    with wave.open(path, "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        data = w.readframes(n)
    a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        n2 = int(len(a) / sr * 16000)
        a = np.interp(np.linspace(0, len(a) - 1, n2), np.arange(len(a)), a).astype(np.float32)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--out", default="/tmp/fixture.wav")
    args = ap.parse_args()

    wavs = []
    for d in sorted(glob.glob(os.path.join(args.models_root, "*") + "/")):
        wavs = sorted(glob.glob(d + "test_wavs/*.wav"))
        if wavs:
            break
    assert wavs, "no test_wavs found"
    speech1 = load_wav(wavs[0])
    speech2 = load_wav(wavs[min(1, len(wavs) - 1)])
    silence = lambda sec: np.zeros(int(16000 * sec), dtype=np.float32)

    # 静音垫一点极低噪声，模拟真实底噪（纯零会让噪声底校准过于理想）
    rng = np.random.default_rng(42)
    def floor(sec):
        return (rng.normal(0, 0.0015, int(16000 * sec))).astype(np.float32)

    # 前置 1s 底噪（VAD 校准段）+ 语音 + 静音交替
    mix = np.concatenate([floor(1.0), speech1, floor(1.5), speech2, floor(2.0), speech1[: 16000 * 3]])
    print(f"fixture: {len(mix)/16000:.1f}s from {len(wavs)} wavs ({os.path.dirname(wavs[0])})")

    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
