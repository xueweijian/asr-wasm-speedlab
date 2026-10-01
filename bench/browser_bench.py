"""浏览器端自动考台 —— Playwright + Chrome 假麦克风
对 site/ 下的实验页面逐个：加载模型 → 开始识别 → 喂完整考卷 → 停止 → 收 __speedlab 指标。
a1 特殊：同 context 两访对比 SW 缓存（二访 transferSize 应≈0）。
输出: results/browser-m1.json
"""
import argparse
import json
import os
import shutil
import subprocess
import time
import wave

from playwright.sync_api import sync_playwright

APPS = ["a0-reference", "a1-lazy-cache", "a2-worklet-16k", "a3-vad-gate", "a4-worklet-vad", "c1-minsize", "c2-minort", "c3-skip-sniff", "c4-fullstack"]


def wav_seconds(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def wait_ready(page, timeout=180000):
    page.wait_for_function("window.__speedlab && window.__speedlab.ready === true",
                           timeout=timeout)


def run_stream(page, dur):
    page.click("#startBtn")
    time.sleep(dur + 2.0)
    if page.is_enabled("#stopBtn"):
        page.click("#stopBtn")
    time.sleep(0.4)
    return page.evaluate("window.__speedlab")


def collect_resources(page):
    """Resource Timing：各资产下载明细（transferSize=0 即 SW/磁盘缓存命中）"""
    return page.evaluate(
        """() => performance.getEntriesByType('resource')
             .filter(r => /sherpa-onnx|capture-worklet/.test(r.name))
             .map(r => ({name: r.name.split('/').pop().slice(0,44),
                         start: Math.round(r.startTime), dur: Math.round(r.duration),
                         tx: r.transferSize, dec: r.decodedBodySize}))"""
    )


def bench_app(browser, base, app, fixture_dur, logs):
    ctx = browser.new_context()
    page = ctx.new_page()
    page.on("pageerror", lambda e: logs.append(f"[{app}] pageerror: {e}"))
    out = {}
    try:
        page.goto(f"{base}/{app}/index.html", wait_until="load", timeout=60000)
        if app == "a1-lazy-cache":
            # 访1：点按钮 → SW 注册+网络拉模型
            page.click("#loadBtn")
            wait_ready(page)
            time.sleep(0.6)
            out["visit1"] = page.evaluate("window.__speedlab.load")
            # 访2：同 context 新页面 → 应命中 SW 缓存
            p2 = ctx.new_page()
            p2.goto(f"{base}/{app}/index.html", wait_until="load", timeout=60000)
            p2.click("#loadBtn")
            wait_ready(p2)
            time.sleep(0.6)
            out["visit2"] = p2.evaluate("window.__speedlab.load")
            out["stream"] = run_stream(p2, fixture_dur)
            p2.close()
        else:
            wait_ready(page)
            out["load"] = page.evaluate("window.__speedlab.load")
            out["resources"] = collect_resources(page)
            out["stream"] = run_stream(page, fixture_dur)
    except Exception as e:
        out["error"] = str(e).split("\n")[0][:300]
        logs.append(f"[{app}] {out['error']}")
    finally:
        ctx.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site")
    ap.add_argument("--fixture", default="/tmp/fixture.wav")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--out", default="results/browser-m1.json")
    args = ap.parse_args()

    dur = wav_seconds(args.fixture)
    print(f"fixture {dur:.1f}s @ http://127.0.0.1:{args.port}")

    server = subprocess.Popen(
        ["python3", "-m", "http.server", str(args.port), "--bind", "127.0.0.1", "--directory", args.site],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    base = f"http://127.0.0.1:{args.port}"
    logs, results = [], {}

    chrome_args = [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={os.path.abspath(args.fixture)}",
        "--autoplay-policy=no-user-gesture-required",
    ]
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=chrome_args)
            ua = browser.browser_type.name
            for app in APPS:
                if not os.path.isdir(os.path.join(args.site, app)):
                    continue
                # 变体实验（本地资产）：glue 未产出时跳过，避免空转 180s 超时污染记录
                if app in ("c1-minsize", "c2-minort") and not os.path.isfile(
                        os.path.join(args.site, app, "sherpa-onnx-wasm-main-asr.js")):
                    print(f"{app}: skipped (no local wasm assets)")
                    continue
                t0 = time.time()
                results[app] = bench_app(browser, base, app, dur, logs)
                print(f"{app}: {time.time()-t0:.0f}s -> "
                      + ("ERR " + results[app].get("error", "")[:60] if "error" in results[app]
                         else "ok"))
            browser.close()
    finally:
        server.terminate()

    out = {
        "meta": {
            "engine": ua, "headless": True, "fixture_s": round(dur, 1),
            "sha": os.environ.get("GITHUB_SHA", "")[:7],
            "runner": os.environ.get("RUNNER_NAME", "local"),
            "note": "Chrome假麦克风喂16k考卷; a0=对照组(官方ScriptProcessor路径)",
        },
        "apps": results,
        "logs": logs[:20],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved ->", args.out)
    if logs:
        print("\n".join(logs[:10]))
    # 至少一个 app 出分，否则视为考台故障（防静默空跑）
    ok = [a for a, r in results.items() if "error" not in r]
    if not ok:
        raise SystemExit("FAIL: no app benched - site layout or fixture wrong?")


if __name__ == "__main__":
    main()
