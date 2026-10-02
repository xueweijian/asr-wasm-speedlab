"""D1 parity farm: run d1/runtime/parity.html in headless Chromium (SwiftShader
WebGPU) on Modal CPU, against the Volume-cached kernels.json/weights.bin/ref.

Usage:
  modal run modal/parity_modal.py                # kernel-level parity (chunk 0)
  modal run modal/parity_modal.py --mode e2e     # 7-chunk streaming gate
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

app = modal.App("parity-farm")
CACHE = modal.Volume.from_name("kgen-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("chromium", "nodejs", "npm")
    .run_commands("npm i --prefix /root/pm puppeteer-core@23 --registry=https://registry.npmjs.org")
    .add_local_dir(str(ROOT / "d1" / "runtime"), "/root/site")
)

DRIVE_JS = r"""
import puppeteer from 'puppeteer-core';
(async () => {
  const mode = process.argv[2];
  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/chromium',
    headless: true,
    dumpio: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage',
           '--enable-unsafe-swiftshader', '--use-angle=swiftshader',
           '--use-webgpu-adapter=swiftshader',
           '--enable-features=Vulkan'],
  });
  const page = await browser.newPage();
  page.on('console', (m) => console.log('[page]', m.text()));
  try {
    await page.goto('chrome://gpu', {timeout: 20000});
    const txt = await page.evaluate(() => document.body.innerText);
    const lines = txt.split('\n').filter((l) => /WebGPU|SwiftShader|Graphics Feature|adapter/i.test(l));
    console.log('[gpu-diag]', lines.slice(0, 20).join(' | '));
  } catch (e) { console.log('[gpu-diag] failed', e.message); }
  page.on('pageerror', (e) => console.log('[pageerr]', e.message));
  await page.goto(`http://127.0.0.1:8080/parity.html?mode=${mode}`, {timeout: 60000});
  try {
    await page.waitForFunction('window.__done === true', {timeout: 1500000, polling: 2000});
  } catch (e) {
    console.log('TIMEOUT waiting for __done');
  }
  const res = await page.evaluate('window.__result');
  console.log('RESULT_JSON:' + JSON.stringify(res));
  await browser.close();
})().catch((e) => { console.error('DRIVE_FAIL', e); process.exit(1); });
"""


@app.function(image=image, volumes={"/cache": CACHE}, cpu=4, memory=8192,
              timeout=1800)
def parity(mode: str = "kernel") -> dict:
    import json
    import os
    import shutil
    import subprocess
    import time

    site = "/root/site"
    # volume artifacts -> serve dir
    for f in ("kernels.json", "weights.bin"):
        src = f"/cache/out/{f}"
        if not os.path.exists(src):
            raise RuntimeError(f"missing {src}; run kgen farm first")
        shutil.copy(src, f"{site}/{f}")
    os.makedirs(f"{site}/ref", exist_ok=True)
    for f in os.listdir("/cache/ref"):
        shutil.copy(f"/cache/ref/{f}", f"{site}/ref/{f}")
    Path(f"{site}/package.json").write_text('{"type":"module"}')
    Path(f"{site}/drive.js").write_text(DRIVE_JS)
    if not os.path.exists(f"{site}/node_modules"):
        os.symlink("/root/pm/node_modules", f"{site}/node_modules")

    # structural smoke before touching the GPU
    r0 = subprocess.run(["node", "/root/site/smoke.js"], cwd=site,
                        capture_output=True, text=True, timeout=300)
    print(r0.stdout[-3000:])
    if r0.returncode:
        return {"pass": False, "smoke_fail": True, "stderr": r0.stderr[-3000:]}

    httpd = subprocess.Popen(["python3", "-m", "http.server", "8080",
                              "--directory", site, "--bind", "127.0.0.1"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1)
        r = subprocess.run(["node", f"{site}/drive.js", mode],
                           capture_output=True, text=True, timeout=1700)
        out = r.stdout
        print(out[-8000:])
        if r.returncode:
            print("STDERR:", r.stderr[-3000:])
            return {"pass": False, "drive_fail": True, "tail": out[-3000:]}
        for line in out.splitlines():
            if line.startswith("RESULT_JSON:"):
                result = json.loads(line[len("RESULT_JSON:"):])
                CACHE.commit()
                return result
        return {"pass": False, "no_result": True, "tail": out[-3000:]}
    finally:
        httpd.terminate()


@app.local_entrypoint()
def main(mode: str = "kernel"):
    import json as _j
    res = parity.remote(mode=mode)
    print("FINAL:", _j.dumps(res))
