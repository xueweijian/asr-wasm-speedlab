// a1-lazy-cache: 模型懒加载（点按钮才拉 36MB）+ Service Worker Cache API 缓存
// 对照 a0：首访预热缓存（install 预取 + 页面动态加载共用），二访 transferSize≈0
const loadBtn = document.getElementById('loadBtn');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusEl = document.getElementById('status');
const textArea = document.getElementById('results');
const metricsEl = document.getElementById('metrics');

const M = {
  app: 'a1-lazy-cache', ready: false,
  load: { swRegisterMs: 0, visit1ToReadyMs: null, visit2ToReadyMs: null, recognizerInitMs: 0, visit1Bytes: null, visit2Bytes: null },
  stream: { audioSec: 0, feedMs: 0, decodeMs: 0, cbCount: 0, cbGapP95: 0, longTasks: 0, results: [] },
};
window.__speedlab = M;

function dump() { metricsEl.textContent = JSON.stringify(M, null, 1); }

function assetBytes() {
  // 统计 4 个大资产的 transferSize（SW 命中时 Chrome 报 0）
  return performance.getEntriesByType('resource')
    .filter(r => r.name.includes('sherpa-onnx'))
    .reduce((s, r) => s + (r.transferSize || 0), 0);
}

// ---- 懒加载：点击后 SW 注册 → 动态注入 emscripten 模块脚本 ----
loadBtn.onclick = async () => {
  loadBtn.disabled = true;
  M._clickT0 = performance.now();
  try {
    await navigator.serviceWorker.register('/asr-wasm-speedlab/sw.js');
    await navigator.serviceWorker.ready;
    // 等 claim() 接管当前页（首次注册也立即生效）
    for (let i = 0; i < 60 && !navigator.serviceWorker.controller; i++) {
      await new Promise(r => setTimeout(r, 50));
    }
    M.load.swRegisterMs = +(performance.now() - t0).toFixed(0);
    statusEl.textContent = 'SW ready，加载模型（首访走网络，二访走缓存）…';
  } catch (err) {
    statusEl.textContent = 'SW 注册失败(需 https/localhost): ' + err.message;
  }
  const s = document.createElement('script');
  s.src = '../sherpa-onnx-wasm-main-asr.js';
  s.onerror = () => { statusEl.textContent = '模块加载失败'; };
  document.body.appendChild(s);
};

let lastResult = '', resultList = [];
Module = {};
Module.locateFile = (p, dir = '') => dir + p;
Module.setStatus = s => { if (s && !M.ready) statusEl.textContent = s; };
Module.print = () => {};
Module.printErr = () => {};
Module.onRuntimeInitialized = function () {
  const t0 = performance.now();
  recognizer = createOnlineRecognizer(Module);
  M.load.recognizerInitMs = +(performance.now() - t0).toFixed(1);
  setTimeout(() => { // 等 resource entry 落账
    const bytes = assetBytes();
    if (M.load.visit1Bytes === null) M.load.visit1Bytes = bytes;
    else if (M.load.visit2Bytes === null) M.load.visit2Bytes = bytes;
    if (M.load.visit1ToReadyMs === null) M.load.visit1ToReadyMs = +(performance.now() - M._clickT0).toFixed(0);
    else if (M.load.visit2ToReadyMs === null) M.load.visit2ToReadyMs = +(performance.now() - M._clickT0).toFixed(0);
    dump();
  }, 300);
  M.ready = true;
  startBtn.disabled = false;
  statusEl.textContent = 'ready · a1 懒加载+SW缓存';
  dump();
};

try {
  new PerformanceObserver(list => { M.stream.longTasks += list.getEntries().length; })
    .observe({ entryTypes: ['longtask'] });
} catch (e) {}

let audioCtx, mediaStream, recorder = null;
let recognizer = null, recognizer_stream = null;
const expectedSampleRate = 16000;
let recordSampleRate;
let lastCbTime = 0;
const cbGaps = [];

function downsampleBuffer(buffer, exportSampleRate) {
  if (exportSampleRate === recordSampleRate) return buffer;
  const sampleRateRatio = recordSampleRate / exportSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0, offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) { accum += buffer[i]; count++; }
    result[offsetResult++] = accum / count;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function onSuccess(stream) {
  audioCtx = new AudioContext({ sampleRate: 16000 });
  recordSampleRate = audioCtx.sampleRate;
  M.capture = { contextRate: recordSampleRate };
  mediaStream = audioCtx.createMediaStreamSource(stream);
  recorder = audioCtx.createScriptProcessor(4096, 1, 2);

  recorder.onaudioprocess = function (e) {
    const now = performance.now();
    if (lastCbTime) cbGaps.push(now - lastCbTime);
    lastCbTime = now;
    let samples = new Float32Array(e.inputBuffer.getChannelData(0));
    samples = downsampleBuffer(samples, expectedSampleRate);
    if (!recognizer_stream) recognizer_stream = recognizer.createStream();
    let t0 = performance.now();
    recognizer_stream.acceptWaveform(expectedSampleRate, samples);
    M.stream.feedMs += performance.now() - t0;
    t0 = performance.now();
    while (recognizer.isReady(recognizer_stream)) recognizer.decode(recognizer_stream);
    M.stream.decodeMs += performance.now() - t0;
    const isEndpoint = recognizer.isEndpoint(recognizer_stream);
    const result = recognizer.getResult(recognizer_stream).text;
    if (result.length > 0 && result !== lastResult) lastResult = result;
    if (isEndpoint) {
      if (lastResult.length) { resultList.push(lastResult); M.stream.results.push(lastResult); lastResult = ''; }
      recognizer.reset(recognizer_stream);
    }
    M.stream.audioSec += samples.length / expectedSampleRate;
    M.stream.cbCount++;
    textArea.value = resultList.join('\n') + (lastResult ? '\n' + lastResult : '');
    dump();
  };

  startBtn.onclick = () => {
    mediaStream.connect(recorder); recorder.connect(audioCtx.destination);
    stopBtn.disabled = false; startBtn.disabled = true;
  };
  stopBtn.onclick = () => {
    recorder.disconnect(audioCtx.destination); mediaStream.disconnect(recorder);
    cbGaps.sort((a, b) => a - b);
    M.stream.cbGapP95 = cbGaps.length ? +cbGaps[Math.floor(cbGaps.length * 0.95)].toFixed(1) : 0;
    M.stream.rtf = +(M.stream.decodeMs / 1000 / Math.max(M.stream.audioSec, 1e-9)).toFixed(4);
    M.stream.text = (resultList.join('') + lastResult).slice(0, 120);
    dump();
    startBtn.disabled = false; stopBtn.disabled = true;
  };
}

// 注意：麦克风授权也在点击加载时才请求（真懒加载）
loadBtn.addEventListener('click', () => {
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(onSuccess, e => { statusEl.textContent = 'mic error: ' + e.message; });
}, { once: true });
