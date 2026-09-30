// a0-reference: 官方采集路径（ScriptProcessor + 主线程解码）+ 性能埋点 = 对照组
// 路径与官方 app-asr.js 完全一致，仅加计时；所有实验与其对表。
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusEl = document.getElementById('status');
const textArea = document.getElementById('results');
const metricsEl = document.getElementById('metrics');

const M = {
  app: 'a0-reference', ready: false,
  load: { recognizerInitMs: 0 },
  stream: { audioSec: 0, feedMs: 0, decodeMs: 0, cbCount: 0, cbGapP95: 0, longTasks: 0, results: [] },
};
window.__speedlab = M;

function dump() { metricsEl.textContent = JSON.stringify(M, null, 1); }

let lastResult = '', resultList = [];
Module = {};
Module.locateFile = (p, dir = '') => dir + p; // dir = wasm 资产所在目录（site 根）
Module.setStatus = s => { if (s) statusEl.textContent = s; };
Module.print = () => {};
Module.printErr = () => {};
Module.onRuntimeInitialized = function () {
  const t0 = performance.now();
  recognizer = createOnlineRecognizer(Module);
  M.load.recognizerInitMs = +(performance.now() - t0).toFixed(1);
  M.ready = true;
  startBtn.disabled = false;
  statusEl.textContent = 'ready · a0 对照组（ScriptProcessor 官方路径）';
  dump();
};

// 主线程长任务计数（ScriptProcessor 丢帧风险的直接观测）
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
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i]; count++;
    }
    result[offsetResult] = accum / count;
    offsetResult++; offsetBuffer = nextOffsetBuffer;
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
    mediaStream.connect(recorder);
    recorder.connect(audioCtx.destination);
    stopBtn.disabled = false; startBtn.disabled = true;
  };
  stopBtn.onclick = () => {
    recorder.disconnect(audioCtx.destination);
    mediaStream.disconnect(recorder);
    cbGaps.sort((a, b) => a - b);
    M.stream.cbGapP95 = cbGaps.length ? +cbGaps[Math.floor(cbGaps.length * 0.95)].toFixed(1) : 0;
    M.stream.rtf = +(M.stream.decodeMs / 1000 / Math.max(M.stream.audioSec, 1e-9)).toFixed(4);
    M.stream.text = (resultList.join('') + lastResult).slice(0, 120);
    dump();
    startBtn.disabled = false; stopBtn.disabled = true;
  };
}

navigator.mediaDevices.getUserMedia({ audio: true })
  .then(onSuccess, e => { statusEl.textContent = 'mic error: ' + e.message; });
