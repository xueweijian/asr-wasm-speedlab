// a2-worklet-16k: AudioWorklet 采集（渲染线程回调→主线程 1024 样本块解码）
// 对照 a0（ScriptProcessor 4096 块）：预期 callback 抖动↓、长任务丢帧风险↓、延迟↓
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusEl = document.getElementById('status');
const textArea = document.getElementById('results');
const metricsEl = document.getElementById('metrics');

const M = {
  app: 'a2-worklet-16k', ready: false,
  load: { recognizerInitMs: 0 },
  stream: { audioSec: 0, feedMs: 0, decodeMs: 0, cbCount: 0, blockCount: 0, cbGapP95: 0, longTasks: 0, results: [] },
};
window.__speedlab = M;

function dump() { metricsEl.textContent = JSON.stringify(M, null, 1); }

let lastResult = '', resultList = [];
Module = {};
Module.locateFile = p => '../' + p; // 实验页在 /<app>/ 子路径，资产在站点根
Module.setStatus = s => { if (s) statusEl.textContent = s; };
Module.print = () => {};
Module.printErr = () => {};
Module.onRuntimeInitialized = function () {
  const t0 = performance.now();
  recognizer = createOnlineRecognizer(Module);
  M.load.recognizerInitMs = +(performance.now() - t0).toFixed(1);
  M.ready = true;
  startBtn.disabled = false;
  statusEl.textContent = 'ready · a2 AudioWorklet 采集';
  dump();
};

try {
  new PerformanceObserver(list => { M.stream.longTasks += list.getEntries().length; })
    .observe({ entryTypes: ['longtask'] });
} catch (e) {}

let audioCtx, mediaStream, workletNode = null, sink = null;
let recognizer = null, recognizer_stream = null;
const expectedSampleRate = 16000;
let recordSampleRate;
const CHUNK = 1024; // 攒到 1024 样本（64ms）喂一次解码：低延迟且调用开销可控
let acc = new Float32Array(CHUNK), accLen = 0;
let lastMsgTime = 0;
const msgGaps = [];

function processBlock(samples, audioTime) {
  if (!M.ready) return; // 模型未就绪：丢弃早期音频（autoplay 下渲染循环可能先于 runtime 启动）
  const now = performance.now();
  if (lastMsgTime) msgGaps.push(now - lastMsgTime);
  lastMsgTime = now;
  M.stream.cbCount++;

  if (accLen + samples.length <= CHUNK) {
    acc.set(samples, accLen);
    accLen += samples.length;
  } else {
    // 溢出保护：理论上不会发生（128 整除 1024）
    acc = new Float32Array(samples);
    accLen = samples.length;
  }
  if (accLen < CHUNK) return;
  const block = acc; acc = new Float32Array(CHUNK); accLen = 0;
  M.stream.blockCount++;

  if (!recognizer_stream) recognizer_stream = recognizer.createStream();
  let t0 = performance.now();
  recognizer_stream.acceptWaveform(expectedSampleRate, block);
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
  M.stream.audioSec += block.length / expectedSampleRate;
  textArea.value = resultList.join('\n') + (lastResult ? '\n' + lastResult : '');
  dump();
}

async function init(stream) {
  audioCtx = new AudioContext({ sampleRate: 16000 });
  recordSampleRate = audioCtx.sampleRate;
  M.capture = { contextRate: recordSampleRate, quantum: 128, chunk: CHUNK };
  if (recordSampleRate !== expectedSampleRate) {
    M.capture.warn = 'context 不是 16k，需重采样（本实验预期浏览器直接给 16k）';
  }
  await audioCtx.audioWorklet.addModule('capture-worklet.js');
  mediaStream = audioCtx.createMediaStreamSource(stream);
  workletNode = new AudioWorkletNode(audioCtx, 'capture-processor', { numberOfOutputs: 1 });
  sink = audioCtx.createGain(); sink.gain.value = 0; // 哑输出，仅维持渲染循环
  workletNode.port.onmessage = e => processBlock(e.data.samples, e.data.currentTime);
  mediaStream.connect(workletNode);
  workletNode.connect(sink).connect(audioCtx.destination);

  startBtn.onclick = () => {
    audioCtx.resume();
    stopBtn.disabled = false; startBtn.disabled = true;
  };
  stopBtn.onclick = () => {
    msgGaps.sort((a, b) => a - b);
    M.stream.cbGapP95 = msgGaps.length ? +msgGaps[Math.floor(msgGaps.length * 0.95)].toFixed(1) : 0;
    M.stream.rtf = +(M.stream.decodeMs / 1000 / Math.max(M.stream.audioSec, 1e-9)).toFixed(4);
    M.stream.text = (resultList.join('') + lastResult).slice(0, 120);
    dump();
    startBtn.disabled = false; stopBtn.disabled = true;
  };
}

navigator.mediaDevices.getUserMedia({ audio: true })
  .then(init, e => { statusEl.textContent = 'mic error: ' + e.message; });
