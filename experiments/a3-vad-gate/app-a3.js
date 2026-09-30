// a3-vad-gate: 能量 VAD 门控 —— 静音段跳过 acceptWaveform+decode，省 CPU/电
// 对照 a0：采集路径一致；唯一变量 = 是否门控。有效 RTF = decodeMs / 全时长（含静音）
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusEl = document.getElementById('status');
const textArea = document.getElementById('results');
const metricsEl = document.getElementById('metrics');

const M = {
  app: 'a3-vad-gate', ready: false,
  load: { recognizerInitMs: 0 },
  stream: { audioSec: 0, feedMs: 0, decodeMs: 0, cbCount: 0, cbGapP95: 0, longTasks: 0, results: [] },
  vad: { mode: 'energy', activeSec: 0, idleSec: 0, speechPct: 0, noiseDb: 0, thresholdDb: 0 },
};
window.__speedlab = M;

function dump() { metricsEl.textContent = JSON.stringify(M, null, 1); }

// ---- 能量 VAD：前 0.5s 校准噪声底 → 门限 = 噪声底 + 12dB，350ms hangover ----
const CAL_MS = 500, THRESH_DB = 12, HANGOVER_MS = 350;
let calStart = 0, noiseDb = -60, inSpeech = false, hangoverUntil = 0;

function vadDecision(samples, now) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  const rms = Math.sqrt(sum / samples.length);
  const db = 20 * Math.log10(rms + 1e-10);
  if (now - calStart < CAL_MS) { // 校准期：指数平滑噪声底
    noiseDb = noiseDb === -60 ? db : noiseDb * 0.95 + db * 0.05;
    M.vad.noiseDb = +noiseDb.toFixed(1);
    return false;
  }
  M.vad.thresholdDb = +(noiseDb + THRESH_DB).toFixed(1);
  if (db > noiseDb + THRESH_DB) {
    inSpeech = true;
    hangoverUntil = now + HANGOVER_MS;
  } else if (now > hangoverUntil) {
    inSpeech = false;
  }
  return inSpeech;
}

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
  statusEl.textContent = 'ready · a3 能量VAD门控';
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
    if (!calStart) calStart = now;

    let samples = new Float32Array(e.inputBuffer.getChannelData(0));
    samples = downsampleBuffer(samples, expectedSampleRate);
    const durSec = samples.length / expectedSampleRate;
    M.stream.audioSec += durSec;
    M.stream.cbCount++;

    const active = vadDecision(samples, now);
    if (active) {
      M.vad.activeSec += durSec;
      if (!recognizer_stream) recognizer_stream = recognizer.createStream();
      let t0 = performance.now();
      recognizer_stream.acceptWaveform(expectedSampleRate, samples);
      M.stream.feedMs += performance.now() - t0;
      t0 = performance.now();
      while (recognizer.isReady(recognizer_stream)) recognizer.decode(recognizer_stream);
      M.stream.decodeMs += performance.now() - t0;
      const result = recognizer.getResult(recognizer_stream).text;
      if (result.length > 0 && result !== lastResult) lastResult = result;
    } else {
      M.vad.idleSec += durSec;
    }

    // endpoint 跟随（即使静音也查，及时固化分段）
    if (recognizer_stream) {
      const isEndpoint = recognizer.isEndpoint(recognizer_stream);
      if (isEndpoint) {
        if (lastResult.length) { resultList.push(lastResult); M.stream.results.push(lastResult); lastResult = ''; }
        recognizer.reset(recognizer_stream);
      }
    }
    M.vad.speechPct = +(M.vad.activeSec / Math.max(M.stream.audioSec, 1e-9) * 100).toFixed(1);
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
    // 有效 RTF：解码耗时 / 全部音频时长（含静音）——静音省下的都是赚的
    M.stream.rtf = +(M.stream.decodeMs / 1000 / Math.max(M.stream.audioSec, 1e-9)).toFixed(4);
    M.stream.activeRtf = +(M.stream.decodeMs / 1000 / Math.max(M.vad.activeSec, 1e-9)).toFixed(4);
    M.stream.text = (resultList.join('') + lastResult).slice(0, 120);
    dump();
    startBtn.disabled = false; stopBtn.disabled = true;
  };
}

navigator.mediaDevices.getUserMedia({ audio: true })
  .then(onSuccess, e => { statusEl.textContent = 'mic error: ' + e.message; });
