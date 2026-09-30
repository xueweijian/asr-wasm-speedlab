// c3-skip-sniff: a4 同逻辑（Worklet+VAD+preroll），唯一变量 = createOnlineRecognizer 传显式配置
// 预期（相对同轮 a0）：cbGapP95 掉到 ~10ms 量级（a2 的收益）且 rtf 掉 ~30%（a3 的收益）
// 新变量 pre-roll：语音起点前 320ms 缓冲在 onset 时一次性补喂，修 a3 已知的首字clip
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusEl = document.getElementById('status');
const textArea = document.getElementById('results');
const metricsEl = document.getElementById('metrics');

const M = {
  app: 'c3-skip-sniff', ready: false,
  load: { recognizerInitMs: 0 },
  stream: { audioSec: 0, feedMs: 0, decodeMs: 0, cbCount: 0, blockCount: 0, cbGapP95: 0, longTasks: 0, results: [] },
  vad: { mode: 'energy+preroll', activeSec: 0, idleSec: 0, speechPct: 0, noiseDb: 0, thresholdDb: 0, onsets: 0, prerollFedSec: 0 },
};
window.__speedlab = M;

function dump() { metricsEl.textContent = JSON.stringify(M, null, 1); }

// ---- 能量 VAD（同 a3 参数）：500ms 校准噪声底 → 门限=噪声底+12dB，350ms hangover ----
const CAL_MS = 500, THRESH_DB = 12, HANGOVER_MS = 350;
const PRE_ROLL_CHUNKS = 5; // 5×1024@16k = 320ms
let calStart = 0, noiseDb = -60, inSpeech = false, hangoverUntil = 0;

function vadDecision(samples, now) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  const rms = Math.sqrt(sum / samples.length);
  const db = 20 * Math.log10(rms + 1e-10);
  if (now - calStart < CAL_MS) { // 校准期：指数平滑噪声底
    noiseDb = noiseDb === -60 ? db : noiseDb * 0.95 + db * 0.05;
    noiseDb = Math.min(noiseDb, -35); // 防呆：校准期撞上语音别把门限抬上天
    M.vad.noiseDb = +noiseDb.toFixed(1);
    return false;
  }
  M.vad.thresholdDb = +(noiseDb + THRESH_DB).toFixed(1);
  if (db > noiseDb + THRESH_DB) {
    hangoverUntil = now + HANGOVER_MS;
    if (!inSpeech) { inSpeech = true; M.vad.onsets++; }
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
  // 显式 modelType：跳过 GetModelType 的嗅探 session（encoder 21.6MB 会被 ORT 完整解析两次）
  // debug:0：官方默认是 1，会往 stdout 打整个 config
  recognizer = createOnlineRecognizer(Module, {
    featConfig: { sampleRate: 16000, featureDim: 80 },
    modelConfig: {
      transducer: { encoder: './encoder.onnx', decoder: './decoder.onnx', joiner: './joiner.onnx' },
      paraformer: { encoder: '', decoder: '' },
      zipformer2Ctc: { model: '' },
      nemoCtc: { model: '' },
      toneCtc: { model: '' },
      tokens: './tokens.txt',
      numThreads: 1,
      provider: 'cpu',
      debug: 0,
      modelType: 'zipformer2',
      modelingUnit: 'cjkchar',
      bpeVocab: '',
    },
    decodingMethod: 'greedy_search',
    maxActivePaths: 4,
    enableEndpoint: 1,
    rule1MinTrailingSilence: 2.4,
    rule2MinTrailingSilence: 1.2,
    rule3MinUtteranceLength: 20,
    hotwordsFile: '',
    hotwordsScore: 1.5,
    ctcFstDecoderConfig: { graph: '', maxActive: 3000 },
    ruleFsts: '',
    ruleFars: '',
  });
  M.load.recognizerInitMs = +(performance.now() - t0).toFixed(1);
  M.load.readySinceNav = Math.round(performance.now()); // 冷启动分解：导航→runtime就绪总墙钟
  M.ready = true;
  startBtn.disabled = false;
  statusEl.textContent = 'ready · c3 跳过类型嗅探';
  dump();
};

try {
  new PerformanceObserver(list => { M.stream.longTasks += list.getEntries().length; })
    .observe({ entryTypes: ['longtask'] });
} catch (e) {}

let audioCtx, mediaStream, workletNode = null, sink = null;
let recognizer = null, recognizer_stream = null;
const expectedSampleRate = 16000;
const CHUNK = 1024; // 64ms：与 a2 一致，低延迟且调用开销可控
let acc = new Float32Array(CHUNK), accLen = 0;
let lastMsgTime = 0;
const msgGaps = [];
let preRoll = []; // 最近 5 个 chunk 的环形缓冲（idle 时也在攒）

function feedRecognizer(samples) {
  if (!recognizer_stream) recognizer_stream = recognizer.createStream();
  let t0 = performance.now();
  recognizer_stream.acceptWaveform(expectedSampleRate, samples);
  M.stream.feedMs += performance.now() - t0;
  t0 = performance.now();
  while (recognizer.isReady(recognizer_stream)) recognizer.decode(recognizer_stream);
  M.stream.decodeMs += performance.now() - t0;
  const result = recognizer.getResult(recognizer_stream).text;
  if (result.length > 0 && result !== lastResult) lastResult = result;
}

function processBlock(samples, audioTime) {
  if (!M.ready) return; // 模型未就绪：丢弃早期音频（autoplay 下渲染循环可能先于 runtime 启动）
  const now = performance.now();
  if (lastMsgTime) msgGaps.push(now - lastMsgTime);
  lastMsgTime = now;
  if (!calStart) calStart = now;
  M.stream.cbCount++;

  if (accLen + samples.length <= CHUNK) {
    acc.set(samples, accLen);
    accLen += samples.length;
  } else {
    acc = new Float32Array(samples); // 溢出保护（理论上不会发生）
    accLen = samples.length;
  }
  if (accLen < CHUNK) return;
  const block = acc; acc = new Float32Array(CHUNK); accLen = 0;
  M.stream.blockCount++;
  const durSec = block.length / expectedSampleRate;
  M.stream.audioSec += durSec;

  const wasSpeech = inSpeech;
  const active = vadDecision(block, now);
  if (active) {
    M.vad.activeSec += durSec;
    if (!wasSpeech) {
      // onset：先补喂 pre-roll（320ms 缓冲），把被门控吃掉的字头找回来
      for (const pre of preRoll) { feedRecognizer(pre); M.vad.prerollFedSec += pre.length / expectedSampleRate; }
    }
    preRoll = []; // 语音期不再需要缓冲
    feedRecognizer(block);
  } else {
    M.vad.idleSec += durSec;
    preRoll.push(block); // idle 也持续攒最近音频
    if (preRoll.length > PRE_ROLL_CHUNKS) preRoll.shift();
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
}

async function init(stream) {
  audioCtx = new AudioContext({ sampleRate: 16000 });
  M.capture = { contextRate: audioCtx.sampleRate, quantum: 128, chunk: CHUNK };
  if (audioCtx.sampleRate !== expectedSampleRate) {
    M.capture.warn = 'context 不是 16k，需重采样';
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
    // 有效 RTF = decodeMs / 全部音频时长（含静音）——静音省下的都是赚的
    M.stream.rtf = +(M.stream.decodeMs / 1000 / Math.max(M.stream.audioSec, 1e-9)).toFixed(4);
    M.stream.activeRtf = +(M.stream.decodeMs / 1000 / Math.max(M.vad.activeSec, 1e-9)).toFixed(4);
    M.stream.text = (resultList.join('') + lastResult).slice(0, 120);
    dump();
    startBtn.disabled = false; stopBtn.disabled = true;
  };
}

navigator.mediaDevices.getUserMedia({ audio: true })
  .then(init, e => { statusEl.textContent = 'mic error: ' + e.message; });
