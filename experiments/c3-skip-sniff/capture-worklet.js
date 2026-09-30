// a2 采集工作线程：跑在音频渲染线程（独立于主线程），128 帧/回调零抖动
// 主线程再忙也不丢样本 —— 这是替代已废弃 ScriptProcessor 的现代路径
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._lastTime = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const samples = input[0];
      const msg = {
        samples: new Float32Array(samples), // 拷贝后转移所有权（零拷贝语义）
        currentTime: currentTime,
      };
      this.port.postMessage(msg, [msg.samples.buffer]);
    }
    return true;
  }
}
registerProcessor('capture-processor', CaptureProcessor);
