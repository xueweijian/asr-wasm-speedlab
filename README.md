# asr-wasm-speedlab

浏览器端中文 ASR 极速推理实验室 —— 纯本地推理（WebAssembly / WebGPU），零服务器。

**两个主角模型**（均为纯中文、25MB 级流式小模型）：

| 模型 | 架构 | 体积 | 特点 |
|---|---|---|---|
| `sherpa-onnx-streaming-zipformer-zh-14M` | Transducer | ~25MB | 三件套，为 Cortex-A7 级 CPU 设计 |
| `sherpa-onnx-streaming-zipformer-small-ctc-zh` | 纯 CTC | ~26MB | 单模型，无自回归解码 |

## 方法论

1. **基准先行**：无 JSON 数字不判优；记绝对毫秒
2. **正确性门**：每实验过 CER 考卷才进排行榜
3. **一实验 = 一分支 = 一个 Pages 活 demo**；成功 merge / 失败 revert

## 基线锚点

- `results/native-baseline.json`：官方 sherpa-onnx 原生 CPU 推理 RTF（GitHub Actions `ubuntu-24.04-arm` runner，1/2/4 线程）
- Pages 在线 demo（官方 wasm 构建，单线程 SIMD）：见仓库 Pages 链接

## 实验轨道

| 轨道 | 内容 | 状态 |
|---|---|---|
| 00-baseline | 官方 sherpa-onnx wasm 原样构建 + 原生基线锚点 | ✅ |
| A1 | 懒加载+Cache API（SW cache-first） | 🚧 |
| A2 | AudioWorklet 16k 采集（替代 ScriptProcessor） | 🚧 |
| A3 | 能量 VAD 门控（静音不解码） | 🚧 |
| B | 多线程 wasm 重编（-pthread + coi-serviceworker；14m 只试 2 线程） | ⏳ |
| C | ORT 瘦身 + 编译激进化 | ⏳ |
| D | 自写引擎：small-ctc 的 raw WGSL GPU-resident 内核 | ⏳ |

实验页部署在 Pages 子路径：`/a0-reference/`（对照组）`/a1-lazy-cache/` `/a2-worklet-16k/` `/a3-vad-gate/`；
自动考台 `bench/browser_bench.py`（Playwright + Chrome 假麦克风），结果落 `results/browser-m1.json`。

## License

Apache-2.0（模型权重遵循各自原始许可，均来自 k2-fsa 官方 Releases）
