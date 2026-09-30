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

## 排行榜（Chrome headless，18.3s 语音+静音混音考卷，Actions x86_64）

> 同轮横向对比（单次 CI 内 5 实验同一 runner），跨轮绝对值有 ±15% 漂移。

| 实验 | RTF | 解码耗时 | 回调抖动P95 | 结论 |
|---|---|---|---|---|
| a0 对照（官方 ScriptProcessor） | 0.0576 | 1165ms | 260.1ms | 基线：17×实时 |
| a1 懒加载+SW缓存 | 0.0564 | 1140ms | 260.1ms | 二访 0 字节 / ready 969ms（vs 1285ms）✅ |
| a2 AudioWorklet 采集 | 0.0575 | 1183ms | **10.1ms** | 抖动 26× 改善，RTF 持平 ✅ |
| a3 能量VAD门控 | **0.0405** | **819ms** | 260.1ms | 有效RTF -30%（speech 69.6% 检出正确）✅ |
| a4 Worklet+VAD 合体（M1.5） | TBD | TBD | TBD | 抖动+CPU 双收益 + 320ms pre-roll 修首字clip |

参照系：原生 CPU 单线程 RTF 0.020（49×）→ 浏览器 wasm 折损 ~2.9×，仍有 17× 实时余量。
数据文件：`results/browser-m1.json`（自动考台产物）。

## 基线锚点

- `results/native-baseline.json`：官方 sherpa-onnx 原生 CPU 推理 RTF（GitHub Actions `ubuntu-24.04-arm` runner，1/2/4 线程）
- Pages 在线 demo（官方 wasm 构建，单线程 SIMD）：见仓库 Pages 链接

## 实验轨道

| 轨道 | 内容 | 状态 |
|---|---|---|
| 00-baseline | 官方 sherpa-onnx wasm 原样构建 + 原生基线锚点 | ✅ |
| A1 | 懒加载+Cache API（SW cache-first） | ✅ |
| A2 | AudioWorklet 16k 采集（替代 ScriptProcessor） | ✅ |
| A3 | 能量 VAD 门控（静音不解码） | ✅ |
| A4/M1.5 | Worklet+VAD 合体 + pre-roll 防首字clip | ✅ |
| B | 多线程 wasm 重编（-pthread + coi-serviceworker） | ⏸ 调研后建议跳过：见 `docs/m2-feasibility.md`（天花板 -10%、需源码编 ORT wasm threads + SAB 基建） |
| C | ORT 瘦身 + 编译激进化 | ⏳ |
| D | 自写引擎：small-ctc 的 raw WGSL GPU-resident 内核 | ⏳ |

实验页部署在 Pages 子路径：`/a0-reference/`（对照组）`/a1-lazy-cache/` `/a2-worklet-16k/` `/a3-vad-gate/` `/a4-worklet-vad/`；
自动考台 `bench/browser_bench.py`（Playwright + Chrome 假麦克风），结果落 `results/browser-m1.json`。

## License

Apache-2.0（模型权重遵循各自原始许可，均来自 k2-fsa 官方 Releases）
