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

> 下表为 M3 终轮（commit d303f28）**同一 runner 同轮**九格对照，零 pageerror；跨轮绝对值有 ±15% 漂移，历史轮数字见 git history。

| 实验 | RTF | 解码耗时 | 回调抖动P95 | ready | recogInit | 结论 |
|---|---|---|---|---|---|---|
| a0 对照（官方 ScriptProcessor） | 0.0463 | 936ms | 260.1ms | 885ms | 699ms | 基线：22×实时 |
| a1 懒加载+SW缓存 | 0.0454 | 918ms | 260.1ms | — | — | 二访 0 字节下载 ✅（RTF 无回归） |
| a2 AudioWorklet 采集 | 0.0461 | 948ms | **10.1ms** | 826ms | 688ms | 抖动 **26×** 改善，RTF 持平 ✅ |
| a3 能量VAD门控 | 0.0365 | 739ms | 260.1ms | 850ms | 710ms | 有效RTF **-21%** ✅ |
| a4 Worklet+VAD 合体（M1.5） | 0.0366 | 784ms | **10.1ms** | 858ms | 715ms | **双收益同持**：CPU -21% + 抖动 26×；320ms pre-roll 保首字 ✅ |
| c1 激进体积旗标 | 0.0371 | 792ms | 10.1ms | 847ms | 701ms | wasm gzip 仅 -0.4%——**证伪**（ORT 预编译 .a 主导体积，LTO 够不着）|
| c3 跳过类型嗅探 | 0.0370 | 784ms | 10.1ms | 653ms | **495ms（-29%）** | RTF 零回归；真机同缓存 1010→507ms（**-50%**）✅ |
| c4 全家桶（交付页） | 0.0382 | 808ms | 10.1ms | 678ms | **489ms** | a4×c3×a1 三合一：非假设检验，**最终栈定型** ✅ |
| c2 最小算子 ORT | 0.0383 | 833ms（**-11%**） | 10.1ms | 807ms | 662ms | 源码编 ORT（--disable_contrib/ml_ops）+ Modal 农场链接：wasm **12.8→11.7MB（-9%）**、解码 -11%、RTF 与 c 族持平 ✅（十三发终成，破案记录见下） |

参照系：原生 CPU 单线程 RTF 0.020（49×）→ 浏览器 wasm 折损 ~2.9×，仍有 21× 实时余量。
数据文件：`results/browser-m1.json`（自动考台产物，每轮覆盖）。
备注：a4 的 CPU 收益略低于纯 a3，差价 = pre-roll 补喂 1.7s 音频 + 9 次 onset 门控开销——**首字不丢**的价钱，值。
c4 相对 a0 的累计：RTF -21%、抖动 26×、recogInit -30%、二访零下载。

## 冷启动分解（C 轨 profile，Actions x86_64）

| 阶段 | a4 | 说明 |
|---|---|---|
| 资产下载（localhost） | ~140ms | 真网络 21MB gzip 是大头 → a1 SW 缓存已解二访 |
| wasm 编译 + runtime init | <100ms | Liftoff 够快 |
| **recognizerInit（ORT session 创建）** | **~930ms** | 真敌人：encoder 21.6MB 被 GetModelType 嗅探 + 真 session **完整解析两次** |

c3（跳嗅探）真机（Pixel 级 Android，Chrome）同缓存对照：1010ms → **507ms**。
坑：zh-14M-2023-02-23 是 **zipformer v1**，modelType 填 'zipformer2' 会走错 ctor **挂死**（无报错）。
c2（最小算子 ORT）**十三发终成**：算子清点 `results/ops-inventory.json`（14m 35 算子/ctc 37 算子，全标准 ai.onnx）。破案链（Modal 编译农场三轮符号审计）：12 轮 undefined 的真凶不是缺源码，是**打包管线损坏**——emar 逐档提取→重归档悄悄吞掉成员内容（env.cc.o 进档后 20KB→134KB、GetOpSchema 4985→115），且 Actions 日志被 continue-on-error 洗白+job log 需 admin 权限，全程盲修。修复 = **llvm-ar MRI `ADDLIB` 合并**（729 成员逐字节保留）+ 五道符号守卫门（EnvTimeC2Ev/GetErrnoInfoEv/GetAllNodeUnits/GetOpSchema/EnvC2Ev，缺一即炸）。现 c2 构建链全在 Modal 农场（`modal/build_ort.py`，增量 2 分钟级），产物走 `site-assets/` 通道入库，CI 只做终审。**M3 六格全齐收官**。

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
| C | ORT 瘦身 + 编译激进化 | ✅ 六格全齐（2026-10-02）：c3 init -29~50%、c4 交付栈、c2 wasm -9%/解码 -11%；c1 证伪入档 |
| D | 自写引擎：small-ctc 的 raw WGSL GPU-resident 内核 | ⏳ |

实验页部署在 Pages 子路径：`/a0-reference/`（对照组）`/a1-lazy-cache/` `/a2-worklet-16k/` `/a3-vad-gate/` `/a4-worklet-vad/`；
自动考台 `bench/browser_bench.py`（Playwright + Chrome 假麦克风），结果落 `results/browser-m1.json`。

## License

Apache-2.0（模型权重遵循各自原始许可，均来自 k2-fsa 官方 Releases）
