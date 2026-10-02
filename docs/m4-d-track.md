# M4 · D 轨预研：CTC-WGSL 自写引擎

> 状态：预研（c2 仍在 CI 自跑，不阻塞本轨）。开工前先跑 §5 的搜索补课。

## 1. 动机与选型

- **为什么自写**：ORT wasm 是通用运行时，单线程、解释执行图；fluidaudio-web 灯塔数据显示 raw WGSL 手写内核比 ort-web 快 ~3×。CPU 路线（A/C 轨）已到 0.038 RTF，再往下是 ORT 内部开销，页面侧无牌可打。
- **为什么选 small-ctc 而不是 14M transducer**：
  - 单模型文件（26.3MB），无 autoregressive decoder 循环；
  - 原生数据：ctc 模型 t4 线程持续扩展（吞吐天花板更高），14M transducer t2 仅 -10% 后倒退；
  - 纯 CTC 解码 = greedy argmax + 去重合并，百行 JS 可写，无需 GPU。

## 2. 模型解剖（results/ops-inventory.json 实测）

```
small-ctc/model.onnx: 26.3MB, 7996 nodes, 1077 initializers (24.7MB)
Constant 2667 | Mul 777 | Add 594 | Unsqueeze 561 | Shape 379 | Cast 350
Gather 337 | Slice 294 | Concat 288 | Sub 280 | MatMulInteger 218
DynamicQuantizeLinear 200 | Reshape 174 | Transpose 171 | Where 102
Exp 77 | Equal 66 | Log 64 | MatMul 60 | Conv 54 | Expand 42 | Div 42
Pow 26 | ConstantOfShape 24 | Neg 24 | Range 24 | Sigmoid 24
ReduceMean 13 | Tile 12 | GatherElements 12 | Softmax 12 | Tanh 12
ReduceSum 6 | Max 4 | Abs 4 | LessOrEqual 1 | LogSoftmax 1
```

**关键洞察**：
- 7996 节点里 ~2/3 是形状/常量元操作（Constant+Unsqueeze+Shape+Cast+Gather+Slice+Concat+Reshape+Transpose ≈ 5800），离线常量折叠后图会缩到 ~2000 真计算节点；
- 真计算核心 = **QDQ int8 路径**（MatMulInteger 218 + DynamicQuantizeLinear 200）+ Conv 54 + MatMul 60 + elementwise 海；
- 无 contrib/ml/自定义域，全是 ai.onnx:13 标准。

## 3. 架构：离线折叠 + 内核清单 + WGSL 运行时

```
[CI: Python 离线 pass]                      [浏览器运行时]
model.onnx ──> 常量折叠/形状推理/死代码消除 ──> kernels.json（拓扑序内核表）
                                              │
fixtures wav ──> ORT fp32 参考输出（真值锚）    ├─ wgsl-runtime.js：内核调度器
                                              ├─ kernels/*.wgsl：手写内核池
                                              └─ ctc-decode.js：greedy 解码（CPU）
```

- 折叠器产出两件套：`kernels.json`（算子序列+权重引用）+ 重排后的权重 blob（fp16 存储，减下载）
- **D1a ✅ kgen v0 上线（2026-10-02，commit a325f98，d1/out/）**：4713 节点 → 4082 逻辑内核，**216 个 MatMulInteger 100% 融合为 matmul_int8_dq**（DQL+MMI+Cast+Mul×2 单内核，dot4I8Packed 目标形态），零 unsupported 算子；权重 326 张量 23.3MB（int8 21.9MB / fp32 1.5MB），已带 manifest（offset/bytes）打包 weights.bin。
- **dispatch 预算诚实账（普查数据）**：裸内核表 compute=2169 + layout=1913，直接跑必重演 ORT dispatch 悲剧。朴素链融合（elementwise+control 单消费链）只压到 1330——**不够**。可达路径 = 模块级融合（whisper-webgpu 路线）：① shape 静态求值消灭 layout/shape 管道；② 残余 Transpose/Slice/Gather 融进生产者写布局；③ zipformer 模式匹配（LayerNorm/BiasAdd+act/attention 链）整段折叠——12 stack × 6-8 模块内核 + CTC 头 ≈ **72-96 dispatch** 才够到"几十"。kgen v1 按此序推进。
- **kgen v1.5 实测阶梯（2026-10-02，Modal 农场快环迭代，bb1d3b5）**：
  | 阶段 | compute | layout | 手段 |
  |---|---|---|---|
  | v0 裸表 | 2169 | 1913 | DQ+MMI 融合（216/216） |
  | v1 静态形状 | 1789 | 686 | overwrite_input_shapes + 定点 300 轮（D0 部署契约：batch1/chunk77/状态定形） |
  | v1.5 fuseq | **1148** | 686 | 贪心单消费链（elementwise+control+reduce → 372 条链，均长 2.7——残差流分支限制链长） |
  剩余构成：206 matmul_int8_dq + 54 conv + 60 matmul_fp + 13 softmax + 372 fuseq + ~400 孤立 elementwise。**到"几十"必须模块级融合**（12 stack 同构 → 手写一次参数化复用）；686 layout 待"融进生产者写布局"。
- 内核池初版范围（普查定案）：matmul_int8_dq(216) + conv(54) + matmul_fp(60) + softmax/logsoftmax(13) + reduce(19) + elementwise 模板 + control(Where/Equal 掩码，静态化后多数进常量)。

## 4. 分层验证门（继承 diar-gpu-engine 方法论）

| 门 | 内容 | 场地 |
|---|---|---|
| D0 | 折叠器 parity：折叠前后 ORT 输出逐元素一致（fp32，容差 = k 点 √k 随机游走标定）；CER 考卷不变 | CI |
| D1 | WGSL 正确性：SwiftShader/headless Chrome 跑 kernels.json，输出 vs ORT 真值锚过容差门；CER 考卷零回归 | CI |
| D2 | 性能认证：真机 Chrome（WebGPU 硬件）RTF vs c4 栈（0.038）——**速度门只在真机判**，CI 只判正确性 | 真机 |
| D3 | 全家桶：d2 引擎 × c4 页面栈（worklet+VAD+SW 缓存）= 最终交付页 | 真机 |

**坑位预告**：
- Android WebView 的 WebGPU 可用性不稳（Chrome 稳定，WebView 看版本）——真机性能测试走 Chrome + Pages 页，沿用 c3 真机验证姿势；
- CI 的 headless Chrome WebGPU = SwiftShader 软渲染，只出正确性不出速度（预期比 wasm 还慢，别误判）；
- QDQ 的 DynamicQuantizeLinear 是逐 token 动态量化（scale/zero-point 每次算），WGSL 里要跟 matmul_int8 融合成单内核，否则中间内存往返吃光收益；
- zipformer-ctc 的流式 cache（encoder 状态跨 chunk 传递）要确认内核表能表达 state 输入输出（D0 阶段验证）。

## 5. 开工前搜索清单（动手前先搜，工作铁律 #4）

**侦察结论（2026-10-01 已完成首轮）**：
- ✅ **反灯塔实锤（佐证 D 轨赌注）**：gpuweb #5292——4090 上 ort-web WebGPU EP 跑 EfficientNet 80→1100ms（比 WASM **慢 14×**）；HF 官方 webgpu-embedding-benchmark 同向（WASM 快 6×）。Google 维护者 Kangz 定性：算子回退软件 + onnxruntime 调度问题，**不是 WebGPU 慢**。→ 小模型上 ORT WebGPU EP 是死路，手写融合内核才是活路。
- ✅ **正灯塔**：Xenova whisper-webgpu（手写 WGSL 全模型内核，74M whisper-base 实时转写）= D 轨同构先例；WONNX（Rust ONNX→WGSL 编译器）证明 ONNX→WGSL 路径可行，其算子覆盖表可抄作业。
- ✅ **fluidaudio-web 原仓库已定位**（2026-10-01 二轮搜索）：`FluidInference/fluidaudio-web`——"In-browser inference for FluidAudio's core models (ASR/TTS/diarization) via WebGPU + WebAssembly"；3× 数据点可直接查源码核对（282× 那条来自 M5 Pro 实测）。
- ✅ **onnx 常量折叠轮子**：onnxsim 已在 D0 验证可用（7996→4713 节点，-41.1%）。
- ✅ **WebGPU int8 matmul 先例（D1 内核形态定案）**：Chrome 官方博客实锤 WGSL 特性 `packed_4x8_integer_dot_product`——内置 `dot4U8Packed` / `dot4I8Packed`，正是 u32 打包 4×i8 路线；特性探测 `navigator.gpu.wgslLanguageFeatures.has('packed_4x8_integer_dot_product')`，不支持则回退手动解包乘加。参考优化：nuss-and-bolts《Optimizing a WebGPU Matmul Kernel for 1TFLOP+》（分块/向量化的 WGSL 教科书）。
- ✅ **zipformer CTC 流式语义**：双重确认——① D0 已实证（多 chunk 回灌 parity 3/3 位级一致，state 张量契约完整）；② sherpa csrc 源码核对（online-zipformer2-ctc-model.cc）：`Forward(features, states)` 进出即全部语义，`GetInitStates()` 来自图 initializers，ChunkLength/ChunkShift 是图外常量——**kernels.json 无障碍可表达**（固定 chunk 形状 + 状态直通 + 回写）。
- ✅ **Android Chrome WebGPU 现状**：Chrome 121 起在 Android 12+ 高通/ARM GPU 设备默认启用（官方博客）；2026 年视作成熟可用。D2 真机测试走 Android Chrome + Pages 页无障碍；WebView 场景需按版本探测兜底（m4 交付页用 Chrome 即可）。

**新增关键预判**：ort-web WebGPU EP 的死因（逐算子 CPU 回退 + dispatch 开销）对 D 轨的启示——折叠后内核表必须**激进的算子融合**（elementwise 全并入邻居、DQ+MatMulInteger 一体化），把每 chunk dispatch 数压到几十以内，否则 WebGPU 的每 dispatch 开销会重演 ORT 悲剧。

## 6. 里程碑定义

- **D0 ✅✅ 真实 fbank 复验通过（2026-10-02，commit 602a330，results/d0-fold-stats.json）**：tarball 自带考卷 test_wavs/0.wav（5.61s 中文）→ numpy kaldi 风格 fbank（povey 窗/预加重 0.97/80 mel，mean −7.1/std 4.2，教科书分布）→ 7×77 帧分块回灌：**max_abs=0.0（位级）× 7 chunk、零 NaN、掩码零失配**，远超 ≤1e-4 门槛；折叠 7996→4713 节点（-41.1%）不变。随机 smoke 门同轮也位级一致。
- **状态契约勘误（D1 必读）**：small-ctc（2025-04-01）与旧 14M zipformer v1 契约不同——非空 T_prev 递增，而是**固定尺寸滑动窗口缓存**：cached_key_[256,1,128]、nonlin_attn_[1,1,256,144]、val1/2_[256,1,48]、conv1/2_[1,192,15]（初始全零张量，batch 维=1），states 完全自管理（`new_*` 输出直通回灌，无外部 len 簿记）。 sherpa csrc `Forward(features, states)` 进出即全部语义。**WGSL 运行时的状态布局 = 这组张量**。
- 输入契约（实测）：x=[?,77,80]（chunk 固定 77 帧）、cached_key/val1/val2 [heads,?,D]（heads=256..32）、cached_nonlin_attn [1,?,heads,192/144]、cached_conv1/2 [?,192/256,15/7]、无 cached_len 输入；输出含 new_embed_states/new_processed_lens（终态 only）。
- **M4-毕业标准**：真机 Chrome 上 d3 页 RTF ≤ 0.020（打平原生 fp32 单线程）或相对 c4 再 -40%，且 CER 考卷零回归、零 pageerror。
- 失败判据（提前认输线）：D1 内核池超 15 个 WGSL 文件还没过 parity / D2 真机首测 RTF > 0.03（WebGPU 吞吐不如预期）→ 归档结论，资源转 M5（worker 多线程 ORT 或模型换小）。
