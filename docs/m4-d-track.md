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
- 内核池初版：elementwise 全家（1 个模板内核泛化）+ matmul_int8（分块累加）+ dq（量化）+ conv1d（im2col 或直接滑窗）+ softmax/logsoftmax + reduce 家族

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

- [ ] fluidaudio-web 源码：3× 数据的测法、内核池结构、QDQ 处理方式
- [ ] onnx 常量折叠现成轮子：onnxruntime optimizer（Python offline）能否直接出折叠图（`onnxruntime.tools` / `onnxsim`）
- [ ] WebGPU matmul_int8 先例：int8 在 WGSL 的表达（u32 打包 4×i8 + 手动解包乘加）
- [ ] zipformer CTC 流式语义：sherpa csrc 里 small-ctc 的 chunk 边界与 state 传递（online-ctc 相关源码）
- [ ] Android Chrome WebGPU 现状（2026）：默认开启版本、adapter 丢弃率

## 6. 里程碑定义

- **M4-毕业标准**：真机 Chrome 上 d3 页 RTF ≤ 0.020（打平原生 fp32 单线程）或相对 c4 再 -40%，且 CER 考卷零回归、零 pageerror。
- 失败判据（提前认输线）：D1 内核池超 15 个 WGSL 文件还没过 parity / D2 真机首测 RTF > 0.03（WebGPU 吞吐不如预期）→ 归档结论，资源转 M5（worker 多线程 ORT 或模型换小）。
