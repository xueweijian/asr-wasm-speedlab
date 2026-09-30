# M2（B 轨 wasm 多线程）可行性调研 — 2026-10-01

## 结论：B 轨成本高、天花板低，建议降级/跳过

### 事实（源码核实，sherpa-onnx v1.13.8）
1. **官方 wasm 构建纯单线程**：`wasm/wasm-common.cmake` 无任何 pthread/PROXY_TO_PTHREAD/PTHREAD_POOL 配置；`build-wasm-simd-asr.sh` 只开 SIMD。
2. **ORT wasm 静态库是预编译下载的**：`cmake/onnxruntime-wasm-simd.cmake` 从 csukuangfj/onnxruntime-libs releases 拉 `onnxruntime-wasm-static_lib-simd-1.28.2.zip`（15MB）。
3. **csukuangfj 全部 release（v1.28.1~v1.30.0）只有 simd 单线程版**，无 threads 版 wasm 静态库。

### 若要做 B 轨，路径 = 
1. 从源码编 ORT wasm threads：`./build.sh --build_wasm --enable_wasm_simd --enable_wasm_threads`（官方支持，CI 约 40-90min）
2. 改 sherpa `onnxruntime-wasm-simd.cmake` 指向自编库 + link 加 `-pthread`
3. 页面加 coi-serviceworker（Pages 无 COOP/COEP，需 SW 注入头）启用 SharedArrayBuffer
4. 预计 2-3 轮 CI 迭代

### 为什么不值得（原生基线已给出答案）
- zipformer-14m 原生 t1→t2：RTF 0.020→0.018，仅 **-10%**（模型太小，线程开销吃掉收益）
- t4 反而倒退到 0.033（线程争抢）
- wasm 侧还要叠 SharedArrayBuffer/Atomics 开销，收益可能归零或为负
- 对比：M1 纯工程手段（VAD 门控）已经拿到 -30%~-43% CPU，零风险

### 替代建议（按性价比排序）
1. **D 轨 CTC-WGSL**（原计划王牌）：small-ctc 原生 t4 持续扩展（0.032→0.020），无自回归天然适合 GPU；fluidaudio-web 实测 raw WGSL 比 ort-web 快 3×
2. **C 轨 ORT 瘦身**：wasm 11MB + 25MB 模型包的下载/初始化优化（minimal ORT build 裁掉无用 op）
3. B 轨若做，只做 **2 线程**且只押 CTC 模型（唯一持续扩展的）
