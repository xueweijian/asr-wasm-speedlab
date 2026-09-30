# A 轨实验目录

每个子目录 = 一个独立实验页，部署到 Pages 子路径 `/a0-reference/` 等。
全部与 **a0-reference（官方 ScriptProcessor 路径 + 埋点）** 对表，唯一变量原则。

| 实验 | 假设 | 关键指标 | 页面 |
|---|---|---|---|
| a0 对照组 | ——（基线） | RTF / cbGapP95 / longTasks | `/a0-reference/` |
| a1 懒加载+缓存 | 页面秒开；二访 0 字节 | visit2ToReadyMs、visit2Bytes | `/a1-lazy-cache/` |
| a2 AudioWorklet | 渲染线程采集消灭丢帧抖动 | cbGapP95↓、longTasks、RTF | `/a2-worklet-16k/` |
| a3 VAD 门控 | 静音段不解码省 CPU | 有效RTF(全时长)、activeRtf | `/a3-vad-gate/` |

统一指标接口：`window.__speedlab`（load/stream/vad 三段）。
自动考台：`bench/browser_bench.py`（Playwright + Chrome 假麦克风，CI 内跑）。

## 方法论调整说明

M1 首轮：实验页集中在 main 一次部署测完（快速拿数字）；后续逐实验调优时
再走 `exp/<name>` 分支（site.yml 已支持，push 即部署+跑分）。
