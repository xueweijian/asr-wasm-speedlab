# 00-baseline：官方 sherpa-onnx wasm 原样构建

- 构建：sherpa-onnx `v1.12.7` + emsdk 4.0.23 + 官方 `build-wasm-simd-asr.sh`（单线程 SIMD）
- 模型：zipformer-14m（int8 三件套，encoder/decoder/joiner 重命名为官方约定）
- 产物：GitHub Pages 根目录即 demo（麦克风实时识别）
- 角色：所有后续实验（A/B/C/D 轨）的性能与正确性参照系
