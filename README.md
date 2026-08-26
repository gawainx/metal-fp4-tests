# Apple Silicon FP4 benchmark

这个项目用一个 Metal 命令行基准检查当前 SDK 的原生 FP4 tensor API 声明，并测量打包 E2M1 FP4 权重、每 32 个权重一个 FP16 缩放因子、FP16 激活的矩阵乘法。它不会根据 M1、M2 或后续芯片型号选择实现，运行时只记录 Metal device、Apple GPU Family 7 支持状态和 SDK 探测结果。

运行 `make run`。程序会在控制台打印每个负载的 FP16 与 FP4 GPU 时间、速度比和 FP4 RMSE，同时把完整环境信息、TFLOPS、误差指标和速度比保存到 `build/fp4_benchmark.json`。可用 `./scripts/build_and_run.sh --hidden-size 2048 --iterations 10 --output results.json` 调整规模、迭代数和输出路径。

负载包括 prefill 线性层、单 token decode 线性层和 QKV 投影。FP4 使用 E2M1 的有限数值编码，权重按输出通道分块量化。FP16 和 FP4 使用同一份标量 Metal 矩阵乘法结构，结果用于衡量打包解码开销与量化误差。

`scripts/detect_native_fp4_api.sh` 会对当前 Xcode SDK 编译 `MTLTensorDataTypeMetalFloat4E2M1` 探针。API 出现时，JSON 会记录该状态；当前软件回退内核仍保持独立，以确保旧 SDK 与所有 M 系列机器能够得到可复现结果。
