# Apple Silicon FP4 benchmark

这个项目用一个 Metal 命令行基准检查当前 SDK 的原生 FP4 tensor API 声明，并测量打包 E2M1 FP4 权重、每 32 个权重一个 FP16 缩放因子、FP16 激活的矩阵乘法。它不会根据 M1、M2 或后续芯片型号选择实现，运行时只记录 Metal device、Apple GPU Family 7 支持状态和 SDK 探测结果。

运行 `make run`。程序会在控制台打印每个负载的 FP16 与 FP4 GPU 时间、速度比和 FP4 RMSE，同时把完整环境信息、TFLOPS、误差指标和速度比保存到带芯片型号的 `results/fp4_benchmark_Apple_M1_Max.json`。可用 `./scripts/build_and_run.sh --hidden-size 2048 --iterations 10 --output results.json` 调整规模、迭代数和输出路径。

负载包括 prefill 线性层、单 token decode 线性层和 QKV 投影。FP4 使用 E2M1 的有限数值编码，权重按输出通道分块量化。FP16 和 FP4 使用同一份标量 Metal 矩阵乘法结构，结果用于衡量打包解码开销与量化误差。

`scripts/detect_native_fp4_api.sh` 会对当前 Xcode SDK 编译 `MTLTensorDataTypeMetalFloat4E2M1` 探针。探针仅表示 SDK 是否声明此类型，不表示 GPU 支持或运行了原生 FP4。当前程序始终执行逐元素解码的标量 Metal 内核；JSON 中的 `execution_backend` 明确记录为 `software_scalar_decode`。因此这些结果可用于比较本项目软件内核的运行时间和量化误差，不能作为原生 FP4 tensor 性能或芯片 FP4 硬件性能的结论。探针环境出错时脚本会报错退出，不再把编译环境故障记成 API 不可用。

## 验证原生 FP4

在 macOS 27、包含 Metal 4.1 的 Xcode SDK 和支持相应 TensorOps 的 GPU 上，运行 `make native-probe`。这个独立探针用 `metal_fp4_e2m1_format` 输入执行 32×32 的 `matmul2d`，逐元素核对 GPU 输出。它不会改写 `results` 中的基准文件。`Apple10=yes` 说明设备声明 Apple GPU Family 10；最终以探针的编译、管线创建、GPU 执行和数值核对共同判断这条原生计算路径是否可用。系统升级本身不会把旧的 `metal3.2` 标量内核自动转换成 FP4 TensorOps。

使用 FP4 TensorOps 的最小步骤见 `src/native_fp4_probe.metal` 和 `src/native_fp4_probe.mm`：以 `-std=metal4.1` 编译，包含 `<metal_tensor>` 与 `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`，用 `tensor<device metal_fp4_e2m1_format, ...>` 描述已打包的 FP4 数据，然后传给 `mpp::tensor_ops::matmul2d`。探针中每个 FP4 元素为 E2M1 编码 `0x2`（值 1），两个元素打包为一个 `0x22` 字节；行步长为 256 个元素，以满足 FP4 tensor 的对齐要求。此最小探针没有缩放平面；现有基准的每 32 值一个 FP16 缩放因子格式不能直接当作 Metal 27 的 FP8 E8M0 缩放平面，若要测量同样负载的原生量化矩阵乘法，需要先实现相应的数据转换与独立的数值验证。

## 原生与软件解码耗时对照

运行 `make compare`，结果写入 `results/fp4_native_vs_software_<芯片型号>.json`。两份独立的计算代码分别是 `src/fp4_native.metal`（Metal 4.1 TensorOps 原生 FP4）和 `src/fp4_software_decode.metal`（逐元素软件解码 FP4）。对照程序为两条路径传入完全相同的打包 E2M1 权重、FP16 激活和矩阵形状，交替执行，每条路径预热 3 次并记录 30 次 `GPUStartTime` 到 `GPUEndTime` 的 GPU 计算耗时。结果还核对两条 GPU 输出及独立 CPU 参考值。

这组对照使用无缩放平面的 E2M1 数据，比较的是相同计算任务下的原生 FP4 消费与软件解码耗时；它不复用上文旧基准的 FP16 block32 缩放权重，因此不能把两组文件中的耗时直接配对计算速度比。
