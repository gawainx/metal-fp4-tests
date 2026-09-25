# Metal 计算耗时对照

## 使用 Release

在 macOS 27 的 Apple Silicon Mac 上，从 [Releases](https://github.com/gawainx/metal-fp4-tests/releases/latest) 下载 `metal-precision-compare-macos27-arm64.tar.gz`，然后运行：

```sh
tar -xzf metal-precision-compare-macos27-arm64.tar.gz
cd metal-precision-compare-macos27-arm64
./run.sh
```

无需安装 Xcode。程序会在终端以表格逐一对比 BF16、FP8 与 FP4 的原生 Ops 和软件解码计算耗时，并显示各精度的软件耗时除以原生耗时的倍数。结果保存到 `results/precision_comparison_<芯片型号>.json`。

## 从源码运行

需要 macOS 27、Apple Silicon 和完整 Xcode。

第一步，执行 `xcodebuild -downloadComponent MetalToolchain` 下载 Metal toolchain。

第二步，在仓库根目录运行：

```sh
make compare
```

命令会构建并运行程序，结果显示在终端，并保存到 `results/precision_comparison_<芯片型号>.json`。

## Metal TensorOps tile 调优

运行 `make tune-metal` 搜索 BF16、FP8 E4M3、FP4 E2M1 的 tile、SIMD group 数量和 K 分块。`make validate-tuned-metal` 仅运行每个 shape 已选的配置。两个命令使用相同的可精确表示输入值和 FP32 输出；结果保存为 JSON。已有结果文件会保留，新一轮使用递增的 `_runN` 后缀。

| 场景 | M × K × N | 已选 FP4 kernel |
| --- | --- | --- |
| prefill_linear | 128 × 1024 × 1024 | `fp4_m32n128_s4` |
| qkv_projection | 32 × 1024 × 3072 | `fp4_m32n128_s4` |
| large_prefill | 512 × 1024 × 2048 | `fp4_m32n32_s1` |
| wide_projection | 32 × 1024 × 8192 | `fp4_m32n128_s4` |

kernel 定义在 `src/tune_tensorops.metal`，调度、计时和 CPU 抽样校验在 `src/tune_tensorops.mm`。`s4` 表示每个 threadgroup 使用 4 个 SIMD group。当前 M5 Max 上，已选 FP4 配置改善了 prefill_linear、qkv_projection、wide_projection 的 FP4 耗时；相同数据场景下，调优后的 BF16 仍更快。完整候选耗时保存在 `results/tensorops_tuning_extended_Apple_M5_Max.json`。

## CUDA BF16、FP8、FP4 对照

在 SM120 GPU 上，使用已安装的 CUDA Toolkit、cuBLAS 和 CUTLASS 运行：

```sh
make cuda-compare NVCC=/usr/local/cuda/bin/nvcc CUTLASS_DIR=/path/to/cutlass
```

程序对齐 Metal 用例的两个矩阵形状和随机输入值，分别运行 BF16、FP8 E4M3 与 FP4 E2M1 GEMM。输入值都可由三种格式精确表示，三条路径使用 FP32 累加与输出。BF16 使用 cuBLAS，FP8 和不带缩放平面的 FP4 使用 CUTLASS SM120 Tensor Core 内核。CUDA event 记录每次 GPU 耗时，先预热 3 次，再测量 30 次；每种输出都与 CPU 参考值逐元素比较。终端显示平均耗时与 `BF16 耗时 / 低精度耗时`，JSON 保存在 `results/precision_comparison_NVIDIA_RTX_PRO_4500_Blackwell.json`。

本次验证使用 CUTLASS `v4.5.0-15-g1fc71b3e`。构建时会在 `build/cutlass-overlay` 中复制并修正该版本的 SM120 FP4 装载移位头文件；CUTLASS 原始检出、CUDA Toolkit 和驱动均不会被修改。
