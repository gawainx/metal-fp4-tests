# FP4 计算耗时对照

适用于 macOS 27、Apple Silicon。无需安装 Xcode。解压后在终端进入本目录，执行 `./run.sh`。程序运行两种矩阵乘法：Metal TensorOps 原生 FP4，以及逐元素软件解码 FP4；两者使用相同的打包 E2M1 权重、FP16 激活与矩阵形状。终端显示 GPU 计算耗时，完整 JSON 写入 `results/fp4_native_vs_software_<芯片型号>.json`。

计时取每条路径预热 3 次后的 30 次 GPU 命令执行时间，报告均值、中位数、最小值和最大值。程序将两条 GPU 输出与独立 CPU 参考值核对；核对失败时返回非零状态。权重没有缩放平面，因此结果不与项目早期使用 FP16 block32 缩放权重的基准直接配对。

运行命令：

```sh
tar -xzf metal-fp4-compare-macos27-arm64.tar.gz
cd metal-fp4-compare-macos27-arm64
./run.sh
```
