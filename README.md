# FP4 计算耗时对照

## 使用 Release

在 macOS 27 的 Apple Silicon Mac 上，从 [Releases](https://github.com/gawainx/metal-fp4-tests/releases/latest) 下载 `metal-fp4-compare-macos27-arm64.tar.gz`，然后运行：

```sh
tar -xzf metal-fp4-compare-macos27-arm64.tar.gz
cd metal-fp4-compare-macos27-arm64
./run.sh
```

无需安装 Xcode。程序会在终端显示原生 FP4 与软件解码 FP4 的 GPU 计算耗时，并将结果保存到 `results/fp4_native_vs_software_<芯片型号>.json`。

## 从源码运行

需要 macOS 27、Apple Silicon 和完整 Xcode。在仓库根目录运行：

```sh
make compare
```

命令会构建并运行程序，结果显示在终端，并保存到 `results/fp4_native_vs_software_<芯片型号>.json`。
