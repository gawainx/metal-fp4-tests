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
