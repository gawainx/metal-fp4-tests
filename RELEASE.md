# Metal 计算耗时对照

适用于 macOS 27 的 Apple Silicon Mac，无需安装 Xcode。下载 `metal-precision-compare-macos27-arm64.tar.gz` 后运行：

```sh
tar -xzf metal-precision-compare-macos27-arm64.tar.gz
cd metal-precision-compare-macos27-arm64
./run.sh
```

终端会以表格逐一显示 BF16、FP8 与 FP4 的原生 Ops、软件解码 GPU 计算耗时，以及各自的软件耗时除以原生耗时的倍数。完整结果保存到 `results/precision_comparison_<芯片型号>.json`。
