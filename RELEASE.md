# FP4 计算耗时对照

适用于 macOS 27 的 Apple Silicon Mac，无需安装 Xcode。下载 `metal-fp4-compare-macos27-arm64.tar.gz` 后运行：

```sh
tar -xzf metal-fp4-compare-macos27-arm64.tar.gz
cd metal-fp4-compare-macos27-arm64
./run.sh
```

终端会以表格显示原生 FP4 与软件解码 FP4 的 GPU 计算耗时。完整结果保存到 `results/fp4_native_vs_software_<芯片型号>.json`。
