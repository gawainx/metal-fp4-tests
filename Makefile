BUILD_DIR := build
METAL_AIR := $(BUILD_DIR)/fp4_kernels.air
METAL_LIB := $(BUILD_DIR)/fp4_kernels.metallib
BENCHMARK := $(BUILD_DIR)/fp4_transformer_bench
NATIVE_PROBE_AIR := $(BUILD_DIR)/native_fp4_probe.air
NATIVE_PROBE_LIB := $(BUILD_DIR)/native_fp4_probe.metallib
NATIVE_PROBE := $(BUILD_DIR)/native_fp4_probe
COMPARE := $(BUILD_DIR)/compare_fp4
NATIVE_COMPARE_AIR := $(BUILD_DIR)/fp4_native.air
NATIVE_COMPARE_LIB := $(BUILD_DIR)/fp4_native.metallib
SOFTWARE_COMPARE_AIR := $(BUILD_DIR)/fp4_software_decode.air
SOFTWARE_COMPARE_LIB := $(BUILD_DIR)/fp4_software_decode.metallib

.PHONY: all run native-probe compare clean

all: $(BENCHMARK) $(METAL_LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(METAL_AIR): src/fp4_kernels.metal | $(BUILD_DIR)
	xcrun -sdk macosx metal -std=metal3.2 -c $< -o $@

$(METAL_LIB): $(METAL_AIR)
	xcrun -sdk macosx metallib $< -o $@

$(BENCHMARK): src/main.mm | $(BUILD_DIR)
	xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -framework Foundation -framework Metal $< -o $@

$(NATIVE_PROBE_AIR): src/native_fp4_probe.metal | $(BUILD_DIR)
	xcrun -sdk macosx metal -std=metal4.1 -c $< -o $@

$(NATIVE_PROBE_LIB): $(NATIVE_PROBE_AIR)
	xcrun -sdk macosx metallib $< -o $@

$(NATIVE_PROBE): src/native_fp4_probe.mm | $(BUILD_DIR)
	xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -framework Foundation -framework Metal $< -o $@

native-probe: $(NATIVE_PROBE) $(NATIVE_PROBE_LIB)
	./$(NATIVE_PROBE)

$(NATIVE_COMPARE_AIR): src/fp4_native.metal | $(BUILD_DIR)
	xcrun -sdk macosx metal -std=metal4.1 -c $< -o $@

$(NATIVE_COMPARE_LIB): $(NATIVE_COMPARE_AIR)
	xcrun -sdk macosx metallib $< -o $@

$(SOFTWARE_COMPARE_AIR): src/fp4_software_decode.metal | $(BUILD_DIR)
	xcrun -sdk macosx metal -std=metal4.1 -c $< -o $@

$(SOFTWARE_COMPARE_LIB): $(SOFTWARE_COMPARE_AIR)
	xcrun -sdk macosx metallib $< -o $@

$(COMPARE): src/compare_fp4.mm | $(BUILD_DIR)
	xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -framework Foundation -framework Metal $< -o $@

compare: $(COMPARE) $(NATIVE_COMPARE_LIB) $(SOFTWARE_COMPARE_LIB)
	./$(COMPARE)

run: all
	./scripts/build_and_run.sh

clean:
	rm -rf $(BUILD_DIR)
