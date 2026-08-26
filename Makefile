BUILD_DIR := build
METAL_AIR := $(BUILD_DIR)/fp4_kernels.air
METAL_LIB := $(BUILD_DIR)/fp4_kernels.metallib
BENCHMARK := $(BUILD_DIR)/fp4_transformer_bench

.PHONY: all run clean

all: $(BENCHMARK) $(METAL_LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(METAL_AIR): src/fp4_kernels.metal | $(BUILD_DIR)
	xcrun -sdk macosx metal -std=metal3.2 -c $< -o $@

$(METAL_LIB): $(METAL_AIR)
	xcrun -sdk macosx metallib $< -o $@

$(BENCHMARK): src/main.mm | $(BUILD_DIR)
	xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -framework Foundation -framework Metal $< -o $@

run: all
	./scripts/build_and_run.sh

clean:
	rm -rf $(BUILD_DIR)
