# mi50grad Makefile
# Builds ISA probes, microbenchmarks, and kernel libraries for gfx906
#
# Usage:
#   make probes        - Build ISA probe suite
#   make bench         - Build microbenchmarks
#   make kernels       - Build assembly kernels
#   make all           - Build everything
#   make clean         - Remove build artifacts
#
# Cross-compilation: This Makefile is intended to run inside the ROCm container
# on the dev server (root@192.168.1.198).

ROCM_PATH ?= /opt/rocm
HIPCC     ?= $(ROCM_PATH)/bin/hipcc
LLVM_MC   ?= $(ROCM_PATH)/llvm/bin/llvm-mc
LD_LLD    ?= $(ROCM_PATH)/llvm/bin/ld.lld
MCPU      ?= gfx906

HIPCC_FLAGS = -O3 --offload-arch=$(MCPU) -std=c++17
ASM_FLAGS   = --triple=amdgcn-amd-amdhsa --mcpu=$(MCPU) --filetype=obj

BUILD_DIR = build
PROBE_DIR = tests/isa_probes
BENCH_DIR = bench/micro
ASM_DIR   = src/asm

# ============================================================
# ISA Probes
# ============================================================

PROBE_SRCS = $(wildcard $(PROBE_DIR)/probe_*.s)
PROBE_HSACOS = $(patsubst $(PROBE_DIR)/%.s,$(BUILD_DIR)/probes/%.hsaco,$(PROBE_SRCS))

.PHONY: probes
probes: $(BUILD_DIR)/probes/run_probes $(PROBE_HSACOS)

$(BUILD_DIR)/probes/run_probes: $(PROBE_DIR)/run_probes.hip.cpp | $(BUILD_DIR)/probes
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $<

$(BUILD_DIR)/probes/%.hsaco: $(PROBE_DIR)/%.s | $(BUILD_DIR)/probes
	$(LLVM_MC) $(ASM_FLAGS) $< -o $(BUILD_DIR)/probes/$*.o
	$(LD_LLD) --shared $(BUILD_DIR)/probes/$*.o -o $@
	rm -f $(BUILD_DIR)/probes/$*.o

# ============================================================
# Microbenchmarks
# ============================================================

BENCH_SRCS = $(wildcard $(BENCH_DIR)/*.hip.cpp)
BENCH_BINS = $(patsubst $(BENCH_DIR)/%.hip.cpp,$(BUILD_DIR)/bench/%,$(BENCH_SRCS))

.PHONY: bench
bench: $(BENCH_BINS)

$(BUILD_DIR)/bench/%: $(BENCH_DIR)/%.hip.cpp | $(BUILD_DIR)/bench
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $<

# ============================================================
# Assembly Kernels (Phase 2+)
# ============================================================

KERNEL_SRCS = $(wildcard $(ASM_DIR)/*.s)
KERNEL_HSACOS = $(patsubst $(ASM_DIR)/%.s,$(BUILD_DIR)/kernels/%.hsaco,$(KERNEL_SRCS))

.PHONY: kernels
kernels: $(KERNEL_HSACOS)

$(BUILD_DIR)/kernels/%.hsaco: $(ASM_DIR)/%.s | $(BUILD_DIR)/kernels
	$(LLVM_MC) $(ASM_FLAGS) $< -o $(BUILD_DIR)/kernels/$*.o
	$(LD_LLD) --shared $(BUILD_DIR)/kernels/$*.o -o $@
	rm -f $(BUILD_DIR)/kernels/$*.o

# ============================================================
# Directories
# ============================================================

$(BUILD_DIR)/probes $(BUILD_DIR)/bench $(BUILD_DIR)/kernels:
	mkdir -p $@

# ============================================================
# HIP Kernel Libraries (shared libraries for ctypes loading)
# ============================================================

KERNEL_HIP_SRCS = $(wildcard src/kernels/kernel_*.hip)
KERNEL_SO = $(patsubst src/kernels/%.hip,$(BUILD_DIR)/kernels/%.so,$(KERNEL_HIP_SRCS))

.PHONY: hip_kernels
hip_kernels: $(KERNEL_SO)

$(BUILD_DIR)/kernels/%.so: src/kernels/%.hip | $(BUILD_DIR)/kernels
	$(HIPCC) $(HIPCC_FLAGS) -shared -fPIC -o $@ $<
	@echo "Built $@"

# ============================================================
# C Extensions (host-side C shared libraries)
# ============================================================

.PHONY: c_extensions
c_extensions: src/runtime/c_graph_dispatch.so src/runtime/c_dispatch.so

src/runtime/c_graph_dispatch.so: src/runtime/c_graph_dispatch.c
	gcc -O3 -shared -fPIC -I$(ROCM_PATH)/include \
	    -L$(ROCM_PATH)/lib -lamdhip64 \
	    -o $@ $<
	@echo "Built $@"

src/runtime/c_dispatch.so: src/runtime/c_dispatch.c
	gcc -O3 -shared -fPIC -I$(ROCM_PATH)/include \
	    -L$(ROCM_PATH)/lib -lamdhip64 \
	    -o $@ $<
	@echo "Built $@"

# ============================================================
# All / Clean
# ============================================================

.PHONY: all clean

all: probes bench kernels hip_kernels c_extensions

clean:
	rm -rf $(BUILD_DIR)
