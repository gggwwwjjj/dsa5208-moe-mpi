# DSA5208 Project 3 - build, test and experiment driver.
#
#   make                      serial + parallel executables (PLATFORM=local)
#   make test                 correctness tests in double precision
#   make PLATFORM=atlas       on Atlas after `source /app1/ebenv; module load ...` (README)
#   make PLATFORM=vanda BACKEND=cuda   CUDA backend (own kernels, no cuBLAS), inside a Vanda job
#                                      after `module load gompi/2023a CUDA/12.1.1`
#   make data                 download Fashion-MNIST and convert it to flat binaries
#
# Executables go to build/<platform>-<real>/ so float timing builds and double
# test builds never mix.

PLATFORM ?= local
REAL     ?= float
BACKEND  ?= cpu
BUILD    := build/$(PLATFORM)-$(REAL)
ifeq ($(BACKEND),cuda)
  BUILD  := build/$(PLATFORM)-cuda-$(REAL)
endif

ifeq ($(PLATFORM),local)
  LIBOMP     ?= /opt/homebrew/opt/libomp
  CXX        := clang++
  MPICXX     := mpicxx
  OMP_CFLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
  OMP_LIBS   := -L$(LIBOMP)/lib -lomp
  PYTHON     ?= python3
  MPIRUN     := mpirun --oversubscribe
else
  # atlas / vanda: compiler and MPI come from `module load` (names in README).
  # No -march=native: login and compute nodes may have different CPUs.
  CXX        := g++
  MPICXX     := mpicxx
  OMP_CFLAGS := -fopenmp
  OMP_LIBS   := -fopenmp
  PYTHON     ?= python3
  MPIRUN     := mpirun
endif

CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Wno-unknown-pragmas -DREAL_T=$(REAL)

CORE_SRC     := src/core/data.cpp src/core/layers.cpp src/core/moe.cpp src/core/model.cpp src/core/train.cpp
BACKEND_SRC  := src/backend/cpu_omp.cpp
BACKEND_LIBS :=
ifeq ($(BACKEND),cuda)
  # A40 = compute capability 8.6; CUDA_HOME is set by the CUDA module
  NVCC         ?= nvcc
  CUDA_ARCH    ?= sm_86
  CUDA_HOME    ?= $(EBROOTCUDA)
  NVCCFLAGS    := -std=c++17 -O3 -arch=$(CUDA_ARCH) -DREAL_T=$(REAL) -Xcompiler -Wall
  BACKEND_SRC  := src/backend/cuda/cuda_backend.cu
  BACKEND_LIBS := -L$(CUDA_HOME)/lib64 -lcudart
endif
LIB_SRC      := $(CORE_SRC) $(BACKEND_SRC)
PARALLEL_SRC := src/parallel/allreduce.cpp src/parallel/alltoall.cpp

serial_obj = $(addprefix $(BUILD)/serial/,$(addsuffix .o,$(basename $(1))))
omp_obj    = $(addprefix $(BUILD)/omp/,$(addsuffix .o,$(basename $(1))))

SERIAL_BIN := $(BUILD)/mlp_serial
PARALLEL_BIN := $(BUILD)/mlp_parallel
PINGPONG_BIN := $(BUILD)/pingpong
GEMM_BENCH_BIN := $(BUILD)/gemm_bench
ALLREDUCE_BENCH_BIN := $(BUILD)/allreduce_bench
TEST_BINS  := $(BUILD)/test_gemm_serial $(BUILD)/test_gemm_omp $(BUILD)/test_gradcheck \
              $(BUILD)/test_allreduce $(BUILD)/test_alltoall $(BUILD)/compare_params

.PHONY: all serial parallel bench tests test data clean
all: serial parallel bench
serial: $(SERIAL_BIN)
parallel: $(PARALLEL_BIN)
bench: $(PINGPONG_BIN) $(GEMM_BENCH_BIN) $(ALLREDUCE_BENCH_BIN)
tests: $(TEST_BINS)

# --- compile rules: same source, two flavours ------------------------------
$(BUILD)/serial/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/omp/%.o: %.cpp
	@mkdir -p $(@D)
	$(MPICXX) $(CXXFLAGS) $(OMP_CFLAGS) -MMD -MP -c $< -o $@

# CUDA kernels (no MPI, no OpenMP inside): the same compile for both flavours.
# Two separate rules: a pattern rule with two targets means "one command makes
# both", so make would build only one of the objects.
$(BUILD)/serial/%.o: %.cu src/backend/backend.hpp src/core/config.hpp
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD)/omp/%.o: %.cu src/backend/backend.hpp src/core/config.hpp
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# --- executables -------------------------------------------------------------
$(SERIAL_BIN): $(call serial_obj,$(LIB_SRC) src/serial/main_serial.cpp)
	$(CXX) $^ $(BACKEND_LIBS) -o $@

$(PARALLEL_BIN): $(call omp_obj,$(LIB_SRC) $(PARALLEL_SRC) src/mpi/main_parallel.cpp)
	$(MPICXX) $^ $(OMP_LIBS) $(BACKEND_LIBS) -o $@

$(PINGPONG_BIN): $(call omp_obj,bench/pingpong.cpp)
	$(MPICXX) $^ $(BACKEND_LIBS) -o $@

$(ALLREDUCE_BENCH_BIN): $(call omp_obj,$(PARALLEL_SRC) $(BACKEND_SRC) bench/allreduce_bench.cpp)
	$(MPICXX) $^ $(OMP_LIBS) $(BACKEND_LIBS) -o $@

$(GEMM_BENCH_BIN): $(call omp_obj,$(BACKEND_SRC) bench/gemm_bench.cpp)
	$(MPICXX) $^ $(OMP_LIBS) $(BACKEND_LIBS) -o $@

$(BUILD)/test_gemm_serial: $(call serial_obj,$(LIB_SRC) tests/test_gemm.cpp)
	$(CXX) $^ $(BACKEND_LIBS) -o $@

$(BUILD)/test_gemm_omp: $(call omp_obj,$(LIB_SRC) tests/test_gemm.cpp)
	$(MPICXX) $^ $(OMP_LIBS) $(BACKEND_LIBS) -o $@

$(BUILD)/test_gradcheck: $(call serial_obj,$(LIB_SRC) tests/test_gradcheck.cpp)
	$(CXX) $^ $(BACKEND_LIBS) -o $@

$(BUILD)/test_allreduce: $(call omp_obj,$(PARALLEL_SRC) $(BACKEND_SRC) tests/test_allreduce.cpp)
	$(MPICXX) $^ $(OMP_LIBS) $(BACKEND_LIBS) -o $@

$(BUILD)/test_alltoall: $(call omp_obj,$(PARALLEL_SRC) $(BACKEND_SRC) tests/test_alltoall.cpp)
	$(MPICXX) $^ $(OMP_LIBS) $(BACKEND_LIBS) -o $@

$(BUILD)/compare_params: $(call serial_obj,tests/compare_params.cpp)
	$(CXX) $^ $(BACKEND_LIBS) -o $@

# --- tests (always double precision) -----------------------------------------
TB := build/$(PLATFORM)-double
test:
	$(MAKE) REAL=double tests parallel serial
	$(TB)/test_gemm_serial
	OMP_NUM_THREADS=4 $(TB)/test_gemm_omp
	$(TB)/test_gradcheck
	for p in 1 2 3 4 5 8; do $(MPIRUN) -np $$p $(TB)/test_allreduce || exit 1; done
	for p in 1 2 3 4 5 8; do $(MPIRUN) -np $$p $(TB)/test_alltoall || exit 1; done
	tests/test_equivalence.sh $(TB) "$(MPIRUN)"

# --- data and experiments ----------------------------------------------------
data:
	$(PYTHON) scripts/get_data.py --out data/fashion_mnist

tables:
	$(PYTHON) scripts/parse_logs.py --raw results/raw --out results/tables

figures:
	$(PYTHON) scripts/plot.py --tables results/tables --out results/figures

clean:
	rm -rf build

-include $(shell find build -name '*.d' 2>/dev/null)
