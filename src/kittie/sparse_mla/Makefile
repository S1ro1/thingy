# Upstream CONFIG=python style: NVCC + pybind11, no LibTorch C++ dependency.
ROOT := $(abspath ../../..)
PYTHON ?= $(ROOT)/.venv/bin/python
CUDA_PATH ?= /usr/local/cuda-13.0
NVCC := $(CUDA_PATH)/bin/nvcc
ARCH ?= 103a
TK := $(ROOT)/src/kittie/third_party/ThunderKittens
OUT := _sparse_mla$(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
PY_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
DEFINES := -DKITTENS_SM$(subst a,,$(ARCH)) -DNDEBUG
ifeq ($(DEBUG),1)
DEFINES += -DKITTIE_DEBUG
endif
DEFINES += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
DEFINES += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
# Avoid the generic PTX fallback from -arch: tcgen05 requires the "a" target.
FLAGS := -std=c++20 -O3 --use_fast_math -lineinfo -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
FLAGS += --expt-extended-lambda --expt-relaxed-constexpr -shared -Xcompiler=-fPIC
FLAGS += $(DEFINES) $(PY_INCLUDES) -I$(TK)/include -I$(TK)/prototype
FLAGS += -L$(CUDA_PATH)/lib64 -lcuda -lcudart_static -ldl -lrt -lpthread

.DEFAULT_GOAL := all
.PHONY: all run clean FORCE
all: $(OUT)

# Rebuild on included-header edits and compile-option changes, not just sparse_mla.cu.
build/options: FORCE
	@mkdir -p build
	@printf '%s\n' '$(NVCC) $(FLAGS)' > $@.tmp
	@cmp -s $@.tmp $@ || cp $@.tmp $@
	@rm $@.tmp

$(OUT): sparse_mla.cu Makefile build/options
	$(NVCC) $(FLAGS) -MMD -MP -MF build/sparse_mla.d -MT $@ $< -o $@

run: all
	$(PYTHON) bench.py $(ARGS)
clean:
	rm -f $(OUT) build/sparse_mla.d build/options

-include build/sparse_mla.d
