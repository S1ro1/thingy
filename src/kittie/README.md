# Kittie

`sparse_mla/` contains the [GLM-5.3 sparse-attention project](sparse_mla/README.md):
a vendored Prime-RL TileLang baseline, packed-document workload suite, and empty
TK kernel scaffold. Install its dependencies with `uv sync --extra attention`.

ThunderKittens experiments alongside the CuTe kernels. Start in `gemm/gemm.cu`:
the GEMM body is intentionally empty. The initial contract is contiguous BF16
`A[M,K]`, `B[N,K]`, and `C[M,N]`, computing `C = A @ B.T`.

## Build and run

From the repository root:

```sh
uv sync
git submodule update --init src/kittie/third_party/ThunderKittens
make -C src/kittie/gemm
```

After implementing the GEMM body:

```sh
uv run src/kittie/gemm/bench.py --M 4096 --N 4096 --K 4096
uv run src/kittie/gemm/bench.py  # same shape suite as cutie
```

These commands automatically perform an incremental build. The benchmark follows
Cutie's normal flow: generate inputs, time PyTorch, check kernel output against
the reference, then time the kernel and report TFLOPS and speedup. It reuses the
same input generation, tolerance, and timing with rotating workspaces.
`DEBUG=1` runs one shape with a single kernel launch and correctness check,
synchronizes to flush device prints, and skips timing. In device code,
`DEBUG_PRINT("tile=%d\n", tile)` prints from thread 0 of CTA 0;
`DEBUG_PRINT_IF(condition, "tile=%d\n", tile)` selects a different caller.
Both macros compile to no-ops unless built with `DEBUG=1`. The benchmark forwards
the environment to `make`, which rebuilds automatically when this setting changes.

Set `BM`, `BN`, `BK`, and `NUM_WARPS` in `gemm.cu`. Initially M/N/K must be divisible by
BM/BN/BK respectively. The grid assigns one output tile to each CTA; persistence
and pipeline stages are yours to add.

## Lightweight extension

The build follows [ThunderKittens' Python binding path](https://github.com/HazyResearch/ThunderKittens/tree/main/include/pyutils):
one NVCC invocation, pybind11, and the upstream `py::bind_kernel` adapter.
It uses Python tensor objects and their device pointers without Torch C++ headers
or LibTorch linking. PyTorch remains the Python allocator/reference/benchmark.
Included headers and build flags are tracked so unchanged builds skip NVCC.

Defaults target this remote's B300: `ARCH=103a`, `KITTENS_SM103`, and CUDA 13.0.
The toolkit is selected explicitly; `/usr/local/cuda` still points to CUDA 12.8.
You can override `CUDA_PATH`, `ARCH`, and `PYTHON` on `make`, for example:

```sh
make -C src/kittie/gemm CUDA_PATH=/usr/local/cuda-13.0 ARCH=103a
```

The remote has `cuda-nvcc-13-0`, `cuda-cudart-dev-13-0`, and
`libcurand-dev-13-0` installed. The last package supplies a header clangd's CUDA
wrapper needs even though this kernel does not use random-number generation.
ThunderKittens is a pinned Git submodule; build outputs are ignored.

## Zed navigation on the remote

The project settings map `.cu`/`.cuh` to C++ and select the installed remote
`/root/.local/bin/clangd` (23.1.0). The checked-in `compile_flags.txt` supplies
CUDA, TK, Python, and pybind11 include paths automatically when a file opens.
No build or editor setup command is needed.

Open `gemm/gemm.cu` in the remote Zed project. If it was already open, run
`editor: restart language server` from Zed's command palette. Go-to-definition
on `st_bf` should enter the TK headers. The build task is also available
through `task: spawn`. These settings follow
[Zed's clangd configuration](https://zed.dev/docs/languages/cpp).

The editor configuration targets this remote's CUDA 13.0, SM103a, and Python 3.13
environment. If you change those, update `compile_flags.txt` as well as your build
settings. On another machine, also update the clangd path in `.zed/settings.json`.
