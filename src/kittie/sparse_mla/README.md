# Sparse MLA — GLM-5.3

`bench.py` compares the TK kernel against Prime-RL's TileLang forward.
`sparse_mla.cu` is an empty kernel with working bindings and an incremental build.

```sh
uv sync --extra attention
make -C src/kittie/sparse_mla
uv run --extra attention src/kittie/sparse_mla/bench.py --baseline-only

# Single case; omit --baseline-only once the TK kernel is implemented.
uv run --extra attention src/kittie/sparse_mla/bench.py --baseline-only \
  --seq-len 8192 --distribution even_2
```

Defaults: 8K/16K/32K tokens, each packed into 1, 2 or 4 equal documents.
Reports milliseconds and **useful TFLOPS** for both baseline and TK, excluding
padded slots (QK + PV, multiply-add = 2 FLOPs). Input generation is outside timing. `DEBUG=1` runs once without timing. The existing parent clangd config applies.

Q is BF16 `[1,S,64,576]`, KV is BF16 `[1,S+1,1,576]`, indices are int32
`[1,S,1,2048]`. Output is BF16 `[1,S,64,512]` with FP32 base-2 LSE `[1,S,64]`.
The scale is `1/16`. Indices select unique positions within each query's causal
document prefix; `S` is a masked sentinel pointing to the final zero KV row.

The forward definition in `vendor/sparse_mla_fwd.py` comes unchanged from
[Prime-RL d5f29c0](https://github.com/PrimeIntellect-ai/prime-rl/blob/d5f29c072a6731a17897f5aed661e0b0e6053d8c/src/prime_rl/trainer/models/kernels/sparse_mla_fwd.py),
without the backward import or autograd wrapper. Its license is in `vendor/LICENSE`.
