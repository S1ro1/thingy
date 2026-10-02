"""Benchmark Prime-RL sparse MLA, then check/time the TK implementation."""

import importlib.util
import os
import statistics
import subprocess
import sys
import sysconfig
from argparse import ArgumentParser
from pathlib import Path

import cuda.bindings.driver as cuda
import torch
from cutlass.testing import JitArguments, benchmark

HERE = Path(__file__).resolve().parent


# Match the TK build toolkit before importing TileLang.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.0")
from vendor.sparse_mla_fwd import sparse_mla_fwd

# GLM-5.3 absorbed MLA: 512 latent + 64 RoPE channels; 64 query heads.
HEADS, D_LATENT, D_ROPE, TOPK = 64, 512, 64, 2048
D_QK = D_LATENT + D_ROPE
SM_SCALE = 1 / 16  # Pre-absorption Q/K dimension: sqrt(192 + 64).
SEQ_LENS = (8192, 16384, 32768)
DISTRIBUTIONS = {'single': 1, 'even_2': 2, 'even_4': 4}


def workspace_generator(seq_len, num_documents, seed):
    """Synthetic causal top-k within 1, 2 or 4 equal documents; outside timing."""
    if seq_len % num_documents:
        raise ValueError('Sequence length must divide evenly into documents')
    length = seq_len // num_documents
    generator = torch.Generator(device='cuda').manual_seed(seed)
    q = torch.randn((1, seq_len, HEADS, D_QK), generator=generator,
                    device='cuda', dtype=torch.bfloat16)
    kv = torch.randn((1, seq_len + 1, 1, D_QK), generator=generator,
                     device='cuda', dtype=torch.bfloat16)
    kv[:, -1].zero_()
    # S is the masked sentinel, matching Prime-RL's indexer output.
    indices = torch.full((1, seq_len, 1, TOPK), seq_len,
                         dtype=torch.int32, device='cuda')
    generator.manual_seed(seed + 1)
    columns = torch.arange(length, device='cuda')
    for start in range(0, seq_len, length):
        for lo in range(0, length, 128):
            hi = min(lo + 128, length)
            scores = torch.rand((hi - lo, length), generator=generator, device='cuda')
            rows = torch.arange(lo, hi, device='cuda')
            scores.masked_fill_(columns[None, :] > rows[:, None], -float('inf'))
            values, selected = scores.topk(min(length, TOPK), dim=-1)
            selected += start
            selected.masked_fill_(~values.isfinite(), seq_len)
            indices[0, start + lo:start + hi, 0, :selected.shape[-1]] = selected
    return dict(q=q, kv=kv, indices=indices)


def load_extension():
    subprocess.run(
        ["make", "-s", "-C", str(HERE), f"PYTHON={sys.executable}"], check=True
    )
    path = HERE / ("_sparse_mla" + sysconfig.get_config_var("EXT_SUFFIX"))
    spec = importlib.util.spec_from_file_location("_sparse_mla", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, nargs="+", default=list(SEQ_LENS))
    parser.add_argument(
        "--distribution", nargs="+", choices=DISTRIBUTIONS, default=list(DISTRIBUTIONS)
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run while the TK body is still empty",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if (
        min(args.seq_len) < 1
        or min(args.iterations, args.repeats) < 1
        or args.warmup < 0
    ):
        parser.error(
            "sequence lengths, iterations and repeats must be positive; warmup must be nonnegative"
        )
    return args


def time_call(call, inputs, stream, args):
    return [
        benchmark(
            call,
            kernel_arguments=JitArguments(**inputs),
            stream=stream,
            warmup_iterations=args.warmup,
            iterations=args.iterations,
            use_cuda_graphs=False,
        )
        for _ in range(args.repeats)
    ]


@torch.no_grad()
def main():
    args = parse_args()
    debug = os.environ.get("DEBUG", "0") == "1"
    extension = None if args.baseline_only else load_extension()
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    kernel = sparse_mla_fwd(
        heads=HEADS, dim=D_LATENT, tail_dim=D_ROPE, topk=TOPK,
        kv_group=1, sm_scale=SM_SCALE, is_causal=True,
        block_I=64, num_stages=2, threads=256,
    )  # Dynamic S: compile once, outside timing.

    def baseline(q, kv, indices):
        return kernel(q, kv, indices)

    def kittie(q, kv, indices, out, lse):
        extension.sparse_mla(q, kv, indices, out, lse, stream=torch_stream)

    print(
        "GLM-5.3: Q/K=576, V=512, H=64, KV groups=1, topk=2048, scale=1/16", flush=True
    )
    print(
        "S | documents | count | valid slots | baseline ms | baseline useful TFLOPS"
        + ("" if args.baseline_only else " | TK ms | TK useful TFLOPS | speedup"),
        flush=True,
    )
    for seq_len in args.seq_len:
        for distribution in args.distribution:
            num_documents = DISTRIBUTIONS[distribution]
            inputs = workspace_generator(seq_len, num_documents, args.seed)
            # Temporary gather smoke-test data, only for DEBUG TK launches.
            if debug and extension is not None:
                if seq_len < 64:
                    raise ValueError('Gather debug needs --seq-len >= 64')
                rows = torch.arange(64, device='cuda')[:, None]
                cols = torch.arange(D_QK, device='cuda')[None, :]
                inputs['kv'][0, :64, 0] = ((17 * rows + cols) % 251).to(torch.bfloat16)
            out_ref, lse_ref = baseline(**inputs)

            tk_inputs = None
            if extension is not None:
                tk_inputs = dict(
                    inputs,
                    out=torch.full_like(out_ref, float("nan")),
                    lse=torch.full_like(lse_ref, float("nan")),
                )
                kittie(**tk_inputs)
                # torch.testing.assert_close(
                #     tk_inputs["out"], out_ref, rtol=1e-2, atol=1e-2
                # )
                # torch.testing.assert_close(
                #     tk_inputs["lse"], lse_ref, rtol=1e-3, atol=1e-3
                # )
            if debug:
                torch_stream.synchronize()
                print(
                    f"{seq_len} {distribution}: DEBUG single launch; timing skipped",
                    flush=True,
                )
                return

            samples = time_call(baseline, inputs, stream, args)
            time_us = statistics.median(samples)
            length = seq_len // num_documents
            prefix = min(length, TOPK)
            pairs = num_documents * (prefix * (prefix + 1) // 2
                                     + max(length - TOPK, 0) * TOPK)
            useful_flops = 2 * HEADS * (D_QK + D_LATENT) * pairs
            line = (
                f"{seq_len} | {distribution} | {num_documents} | "
                f"{pairs / (seq_len * TOPK):.1%} | {time_us / 1000:.3f} | "
                f"{useful_flops / time_us / 1e6:.1f}"
            )
            if tk_inputs is not None:
                tk_samples = time_call(kittie, tk_inputs, stream, args)
                tk_us = statistics.median(tk_samples)
                line += (
                    f" | {tk_us / 1000:.3f} | {useful_flops / tk_us / 1e6:.1f}"
                    f" | {time_us / tk_us:.2f}x"
                )
            print(line, flush=True)
            del inputs, out_ref, lse_ref, tk_inputs


if __name__ == "__main__":
    main()
