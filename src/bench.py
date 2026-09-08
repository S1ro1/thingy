"""Residual add -> RMSNorm -> 1x128 FP8 activation benchmark.

One torch.compile region, with the actual Prime-RL Triton quantizer vendored from
04a61d3b75c3c99f263b2c133e822f998909adf7 under cutie/vendor/prime_rl.
RMSNorm matches Prime-RL's PyTorch fallback (not its optional Quack kernel).
The quantization custom-op boundary mirrors Prime-RL's FP8 linear boundary,
but omits weight quantization and GEMM. FP32 scales never round to powers of two,
an intentional override of Prime-RL's Blackwell default.

Future fused kernel contract: (q [M,H] E4M3FN, scales [M,H/128] FP32,
residual_out [M,H] BF16, normalized [M,H] BF16).
Inputs and RMSNorm weights are BF16.
The candidate kernel is imported from rms_norm.py and launched on the current
PyTorch stream. Its interface must be extended to the full fusion below.
No CUDA graphs or backward pass.
"""

import argparse
import functools

import cuda.bindings.driver as cuda_driver
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.testing import JitArguments, benchmark

from cutie.vendor.prime_rl.fp8_utils import per_token_cast_to_fp8_triton
from rms_norm import RMSNorm

GROUP_SIZE = 128

test_configs = [
    (1024, 3072),
    (1024, 7168),
    (4096, 3072),
    (4096, 7168),
    (16384, 3072),
    (16384, 7168),
]


def rms_norm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Prime-RL's PyTorch fallback: FP32 statistics, cast before weight multiply."""
    x_f32 = x.float()
    variance = x_f32.square().mean(dim=-1, keepdim=True)
    normalized = x_f32 * torch.rsqrt(variance + eps)
    return normalized.to(x.dtype) * weight


@torch.library.custom_op("cutie::per_token_fp8_quant", mutates_args=())
def quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Prime-RL quantizes inside its opaque fp8_blockwise_mm custom op. We stop
    # before GEMM, so this smaller boundary contains only its activation quantizer.
    return per_token_cast_to_fp8_triton(x, use_ue8m0=False)


@quantize_fp8.register_fake
def _quantize_fp8_fake(x):
    return (
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        x.new_empty((x.shape[0], x.shape[1] // GROUP_SIZE), dtype=torch.float32),
    )


def residual_rms_norm_fp8_reference(
    x,
    residual,
    weight,
    eps=1e-6,
    out=None,
    scale=None,
    residual_out=None,
    normalized_out=None,
):
    # Like the old RMSNorm baseline, accept the kernel's output arguments but
    # return independently computed reference tensors instead of writing them.
    # Materialize the BF16 residual sum, then normalize with FP32 statistics.
    # Return the sum for the next residual connection; never mutate the input.
    residual_out = x + residual
    normalized = rms_norm_reference(residual_out, weight, eps)
    q, scale = quantize_fp8(normalized)
    return q, scale, residual_out, normalized


# One compiled region includes addition, RMSNorm, and the quantization custom op.
# Inductor can fuse the torch operations; the vendored Triton launch stays separate.
# "Unfused" means no hand-written residual + RMSNorm + quantization kernel yet.
rms_norm_fp8_unfused = torch.compile(
    residual_rms_norm_fp8_reference, fullgraph=True, dynamic=False
)


def workspace_to_cute(x, residual, weight, out, scale, residual_out, normalized_out):
    return tuple(
        from_dlpack(tensor, assumed_align=32)
        for tensor in (x, residual, weight, out, scale, residual_out, normalized_out)
    )


def workspace_generator(
    M: int, H: int, eps: float = 1e-6, to_cute: bool = False
) -> JitArguments:
    x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(H, dtype=torch.bfloat16, device="cuda")
    # NaNs make missing kernel writes fail correctness checks.
    out = torch.full_like(x, float("nan"), dtype=torch.float8_e4m3fn)
    scale = torch.full(
        (M, H // GROUP_SIZE), float("nan"), device=x.device, dtype=torch.float32
    )
    residual_out = torch.full_like(x, float("nan"))
    normalized_out = torch.full_like(x, float("nan"))
    if to_cute:
        x, residual, weight, out, scale, residual_out, normalized_out = (
            workspace_to_cute(
                x, residual, weight, out, scale, residual_out, normalized_out
            )
        )
    return JitArguments(
        x=x,
        residual=residual,
        weight=weight,
        eps=eps,
        out=out,
        scale=scale,
        residual_out=residual_out,
        normalized_out=normalized_out,
    )


def fused_memory_bytes(M: int, H: int) -> int:
    """Ideal full-fusion traffic: each input read and each output written once.

    Count weights once across all rows (ideal cache reuse). Exclude intermediate
    tensors. This is effective bandwidth, not measured DRAM traffic.
    """
    reads = 2 * M * H + 2 * M * H + 2 * H  # BF16 x, residual, weight
    # FP8 activations, FP32 scales, BF16 residual and normalized activations.
    writes = M * H + 4 * M * (H // GROUP_SIZE) + 2 * M * H + 2 * M * H
    return reads + writes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int)
    parser.add_argument("--H", type=int)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--elements_per_thread", type=int, default=8)
    parser.add_argument("--threads_per_row", type=int, default=128)
    parser.add_argument("--load-path", choices=("shared", "direct"), default="shared")
    parser.add_argument("--reduction", choices=("warp0", "all"), default="all")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--workspace-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (args.M is None) != (args.H is None):
        parser.error("Pass --M and --H together, or omit both to run all test shapes")
    if args.M is not None and (args.M <= 0 or args.H <= 0 or args.H % GROUP_SIZE):
        parser.error("M must be positive and H must be a positive multiple of 128")
    if (
        args.eps <= 0
        or args.warmup < 1
        or args.iterations < 1
        or args.workspace_count < 1
    ):
        parser.error("eps, warmup, iterations, and workspace-count must be positive")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    shapes = [(args.M, args.H)] if args.M is not None else test_configs

    norm = RMSNorm(
        elements_per_thread=args.elements_per_thread,
        threads_per_row=args.threads_per_row,
        load_path=args.load_path,
        reduction=args.reduction,
    )
    # Keep the current stream and non-graph timing requested for this harness.
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    for M, H in shapes:
        test_input = workspace_generator(M, H, eps=args.eps)
        (
            x_cute,
            residual_cute,
            weight_cute,
            out_cute,
            scale_cute,
            residual_out_cute,
            normalized_out_cute,
        ) = workspace_to_cute(
            test_input.kwargs["x"],
            test_input.kwargs["residual"],
            test_input.kwargs["weight"],
            test_input.kwargs["out"],
            test_input.kwargs["scale"],
            test_input.kwargs["residual_out"],
            test_input.kwargs["normalized_out"],
        )
        cutie_norm = cute.compile(
            norm,
            x=x_cute,
            residual=residual_cute,
            weight=weight_cute,
            eps=args.eps,
            out=out_cute,
            scale=scale_cute,
            residual_out=residual_out_cute,
            normalized_out=normalized_out_cute,
            stream=make_fake_stream(),
        )
        cutie_on_stream = functools.partial(cutie_norm, stream=stream)

        baseline_result = rms_norm_fp8_unfused(**test_input.kwargs)
        cutie_on_stream(
            x=x_cute,
            residual=residual_cute,
            weight=weight_cute,
            eps=args.eps,
            out=out_cute,
            scale=scale_cute,
            residual_out=residual_out_cute,
            normalized_out=normalized_out_cute,
        )
        # Check the four kernel outputs against Prime-RL, never against a
        # separate quantization reference. Allow one FP8 step near rounding ties.
        output_names = ("out", "scale", "residual_out", "normalized_out")
        tolerances = ((0.15, 2**-8), (0.02, 1e-7), (0, 0), (0.02, 1e-5))
        passed = "✅"
        for name, reference, (rtol, atol) in zip(
            output_names, baseline_result, tolerances
        ):
            actual = test_input.kwargs[name]
            try:
                is_close = torch.allclose(
                    actual.float(),
                    reference.float(),
                    rtol=rtol,
                    atol=atol,
                )
            except Exception as _:
                is_close = False

            if not is_close:
                passed = "❌"

        time_us = benchmark(
            rms_norm_fp8_unfused,
            workspace_generator=functools.partial(
                workspace_generator, M, H, eps=args.eps
            ),
            workspace_count=args.workspace_count,
            warmup_iterations=args.warmup,
            iterations=args.iterations,
            stream=stream,
            use_cuda_graphs=False,
        )
        kernel_time_us = benchmark(
            cutie_on_stream,
            workspace_generator=functools.partial(
                workspace_generator, M, H, eps=args.eps, to_cute=True
            ),
            workspace_count=args.workspace_count,
            warmup_iterations=args.warmup,
            iterations=args.iterations,
            stream=stream,
            use_cuda_graphs=False,
        )
        traffic_bytes = fused_memory_bytes(M, H)
        effective_tbps = traffic_bytes / (time_us * 1e6)
        result = (
            f"M: {M}, H: {H}, Correctness: {passed}, Unfused: {time_us:.3f} us, "
            f"Cutie: {kernel_time_us:.3f} us, Speedup: {time_us / kernel_time_us:.2f}x, "
            f"Fused traffic: {traffic_bytes / 1e6:.3f} MB, "
            f"Effective BW: baseline {effective_tbps:.3f}, "
            f"Cutie {traffic_bytes / (kernel_time_us * 1e6):.3f} TB/s"
        )
        print(result)


if __name__ == "__main__":
    main()
