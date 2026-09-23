import functools
import os
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from itertools import product

import cuda.bindings.driver as cuda_driver
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.testing import JitArguments, benchmark

from cutie.gemm.gemm import Gemm

test_configs = [
    (2048, 2048, 2048),
    (2048, 3072, 3072),
    (2048, 4096, 4096),
    (2048, 6144, 6144),
    (4096, 2048, 2048),
    (4096, 3072, 3072),
    (4096, 4096, 4096),
    (4096, 6144, 6144),
    (6144, 2048, 2048),
    (6144, 3072, 3072),
    (6144, 4096, 4096),
    (6144, 6144, 6144),
    (8192, 2048, 2048),
    (8192, 3072, 3072),
    (8192, 4096, 4096),
    (8192, 6144, 6144),
    (16384, 2048, 2048),
    (16384, 3072, 3072),
    (16384, 4096, 4096),
    (16384, 6144, 6144),
]


@dataclass(frozen=True)
class Config:
    BM: int
    BN: int
    BK: int
    num_smem_stages: int
    scheduling: str
    super_m: int
    num_acc_stages: int
    num_m_ctas: int


def candidate_configs(M, N, K, smem_capacity, num_sms):
    """Search the requested bounded space, filtering unsupported geometries."""
    # A one-CTA MMA supports M=128; two-CTA MMA supports M=128 or M=256.
    geometries = product(((1, 128), (2, 128), (2, 256)), (128, 256), (32, 64))
    for (ctas, bm), bn, bk in geometries:
        if M % bm or N % bn or K % bk or num_sms % ctas:
            continue
        # BF16 A/B are both partitioned across the CTA pair. Reserve space for
        # alignment, pipeline barriers, and the TMEM allocator's bookkeeping.
        bytes_per_stage = (bm + bn) * bk * 2 // ctas
        max_stages = min(K // bk, (smem_capacity - 2048) // bytes_per_stage)
        stages = [s for s in (1, 4) if s <= max_stages]
        # Each FP32 accumulator uses BN columns; TMEM has 512 per SM.
        accumulators = [a for a in (2,) if bn * a <= 512]
        for smem_stages, acc_stages, group in product(stages, accumulators, (4, 8, 12)):
            yield Config(
                bm, bn, bk, smem_stages,
                "super_m", group, acc_stages, ctas,
            )


def parse_args():
    parser = ArgumentParser(description="Run GEMM benchmark")

    parser.add_argument(
        "--BM", type=int, help="Block size for M dimension", default=128
    )
    parser.add_argument(
        "--BN", type=int, help="Block size for N dimension", default=256
    )
    parser.add_argument("--BK", type=int, help="Block size for K dimension", default=64)
    parser.add_argument(
        "--num_smem_stages", type=int, help="Number of shared memory stages", default=4
    )
    parser.add_argument(
        "--scheduling",
        type=str,
        help="Tile scheduling strategy",
        default="super_m",
        choices=["rowwise", "super_m"],
    )
    parser.add_argument("--super_m", type=int, default=12)
    parser.add_argument("--num_acc_stages", type=int, default=2)
    parser.add_argument("--num_m_ctas", type=int, default=1)
    parser.add_argument(
        "--find_best_config", "--find_config", dest="find_best_config",
        action="store_true",
        help="Search valid GEMM configurations per shape (overrides tile/path options)",
    )

    parser.add_argument("--M", type=int, help="Size for M dimension")
    parser.add_argument("--N", type=int, help="Size for N dimension")
    parser.add_argument("--K", type=int, help="Size for K dimension")

    args = parser.parse_args()
    dims = (args.M, args.N, args.K)
    if any(d is not None for d in dims) and not all(d is not None and d > 0 for d in dims):
        parser.error("provide positive --M, --N, and --K together")
    return args


def gemm_reference(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(A, B.T, out=out)


def workspace_to_cute(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    return tuple(from_dlpack(tensor, assumed_align=32) for tensor in (A, B, out))


def workspace_generator(
    M: int,
    N: int,
    K: int,
    to_cute: bool = False,
) -> JitArguments:
    A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

    if to_cute:
        A, B, out = workspace_to_cute(A, B, out)

    return JitArguments(
        A=A,
        B=B,
        out=out,
    )


def gemm_flops(M: int, N: int, K: int) -> int:
    return 2 * M * N * K


def time_us_to_tflops(time_us: float, op_flops: int) -> float:
    flops_per_sec = op_flops / (time_us * 1e-6)
    tflops_per_sec = flops_per_sec / 1e12

    return tflops_per_sec


def main():
    args = parse_args()
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    selected_config = Config(
        args.BM,
        args.BN,
        args.BK,
        args.num_smem_stages,
        args.scheduling,
        args.super_m,
        args.num_acc_stages,
        args.num_m_ctas,
    )

    if args.M and args.N and args.K:
        runnable = [(args.M, args.N, args.K)]
    else:
        runnable = test_configs

    for M, N, K in runnable:
        inputs = workspace_generator(M, N, K)
        flops = gemm_flops(M, N, K)

        A_cute, B_cute, out_cute = workspace_to_cute(
            inputs.kwargs["A"], inputs.kwargs["B"], inputs.kwargs["out"]
        )

        # from_dlpack shares storage: out_cute writes into inputs.kwargs["out"].
        # Keep the reference in a separate allocation so the kernel cannot
        # overwrite it (or inherit correct values for elements it never writes).
        baseline_result = gemm_reference(
            inputs.kwargs["A"],
            inputs.kwargs["B"],
            out=torch.empty_like(inputs.kwargs["out"]),
        )
        time_us_reference = benchmark(
            gemm_reference,
            workspace_generator=functools.partial(
                workspace_generator, M=M, N=N, K=K, to_cute=False
            ),
            stream=stream,
        )

        tflops_reference = time_us_to_tflops(time_us_reference, flops)
        configs = [selected_config]
        if args.find_best_config:
            device = torch.cuda.get_device_properties(torch.cuda.current_device())
            configs = list(candidate_configs(
                M, N, K, device.shared_memory_per_block_optin, device.multi_processor_count
            ))

        best_config, best_time = None, float("inf")
        for config in configs:
            if M % config.BM or N % config.BN or K % config.BK:
                raise ValueError("M, N, K must be divisible by BM, BN, BK")
            try:
                cutie_gemm = cute.compile(
                    Gemm(**asdict(config)), A=A_cute, B=B_cute, out=out_cute,
                    stream=make_fake_stream(),
                )
            except Exception:
                if not args.find_best_config:
                    raise
                continue

            cutie_gemm = functools.partial(cutie_gemm, stream=stream)
            inputs.kwargs["out"].fill_(float("nan"))
            # Runtime CUDA failures propagate: continuing with a poisoned device
            # context would make subsequent candidate results unreliable.
            cutie_gemm(A_cute, B_cute, out_cute)
            is_correct = torch.allclose(
                baseline_result, inputs.kwargs["out"], rtol=1e-2, atol=1e-2
            )
            if not is_correct:
                if not args.find_best_config:
                    raise AssertionError("Kernel output does not match PyTorch")
                continue

            time_us_cutie = benchmark(
                cutie_gemm,
                workspace_generator=functools.partial(
                    workspace_generator, M=M, N=N, K=K, to_cute=True
                ),
                stream=stream,
            )
            tflops_cutie = time_us_to_tflops(time_us_cutie, flops)
            if not args.find_best_config:
                print(
                    f"gemm({M}, {N}, {K}) | ✅ | Torch: {tflops_reference:.2f} TFLOPS | Cutie: {tflops_cutie:.2f} TFLOPS | Speedup: {(tflops_cutie / tflops_reference):.2f}X"
                )
            if time_us_cutie < best_time:
                best_config, best_time = config, time_us_cutie

        if args.find_best_config:
            if best_config is None:
                raise RuntimeError(f"No correct, benchmarkable config for M={M}, N={N}, K={K}")
            print(
                f"Best config for M={M}, N={N}, K={K}: {best_config} | "
                f"{best_time:.3f} us | {time_us_to_tflops(best_time, flops):.2f} TFLOPS | "
                f"Speedup: {time_us_reference / best_time:.2f}X", flush=True,
            )

        if os.environ.get("DEBUG", "0") == "1":
            break


if __name__ == "__main__":
    main()
