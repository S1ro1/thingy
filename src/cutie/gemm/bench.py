import functools
import os
from argparse import ArgumentParser

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
        "--num_smem_stages", type=int, help="Number of shared memory stages", default=1
    )

    parser.add_argument("--M", type=int, help="Size for M dimension")
    parser.add_argument("--N", type=int, help="Size for N dimension")
    parser.add_argument("--K", type=int, help="Size for K dimension")

    return parser.parse_args()


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
    gemm = Gemm(args.BM, args.BN, args.BK, args.num_smem_stages)

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

        cutie_gemm = cute.compile(
            gemm, A=A_cute, B=B_cute, out=out_cute, stream=make_fake_stream()
        )
        cutie_gemm = functools.partial(cutie_gemm, stream=stream)

        # from_dlpack shares storage: out_cute writes into inputs.kwargs["out"].
        # Keep the reference in a separate allocation so the kernel cannot
        # overwrite it (or inherit correct values for elements it never writes).
        baseline_result = gemm_reference(
            inputs.kwargs["A"],
            inputs.kwargs["B"],
            out=torch.empty_like(inputs.kwargs["out"]),
        )
        inputs.kwargs["out"].fill_(float("nan"))
        cutie_gemm(A_cute, B_cute, out_cute)

        is_correct = torch.allclose(
            baseline_result, inputs.kwargs["out"], rtol=1e-2, atol=1e-2
        )

        time_us_reference = benchmark(
            gemm_reference,
            workspace_generator=functools.partial(
                workspace_generator, M=M, N=N, K=K, to_cute=False
            ),
            stream=stream,
        )

        time_us_cutie = benchmark(
            cutie_gemm,
            workspace_generator=functools.partial(
                workspace_generator, M=M, N=N, K=K, to_cute=True
            ),
            stream=stream,
        )

        tflops_reference = time_us_to_tflops(time_us_reference, flops)
        tflops_cutie = time_us_to_tflops(time_us_cutie, flops)
        print(
            f"gemm({M}, {N}, {K}) | {'✅' if is_correct else '❌'} | Torch: {tflops_reference:.2f} TFLOPS | Cutie: {tflops_cutie:.2f} TFLOPS | Speedup: {(tflops_cutie / tflops_reference):.2f}X"
        )

        if os.environ.get("DEBUG", "0") == "1":
            break


if __name__ == "__main__":
    main()
