import argparse
import functools

import cuda.bindings.driver as cuda_driver
import torch
from cutlass import const_expr, cute
from cutlass.cute.runtime import make_fake_stream
from cutlass.testing import benchmark
from cutlass.utils import SmemAllocator

from cutie.utils import get_mem_util, workspace_generator, workspace_to_cute

torch._dynamo.config.recompile_limit = 16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=None, required=False)
    parser.add_argument("--H", type=int, default=None, required=False)
    parser.add_argument("--elements_per_thread", type=int, default=8)
    parser.add_argument("--threads_per_row", type=int, default=128)
    parser.add_argument(
        "--load-path",
        choices=("shared", "direct"),
        default="shared",
        help="Stage X asynchronously through shared memory, or load X directly into registers.",
    )
    parser.add_argument(
        "--reduction",
        choices=("warp0", "all"),
        default="all",
        help="Finish the row reduction in warp 0 and broadcast, or repeat it in every warp.",
    )
    return parser.parse_args()


test_configs = [
    (1024, 3072),
    (1024, 7168),
    (4096, 3072),
    (4096, 7168),
    (16384, 3072),
    (16384, 7168),
]


@torch.compile(dynamic=False)
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True) + eps
    x = x / torch.sqrt(variance)
    return x * weight


def rms_norm_on_stream(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None,
    *,
    stream: torch.cuda.Stream,
) -> torch.Tensor:
    """Launch the compiled PyTorch baseline on a specific CUDA stream."""
    with torch.cuda.stream(stream):
        return rms_norm(x, weight, eps, out)


class RMSNorm:
    def __init__(
        self,
        elements_per_thread: int,
        threads_per_row: int,
        load_path: str = "shared",
        reduction: str = "all",
    ):
        if load_path not in ("shared", "direct"):
            raise ValueError("load_path must be 'shared' or 'direct'")
        if reduction not in ("warp0", "all"):
            raise ValueError("reduction must be 'warp0' or 'all'")
        if elements_per_thread <= 0:
            raise ValueError("elements_per_thread must be positive")
        if not 32 <= threads_per_row <= 1024 or threads_per_row % 32:
            raise ValueError("threads_per_row must be 32–1024, in full warps")
        self.elements_per_thread = elements_per_thread
        self.threads_per_row = threads_per_row
        # These Python configuration values select paths at compile time.
        self.load_path = load_path
        self.reduction = reduction

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        stream: cuda_driver.CUstream,
        eps: float = 1e-5,
        out: cute.Tensor | None = None,
    ) -> None:
        if const_expr(
            x.shape[1] % (self.threads_per_row * self.elements_per_thread) != 0
        ):
            raise ValueError(
                "Row width must be divisible by threads_per_row * elements_per_thread"
            )
        if const_expr(self.load_path == "shared") and const_expr(
            (self.elements_per_thread * x.element_type.width) % 128 != 0
        ):
            raise ValueError(
                "Shared staging requires vectors divisible into 128-bit copies"
            )
        grid = (x.shape[0], 1, 1)
        block = (self.threads_per_row, 1, 1)

        thr_layout = cute.make_layout(
            (1, self.threads_per_row), stride=(self.threads_per_row, 1)
        )
        val_layout = cute.make_layout(
            (1, self.elements_per_thread), stride=(self.elements_per_thread, 1)
        )

        tiled_copy_g2r = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), x.element_type),
            thr_layout=thr_layout,
            val_layout=val_layout,
        )
        tiled_copy_g2s = None
        if const_expr(self.load_path == "shared"):
            tiled_copy_g2s = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(),
                    x.element_type,
                    num_bits_per_copy=128,
                ),
                thr_layout=thr_layout,
                val_layout=val_layout,
            )

        self.rms_norm_kernel(
            x, weight, eps, out, tiled_copy_g2r, tiled_copy_g2s
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def rms_norm_kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        eps: cute.Float32,
        mO: cute.Tensor,
        tiled_copy_g2r: cute.TiledCopy,
        tiled_copy_g2s: cute.TiledCopy | None,
    ) -> None:
        salloc = SmemAllocator()

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        num_warps = self.threads_per_row // cute.arch.WARP_SIZE

        tiler_mn = (1, mX.shape[1])

        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gO = cute.local_tile(mO, tiler_mn, (bidx, 0))
        gW = cute.make_tensor(
            mW.iterator, cute.make_layout((1, mW.shape[0]), stride=(0, mW.stride[0]))
        )

        thr_g2r = tiled_copy_g2r.get_slice(tidx)
        # Use the same value grouping for X, weights, and output arithmetic.
        tXgX = thr_g2r.partition_S(gX)
        tWgW = thr_g2r.partition_S(gW)
        tOgO = thr_g2r.partition_D(gO)

        rX = cute.make_rmem_tensor_like(tXgX)
        rW = cute.make_rmem_tensor_like(tWgW)
        rO = cute.make_rmem_tensor_like(tOgO)

        thread_sum = cute.Float32(0.0)

        if const_expr(self.load_path == "shared"):
            # Async global → shared staging overlaps with the weight load.
            # The async atom has its own value grouping; use its views for G2S.
            sX = salloc.allocate_tensor(
                mX.element_type,
                cute.make_layout((1, mX.shape[1]), stride=(mX.shape[1], 1)),
                byte_alignment=16,
            )
            thr_g2s = tiled_copy_g2s.get_slice(tidx)
            tXgX_async = thr_g2s.partition_S(gX)
            tXsX_async = thr_g2s.partition_D(sX)
            cute.copy(tiled_copy_g2s, tXgX_async, tXsX_async)
            cute.arch.cp_async_commit_group()
            cute.copy(tiled_copy_g2r, tWgW, rW)
            cute.arch.cp_async_wait_group(0)

            # Read shared memory with the compute grouping used by rX/rW/rO.
            # Each thread reads its own staged values, so no block barrier is needed.
            tCsX = thr_g2r.partition_S(sX)
            cute.autovec_copy(tCsX, rX)
        else:
            # Direct global → registers: no shared input row or async-copy wait.
            # Load weights early so independent memory operations can overlap.
            cute.copy(tiled_copy_g2r, tXgX, rX)
            cute.copy(tiled_copy_g2r, tWgW, rW)

        # Both paths retain all thread-owned X values for the output calculation.
        x_chunk = rX.load().to(cute.Float32)
        thread_sum += (x_chunk * x_chunk).reduce(
            cute.ReductionOp.ADD,
            init_val=cute.Float32(0.0),
            reduction_profile=0,
        )

        warp_sum_x = cute.arch.warp_reduction_sum(thread_sum)

        partials = salloc.allocate_tensor(
            cute.Float32,
            cute.make_layout(num_warps),
            byte_alignment=16,
        )

        if lane_idx == 0:
            partials[warp_idx] = warp_sum_x

        # Publish one sum per warp before any warp reads the other partials.
        cute.arch.sync_threads()

        row_sum = cute.Float32(0.0)
        if const_expr(self.reduction == "warp0"):
            # Only warp 0 combines the partials. Lane 0 broadcasts inverse RMS
            # through shared memory; the second barrier makes that write visible.
            if warp_idx == 0:
                if lane_idx < num_warps:
                    row_sum = partials[lane_idx]
                row_sum = cute.arch.warp_reduction_sum(row_sum)
                if lane_idx == 0:
                    partials[0] = cute.rsqrt(row_sum / cute.Float32(mX.shape[1]) + eps)
            cute.arch.sync_threads()
            inv_rms = partials[0]
        else:
            # Every warp repeats the small final reduction and computes inverse RMS.
            # The partials are never overwritten, so no second barrier is needed.
            if lane_idx < num_warps:
                row_sum = partials[lane_idx]
            row_sum = cute.arch.warp_reduction_sum(row_sum)
            inv_rms = cute.rsqrt(row_sum / cute.Float32(mX.shape[1]) + eps)

        result = x_chunk * inv_rms * rW.load().to(cute.Float32)
        rO.store(result.to(rO.element_type))

        cute.copy(tiled_copy_g2r, rO, tOgO)


if __name__ == "__main__":
    args = parse_args()
    norm = RMSNorm(
        elements_per_thread=args.elements_per_thread,
        threads_per_row=args.threads_per_row,
        load_path=args.load_path,
        reduction=args.reduction,
    )
    baseline_torch_stream = torch.cuda.Stream()
    cutie_torch_stream = torch.cuda.Stream()
    baseline_stream = cuda_driver.CUstream(baseline_torch_stream.cuda_stream)
    cutie_stream = cuda_driver.CUstream(cutie_torch_stream.cuda_stream)

    for M, H in test_configs:
        if args.M and args.H and (M != args.M or H != args.H):
            continue
        test_input = workspace_generator(M, H, to_cute=False)
        x_cute, weight_cute, out_cute = workspace_to_cute(
            test_input.kwargs["x"],
            test_input.kwargs["weight"],
            test_input.kwargs["out"],
        )
        cutie_norm = cute.compile(
            norm,
            x=x_cute,
            weight=weight_cute,
            eps=test_input.kwargs["eps"],
            out=out_cute,
            stream=make_fake_stream(),
        )

        baseline_on_stream = functools.partial(
            rms_norm_on_stream, stream=baseline_torch_stream
        )
        cutie_on_stream = functools.partial(cutie_norm, stream=cutie_stream)

        baseline_result = rms_norm(
            test_input.kwargs["x"],
            test_input.kwargs["weight"],
            test_input.kwargs["eps"],
        )
        cutie_torch_stream.wait_stream(torch.cuda.current_stream())
        cutie_on_stream(
            x=x_cute,
            weight=weight_cute,
            eps=test_input.kwargs["eps"],
            out=out_cute,
        )
        cutie_torch_stream.synchronize()
        is_correct = (
            "❌"
            if not torch.allclose(baseline_result, test_input.kwargs["out"])
            else "✅"
        )
        baseline_time_us = benchmark(
            baseline_on_stream,
            workspace_generator=functools.partial(
                workspace_generator, M, H, torch_stream=baseline_torch_stream
            ),
            warmup_iterations=20,
            iterations=200,
            workspace_count=200,
            stream=baseline_stream,
            use_cuda_graphs=True,
        )
        cutlass_time_us = benchmark(
            cutie_on_stream,
            workspace_generator=functools.partial(
                workspace_generator,
                M,
                H,
                to_cute=True,
                torch_stream=cutie_torch_stream,
            ),
            warmup_iterations=20,
            iterations=200,
            workspace_count=200,
            stream=cutie_stream,
            use_cuda_graphs=True,
        )
        baseline_mem_util = get_mem_util(baseline_time_us, M, H)
        cutlass_mem_util = get_mem_util(cutlass_time_us, M, H)
        print(
            f"M: {M}, H: {H}, Load: {args.load_path}, Reduction: {args.reduction}, Correctness: {is_correct}, Baseline: {baseline_mem_util:.2f} % vs Cutie: {cutlass_mem_util:.2f} %, Speedup: {baseline_time_us / cutlass_time_us:.2f}x"
        )
