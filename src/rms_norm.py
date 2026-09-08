import argparse
import functools

import cuda.bindings.driver as cuda_driver
import torch
from cutlass import const_expr, cute, range_constexpr
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
        self.elements_per_thread = elements_per_thread
        self.threads_per_row = threads_per_row
        # These Python configuration values select paths at compile time.
        self.load_path = load_path
        self.reduction = reduction

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        weight: cute.Tensor,
        eps: float,
        out: cute.Tensor,
        scale: cute.Tensor,
        residual_out: cute.Tensor,
        normalized_out: cute.Tensor,
        stream: cuda_driver.CUstream,
    ) -> None:
        grid = (x.shape[0], 1, 1)
        block = (self.threads_per_row, 1, 1)

        thr_layout = cute.make_layout(
            (1, self.threads_per_row), stride=(self.threads_per_row, 1)
        )
        val_layout = cute.make_layout(
            (1, self.elements_per_thread), stride=(self.elements_per_thread, 1)
        )

        tiled_copy_x = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), x.element_type),
            thr_layout=thr_layout,
            val_layout=val_layout,
        )
        threads_per_scale = 128 // self.elements_per_thread
        num_scale_writers = self.threads_per_row // threads_per_scale
        tiled_copy_s = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), scale.element_type),
            thr_layout=cute.make_layout(
                (1, num_scale_writers),
                stride=(num_scale_writers, 1),
            ),
            val_layout=cute.make_layout((1, 1)),
        )
        tiled_copy_a = None
        if const_expr(self.load_path == "shared"):
            copy_bits = min(128, self.elements_per_thread * x.element_type.width)
            tiled_copy_a = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(),
                    x.element_type,
                    num_bits_per_copy=copy_bits,
                ),
                thr_layout=thr_layout,
                val_layout=val_layout,
            )

        self.rms_norm_kernel(
            x,
            residual,
            weight,
            eps,
            out,
            scale,
            residual_out,
            normalized_out,
            tiled_copy_x,
            tiled_copy_a,
            tiled_copy_s,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def rms_norm_kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        eps: cute.Float32,
        mO: cute.Tensor,
        mSO: cute.Tensor,
        mRO: cute.Tensor,
        mNO: cute.Tensor,
        tiled_copy_x: cute.TiledCopy,
        tiled_copy_a: cute.TiledCopy | None,
        tiled_copy_sc: cute.TiledCopy,
    ) -> None:
        salloc = SmemAllocator()
        threads_per_scale = 128 // self.elements_per_thread

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        num_warps = self.threads_per_row // cute.arch.WARP_SIZE
        scale_writer_idx = tidx // threads_per_scale

        tiler_mn = (1, mX.shape[1])
        tiler_mn_scales = (1, mSO.shape[1])

        # block input tiles
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gR = cute.local_tile(mR, tiler_mn, (bidx, 0))
        gW = cute.make_tensor(
            mW.iterator, cute.make_layout((1, mW.shape[0]), stride=(0, mW.stride[0]))
        )

        #  block output tiles
        gNO = cute.local_tile(mNO, tiler_mn, (bidx, 0))
        gRO = cute.local_tile(mRO, tiler_mn, (bidx, 0))
        gO = cute.local_tile(mO, tiler_mn, (bidx, 0))
        gSO = cute.local_tile(mSO, tiler_mn_scales, (bidx, 0))

        # thread local copies
        thr_copy_x = tiled_copy_x.get_slice(tidx)
        thr_copy_sc = tiled_copy_sc.get_slice(scale_writer_idx)

        # partition the global tiles into thread local for X compute path
        tXgX = thr_copy_x.partition_S(gX)
        tXgR = thr_copy_x.partition_S(gR)
        tXgW = thr_copy_x.partition_S(gW)
        tXgRO = thr_copy_x.partition_D(gRO)
        tXgNO = thr_copy_x.partition_D(gNO)
        tXgO = thr_copy_x.partition_D(gO)
        
        tSgSO = thr_copy_sc.partition_D(gSO)

        # Register fragments
        rX = cute.make_rmem_tensor_like(tXgX)
        rR = cute.make_rmem_tensor_like(tXgR)
        rW = cute.make_rmem_tensor_like(tXgW)
        rNO = cute.make_rmem_tensor_like(tXgNO)
        rRO = cute.make_rmem_tensor_like(tXgRO)
        rO = cute.make_rmem_tensor_like(tXgO)

        thread_sum = cute.Float32(0.0)

        if const_expr(self.load_path == "shared"):
            # Async global → shared staging overlaps with the weight load.
            # The async atom has its own value grouping; use its views for G2S.
            sX = salloc.allocate_tensor(
                mX.element_type,
                cute.make_layout((1, mX.shape[1]), stride=(mX.shape[1], 1)),
                byte_alignment=16,
            )
            sR = salloc.allocate_tensor(
                mR.element_type,
                cute.make_layout((1, mR.shape[1]), stride=(mR.shape[1], 1)),
                byte_alignment=16,
            )

            thr_copy_a = tiled_copy_a.get_slice(tidx)
            tAgX = thr_copy_a.partition_S(gX)
            tAgR = thr_copy_a.partition_S(gR)

            tAsX = thr_copy_a.partition_D(sX)
            tAsR = thr_copy_a.partition_D(sR)

            cute.copy(tiled_copy_a, tAgX, tAsX)
            cute.copy(tiled_copy_a, tAgR, tAsR)

            cute.arch.cp_async_commit_group()
            cute.copy(tiled_copy_x, tXgW, rW)
            cute.arch.cp_async_wait_group(0)

            # Read shared memory with the compute grouping used by rX/rW/rO.
            # Each thread reads its own staged values, so no block barrier is needed.
            tXsX = thr_copy_x.partition_S(sX)
            tXsR = thr_copy_x.partition_S(sR)

            cute.autovec_copy(tXsX, rX)
            cute.autovec_copy(tXsR, rR)
        else:
            # Direct global → registers: no shared input row or async-copy wait.
            # Load weights early so independent memory operations can overlap.
            cute.copy(tiled_copy_x, tXgX, rX)
            cute.copy(tiled_copy_x, tXgR, rR)
            cute.copy(tiled_copy_x, tXgW, rW)

        # Both paths retain all thread-owned X values for the output calculation.
        x_chunk = rX.load()
        r_chunk = rR.load()

        residual = x_chunk + r_chunk
        rRO.store(residual.to(rRO.element_type))
        cute.copy(tiled_copy_x, rRO, tXgRO)

        residual = residual.to(cute.Float32)
        thread_sum += (residual * residual).reduce(
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

        normalized = (residual * inv_rms).to(rNO.element_type)
        normalized = normalized * rW.load()

        rNO.store(normalized)
        cute.copy(tiled_copy_x, rNO, tXgNO)

        thread_max = cute.math.abs(normalized.to(cute.Float32)).reduce(
            cute.ReductionOp.MAX,
            init_val=cute.Float32(0.0),
            reduction_profile=(1, None, None),
        )
        # Each shuffle handles one FP32 value. Unroll across independent chunks;
        # the reduction preserves their grouping instead of mixing them together.
        group_max = cute.make_rmem_tensor(thread_max.shape, cute.Float32)
        for group in range_constexpr(cute.size(thread_max.shape)):
            group_max[group] = cute.arch.warp_reduction_max(
                thread_max[group],
                threads_in_group=128 // self.elements_per_thread,
            )
        scale = group_max.load() / cute.Float32(448.0)
        scale = cute.math.max(scale, cute.full_like(scale, 1e-4))

        rS = cute.make_rmem_tensor_like(scale)
        rS.store(scale)

        tSrS = cute.make_tensor(rS.iterator, cute.make_layout(tSgSO.shape))
        if tidx % threads_per_scale == 0:
            cute.copy(tiled_copy_sc, tSrS, tSgSO)

        scale_broadcast = cute.make_tensor(
            rS.iterator,
            cute.make_layout(
                rNO.shape,
                stride=((0, 0), rS.stride[0], rS.stride[1]),
            ),
        )
        quantized = (rNO.load().to(cute.Float32) / scale_broadcast.load()).to(
            cute.Float8E4M3FN
        )
        rO.store(quantized)
        cute.copy(tiled_copy_x, rO, tXgO)


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
