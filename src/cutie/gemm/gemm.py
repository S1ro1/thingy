import os

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import const_expr, cute, dsl_user_op, pipeline, range_constexpr, utils
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils import SmemAllocator


@dsl_user_op
def fence_after_thread_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(
        nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC,
        loc=loc,
        ip=ip,
    )


def _print(*args, **kwargs):
    if os.environ.get("DEBUG", "0") == "1":
        print(*args, **kwargs)


class Gemm:
    def __init__(
        self,
        BM: int,
        BN: int,
        BK: int,
        num_smem_stages: int = 1,
        scheduling: str = "rowwise",
        super_m: int = 1,
    ):
        self.BM = BM
        self.BN = BN
        self.BK = BK
        self.num_smem_stages = num_smem_stages
        self.scheduling = scheduling
        self.super_m = super_m

        if super_m != 1 and scheduling != "super_m":
            raise ValueError(
                f"Invalid scheduling: {scheduling}, must be 'super_m' when super_m is not 1"
            )

        assert self.scheduling in ("rowwise", "super_m"), (
            f"Invalid scheduling: {self.scheduling}, must be one of ('rowwise', 'super_m')"
        )

        self._elements_per_copy = 8

    @cute.jit
    def rowwise_scheduling(
        self, tile_idx: cute.Int32, num_m_tiles: cute.Int32, num_n_tiles: cute.Int32
    ) -> tuple[cute.Int32, cute.Int32]:
        bidx = tile_idx // num_n_tiles
        bidy = tile_idx % num_n_tiles
        return bidx, bidy

    @cute.jit
    def super_m_scheduling(
        self, tile_idx: cute.Int32, num_m_tiles: cute.Int32, num_n_tiles: cute.Int32
    ) -> tuple[cute.Int32, cute.Int32]:
        tiles_per_band = self.super_m * num_n_tiles

        band_idx = tile_idx // tiles_per_band
        tile_in_band = tile_idx % tiles_per_band

        first_m = band_idx * self.super_m
        band_height = cutlass.min(num_m_tiles - first_m, self.super_m)

        m_in_band = tile_in_band % band_height
        n_in_band = tile_in_band // band_height

        bidx = first_m + m_in_band
        bidy = n_in_band

        return bidx, bidy

    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        out: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        M, _ = A.shape
        N, _ = B.shape

        mma_op = tcgen05.MmaF16BF16Op(
            ab_dtype=cute.BFloat16,
            acc_dtype=cute.Float32,
            instruction_shape=(self.BM, self.BN, 16),
            cta_group=tcgen05.CtaGroup.ONE,
            a_src=tcgen05.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.K,
            b_major_mode=cute.nvgpu.OperandMajorMode.K,
        )

        tiled_mma = cute.make_tiled_mma(
            mma_op, permutation_mnk=(self.BM, self.BN, self.BK)
        )
        smem_atom_kind = sm100_utils.get_smem_layout_atom_ab(
            cute.nvgpu.OperandMajorMode.K,
            cute.BFloat16,
            (self.BM, self.BK),
        )
        _print(f"smem_atom_kind: {smem_atom_kind}")
        smem_atom = tcgen05.make_smem_layout_atom(smem_atom_kind, cute.BFloat16)

        _print(f"smem_atom: {smem_atom}")

        # ((rows_in_band, num_bands), (K_elements_per_swizzle_atom, num_K_tiles))
        # - with BK=128, get_smem_layout_atom_ab will choose K_SW128, where K atom is 128 bytes (64 elements wide), so for us we have 2 K tiles per swizzle atom - ((..., ...), (64, 2))
        # - with BK=64, we still get K_SW128, but now we have 1 K tile per swizzle atom - ((..., ...), (64, 1))
        # - with BK=32, we get K_SW64, where K atom is 64 bytes (32 elements wide), so we have 1 K tile per swizzle atom - ((..., ...), (32, 1))
        smem_layout_A = cute.tile_to_shape(
            smem_atom, (self.BM, self.BK, self.num_smem_stages), order=(1, 0, 2)
        )
        smem_layout_B = cute.tile_to_shape(
            smem_atom, (self.BN, self.BK, self.num_smem_stages), order=(1, 0, 2)
        )

        tma_info_A = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            A,
            cute.select(smem_layout_A, mode=[0, 1]),
            (self.BM, self.BK),
        )
        tma_info_B = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            B,
            cute.select(smem_layout_B, mode=[0, 1]),
            (self.BN, self.BK),
        )

        num_ctas = utils.HardwareInfo(0).get_device_multiprocessor_count()
        block = (128, 1, 1)
        grid = (num_ctas, 1, 1)
        self.gemm_kernel(
            A, B, out, tma_info_A, tma_info_B, smem_layout_A, smem_layout_B, tiled_mma
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def gemm_kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        tma_info_A: cute.nvgpu.cpasync.TmaInfo,
        tma_info_B: cute.nvgpu.cpasync.TmaInfo,
        # we need to pass these as tma_info.smem_layout loses the notion of stages
        smem_layout_A: cute.ComposedLayout,
        smem_layout_B: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
    ):
        salloc = SmemAllocator()

        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        num_ctas, _, _ = cute.arch.grid_dim()

        thr_mma = tiled_mma.get_slice(0)
        acc_shape = tiled_mma.partition_shape_C((self.BM, self.BN))
        acc_fragment = tiled_mma.make_fragment_C(acc_shape)

        num_tmem_cols = utils.get_num_tmem_alloc_cols(acc_fragment)
        _print(f"num_tmem_cols: {num_tmem_cols}")  # as far as I knw this is also BN
        tmem = utils.TmemAllocator(
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=128)
        )
        tmem.allocate(num_columns=num_tmem_cols)
        tmem.wait_for_alloc()

        tmem_ptr = tmem.retrieve_ptr(cute.Float32)
        tCtAcc = cute.make_tensor(tmem_ptr, acc_fragment.layout)

        tmem_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32),
            cute.Float32,
        )
        tmem_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc)
        thr_tmem = tmem_copy.get_slice(tidx)

        # allocate with outer/inner
        sA = salloc.allocate_tensor(
            mA.element_type,
            smem_layout_A.outer,
            byte_alignment=1024,
            swizzle=smem_layout_A.inner,
        )
        sB = salloc.allocate_tensor(
            mB.element_type,
            smem_layout_B.outer,
            byte_alignment=1024,
            swizzle=smem_layout_B.inner,
        )
        mma_done = salloc.allocate_tensor(
            cute.Int64, cute.make_layout(self.num_smem_stages), byte_alignment=8
        ).iterator
        load_done = salloc.allocate_tensor(
            cute.Int64, cute.make_layout(self.num_smem_stages), byte_alignment=8
        ).iterator
        expected_bytes = ((self.BM + self.BN) * self.BK * mA.element_type.width) // 8
        if tidx == 0:
            for stage in range_constexpr(self.num_smem_stages):
                cute.arch.mbarrier_init(mma_done + stage, 1)
                cute.arch.mbarrier_init(load_done + stage, 1)

        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        _print(f"sA: {sA}")
        _print(f"sB: {sB}")

        # (MMA, MMA_M, MMA_K)
        tCsA = thr_mma.partition_A(sA)
        # (MMA, MMA_N, MMA_K)
        tCsB = thr_mma.partition_B(sB)
        _print(f"tCsA: {tCsA}")
        _print(f"tCsB: {tCsB}")

        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        _print(f"tCrA: {tCrA}")
        _print(f"tCrB: {tCrB}")

        tile_idx = bidx
        total_output_tiles = (cute.size(mC, mode=[0]) // self.BM) * (
            cute.size(mC, mode=[1]) // self.BN
        )

        num_m_tiles = cute.size(mC, mode=[0]) // self.BM
        num_n_tiles = cute.size(mC, mode=[1]) // self.BN
        num_k_tiles = cute.size(mA, mode=[1]) // self.BK
        # this signals the base for current tile
        pipeline_base = 0
        bidx, bidy = 0, 0

        while tile_idx < total_output_tiles:
            if const_expr(self.scheduling == "rowwise"):
                bidx, bidy = self.rowwise_scheduling(tile_idx, num_m_tiles, num_n_tiles)
            elif const_expr(self.scheduling == "super_m"):
                bidx, bidy = self.super_m_scheduling(tile_idx, num_m_tiles, num_n_tiles)

            gA = cute.local_tile(
                tma_info_A.tma_tensor, (self.BM, self.BK), (bidx, None)
            )
            gB = cute.local_tile(
                tma_info_B.tma_tensor, (self.BN, self.BK), (bidy, None)
            )
            _print(f"gA: {gA}")
            _print(f"sA: {sA}")

            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                tma_info_A.atom,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA, 0, 2),
            )
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                tma_info_B.atom,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB, 0, 2),
            )
            gC = cute.local_tile(mC, (self.BM, self.BN), (bidx, bidy))

            # this threads partition of C in global mem
            tCgC = thr_mma.partition_C(gC)

            # this threads partition of the TMEM
            tDtC = thr_tmem.partition_S(tCtAcc)
            # this threads partition of C in global mem
            tDgC = thr_tmem.partition_D(tCgC)

            # accum dtype fp32, we copy from tmem to registers
            rAcc = cute.make_rmem_tensor(tDgC.shape, cute.Float32)
            # output dtype bf16, we copy from reg to reg
            rC = cute.make_rmem_tensor(tDgC.shape, mC.element_type)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

            for k_tile in range(num_k_tiles):
                pipeline_tile = pipeline_base + k_tile
                smem_stage = pipeline_tile % self.num_smem_stages
                phase = (pipeline_tile // self.num_smem_stages) % 2
                if cute.arch.warp_idx() == 0:
                    if pipeline_tile >= self.num_smem_stages:
                        previous_tile = pipeline_tile - self.num_smem_stages
                        previous_phase = (previous_tile // self.num_smem_stages) % 2
                        cute.arch.mbarrier_wait(
                            mma_done + smem_stage,
                            previous_phase,
                        )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            load_done + smem_stage, expected_bytes
                        )
                    cute.copy(
                        tma_info_A.atom,
                        tAgA[None, k_tile],
                        tAsA[None, smem_stage],
                        tma_bar_ptr=load_done + smem_stage,
                    )
                    cute.copy(
                        tma_info_B.atom,
                        tBgB[None, k_tile],
                        tBsB[None, smem_stage],
                        tma_bar_ptr=load_done + smem_stage,
                    )
                elif cute.arch.warp_idx() == 1:
                    cute.arch.mbarrier_wait(load_done + smem_stage, phase)
                    for k_atom in range_constexpr(cute.size(tCrA, mode=[2])):
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[None, None, k_atom, smem_stage],
                            tCrB[None, None, k_atom, smem_stage],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    with cute.arch.elect_one():
                        tcgen05.commit(mma_done + smem_stage)

            # wait for non-tma/mma warps
            cute.arch.sync_threads()
            last_tile = pipeline_base + num_k_tiles - 1
            last_stage = (num_k_tiles - 1) % self.num_smem_stages
            last_phase = (last_tile // self.num_smem_stages) % 2
            cute.arch.mbarrier_wait(mma_done + last_stage, last_phase)

            fence_after_thread_sync()
            # tmem to reg
            cute.copy(tmem_copy, tDtC, rAcc)
            # signal that the data is fully gone from tmem
            cute.arch.fence_view_async_tmem_load()
            rC.store(rAcc.load().to(mC.element_type))
            cute.autovec_copy(rC, tDgC)

            cute.arch.sync_threads()
            pipeline_base += num_k_tiles
            tile_idx += num_ctas

        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)
