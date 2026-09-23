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
        num_acc_stages: int = 1,
        num_m_ctas: int = 1,
    ):
        self.BM = BM
        self.BN = BN
        self.BK = BK
        self.num_smem_stages = num_smem_stages
        self.scheduling = scheduling
        self.super_m = super_m
        self.num_acc_stages = num_acc_stages
        self.num_m_ctas = num_m_ctas

        if super_m != 1 and scheduling != "super_m":
            raise ValueError(
                f"Invalid scheduling: {scheduling}, must be 'super_m' when super_m is not 1"
            )

        assert self.scheduling in ("rowwise", "super_m"), (
            f"Invalid scheduling: {self.scheduling}, must be one of ('rowwise', 'super_m')"
        )

        assert self.num_m_ctas in (1, 2), (
            f"Invalid num_m_ctas: {self.num_m_ctas}, only 1 or 2 is supported"
        )

        if self.num_m_ctas == 2 and self.BM not in (128, 256):
            raise ValueError("With num_m_ctas=2, BM must be either 128, or 256.")

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
        cta_group_enum = tcgen05.CtaGroup.ONE
        if const_expr(self.num_m_ctas == 2):
            cta_group_enum = tcgen05.CtaGroup.TWO

        mma_op = tcgen05.MmaF16BF16Op(
            ab_dtype=cute.BFloat16,
            acc_dtype=cute.Float32,
            instruction_shape=(self.BM, self.BN, 16),
            cta_group=cta_group_enum,
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
        # smem_layout_A = cute.tile_to_shape(
        #     smem_atom, (self.BM, self.BK, self.num_smem_stages), order=(1, 0, 2)
        # )
        # smem_layout_B = cute.tile_to_shape(
        #     smem_atom, (self.BN, self.BK, self.num_smem_stages), order=(1, 0, 2)
        # )
        #
        # with make_smem_layout_a the shapes are already grouped
        # ((ATOM_M, ATOM_K), MMA_M, MMA_K, num_stages)
        smem_layout_A = sm100_utils.make_smem_layout_a(
            tiled_mma, (self.BM, self.BN, self.BK), A.element_type, self.num_smem_stages
        )
        # ((ATOM_N, ATOM_K), MMA_N, MMA_K, num_stages)
        smem_layout_B = sm100_utils.make_smem_layout_b(
            tiled_mma, (self.BM, self.BN, self.BK), B.element_type, self.num_smem_stages
        )

        def _group_modes(_layout: cute.ComposedLayout) -> cute.ComposedLayout:
            _layout = cute.select(_layout, mode=[0, 1, 2])

            outer = cute.make_layout(
                (
                    (_layout.outer.shape[0][0], _layout.outer.shape[1]),
                    (_layout.outer.shape[0][1], _layout.outer.shape[2]),
                ),
                stride=(
                    (_layout.outer.stride[0][0], _layout.outer.stride[1]),
                    (_layout.outer.stride[0][1], _layout.outer.stride[2]),
                ),
            )
            layout = cute.make_composed_layout(_layout.inner, _layout.offset, outer)
            return layout

        mk_layout = _group_modes(smem_layout_A)
        nk_layout = _group_modes(smem_layout_B)

        tma_info_A = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group=cta_group_enum),
            A,
            mk_layout,
            (self.BM // self.num_m_ctas, self.BK),
        )
        tma_info_B = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group=cta_group_enum),
            B,
            nk_layout,
            (self.BN // self.num_m_ctas, self.BK),
        )

        num_ctas = utils.HardwareInfo(0).get_device_multiprocessor_count()
        block = (192, 1, 1)
        grid = (num_ctas, 1, 1)
        cluster = (self.num_m_ctas, 1, 1)
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.tma_warp_id = 4
        self.mma_warp_id = 5
        self.leader_cta_id = 0
        self.gemm_kernel(
            A, B, out, tma_info_A, tma_info_B, smem_layout_A, smem_layout_B, tiled_mma
        ).launch(grid=grid, block=block, stream=stream, cluster=cluster)

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
        cluster_cta_idx = cute.arch.block_idx_in_cluster()

        cluster_mask = None
        cta_group = tcgen05.CtaGroup.ONE
        if const_expr(self.num_m_ctas == 2):
            cluster_mask = 0b11
            cta_group = tcgen05.CtaGroup.TWO

        thr_mma = tiled_mma.get_slice(cluster_cta_idx)
        acc_shape = tiled_mma.partition_shape_C((self.BM, self.BN))
        acc_fragment = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stages)
        )

        num_tmem_cols = utils.get_num_tmem_alloc_cols(acc_fragment)
        _print(f"num_tmem_cols: {num_tmem_cols}")  # as far as I knw this is also BN
        tmem = utils.TmemAllocator(
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=192),
            is_two_cta=const_expr(self.num_m_ctas == 2),
        )
        tmem.allocate(num_columns=num_tmem_cols)
        tmem.wait_for_alloc()

        tmem_ptr = tmem.retrieve_ptr(cute.Float32)
        tCtAcc_base = cute.make_tensor(tmem_ptr, acc_fragment.layout)

        tmem_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32),
            cute.Float32,
        )
        tmem_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_base[None, None, None, 0])
        thr_tmem = tmem_copy.get_slice(tidx)

        # allocate with outer/inner
        tCsA = salloc.allocate_tensor(
            mA.element_type,
            smem_layout_A.outer,
            byte_alignment=1024,
            swizzle=smem_layout_A.inner,
        )
        tCsB = salloc.allocate_tensor(
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
        acc_empty = salloc.allocate_tensor(
            cute.Int64, cute.make_layout(self.num_acc_stages), byte_alignment=8
        ).iterator
        acc_ready = salloc.allocate_tensor(
            cute.Int64, cute.make_layout(self.num_acc_stages), byte_alignment=8
        ).iterator
        expected_bytes = ((self.BM + self.BN) * self.BK * mA.element_type.width) // 8
        if tidx == 0:
            for stage in range_constexpr(self.num_smem_stages):
                cute.arch.mbarrier_init(mma_done + stage, 1)
                cute.arch.mbarrier_init(load_done + stage, 1)

            for stage in range_constexpr(self.num_acc_stages):
                cute.arch.mbarrier_init(acc_empty + stage, 4 * self.num_m_ctas)
                cute.arch.mbarrier_init(acc_ready + stage, 1)

        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        _print(f"tCrA: {tCrA}")
        _print(f"tCrB: {tCrB}")

        tile_idx = bidx // self.num_m_ctas
        num_clusters = num_ctas // self.num_m_ctas
        total_output_tiles = (cute.size(mC, mode=[0]) // self.BM) * (
            cute.size(mC, mode=[1]) // self.BN
        )

        num_m_tiles = cute.size(mC, mode=[0]) // self.BM
        num_n_tiles = cute.size(mC, mode=[1]) // self.BN
        num_k_tiles = cute.size(mA, mode=[1]) // self.BK
        # this signals the base for current tile
        pipeline_base = 0
        bidx, bidy = 0, 0
        acc_stage = 0
        t = 0

        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        while tile_idx < total_output_tiles:
            acc_stage = t % self.num_acc_stages
            acc_phase = (t // self.num_acc_stages) % 2

            tCtAcc = tCtAcc_base[None, None, None, acc_stage]
            if const_expr(self.scheduling == "rowwise"):
                bidx, bidy = self.rowwise_scheduling(tile_idx, num_m_tiles, num_n_tiles)
            elif const_expr(self.scheduling == "super_m"):
                bidx, bidy = self.super_m_scheduling(tile_idx, num_m_tiles, num_n_tiles)

            # we get the tiles across the K schedule, we can just index into it
            # (BM, BK, num_k_tiles)
            gA = cute.local_tile(
                tma_info_A.tma_tensor, (self.BM, self.BK), (bidx, None)
            )
            # ((M_atom, K_atom), MMA_M, MMA_K, num_k_tiles)
            tCgA = thr_mma.partition_A(gA)
            # (BN, BK, num_k_tiles)
            gB = cute.local_tile(
                tma_info_B.tma_tensor, (self.BN, self.BK), (bidy, None)
            )
            # ((N_atom, K_atom), MMA_N, MMA_K, num_k_tiles)
            tCgB = thr_mma.partition_B(gB)

            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                tma_info_A.atom,
                0,
                cute.make_layout(1),
                cute.group_modes(tCsA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                tma_info_B.atom,
                0,
                cute.make_layout(1),
                cute.group_modes(tCsB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )
            gC = cute.local_tile(mC, (self.BM, self.BN), (bidx, bidy))

            # 1. we partition with mma gC for the portion that this mma will compute
            tCgC = thr_mma.partition_C(gC)
            # 2. we partition with thr_tmem the tmem accumulator
            tDtC = thr_tmem.partition_S(tCtAcc)
            # 3. we FURTHER partition the local accumulator for epilogue copies
            tDgC = thr_tmem.partition_D(tCgC)

            # accum dtype fp32, we copy from tmem to registers
            rAcc = cute.make_rmem_tensor(tDgC.shape, cute.Float32)
            # output dtype bf16, we copy from reg to reg
            rC = cute.make_rmem_tensor(tDgC.shape, mC.element_type)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

            # wait for the current epilogue tmem to be empty
            if (
                cluster_cta_idx == self.leader_cta_id
                and cute.arch.warp_idx() == self.mma_warp_id
                and t >= self.num_acc_stages
            ):
                previous_acc_phase = (
                    (t - self.num_acc_stages) // self.num_acc_stages
                ) % 2
                cute.arch.mbarrier_wait(acc_empty + acc_stage, previous_acc_phase)

            for k_tile in range(num_k_tiles):
                pipeline_tile = pipeline_base + k_tile
                smem_stage = pipeline_tile % self.num_smem_stages
                phase = (pipeline_tile // self.num_smem_stages) % 2
                leader_load_done = cute.arch.map_dsmem_ptr(
                    load_done + smem_stage, cta_rank_in_cluster=self.leader_cta_id
                )
                if cute.arch.warp_idx() == self.tma_warp_id:
                    if pipeline_tile >= self.num_smem_stages:
                        # if we are after num_smem_stages in tiles, we need to wait for the previous mma that was consuming this stage to finish
                        previous_tile = pipeline_tile - self.num_smem_stages
                        previous_phase = (previous_tile // self.num_smem_stages) % 2
                        cute.arch.mbarrier_wait(
                            mma_done + smem_stage,
                            previous_phase,
                        )
                    # we issue the loads as one warp
                    if cluster_cta_idx == self.leader_cta_id:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                load_done + smem_stage, expected_bytes
                            )
                    cute.copy(
                        tma_info_A.atom,
                        tAgA[None, k_tile],
                        tAsA[None, smem_stage],
                        tma_bar_ptr=leader_load_done,
                    )
                    cute.copy(
                        tma_info_B.atom,
                        tBgB[None, k_tile],
                        tBsB[None, smem_stage],
                        tma_bar_ptr=leader_load_done,
                    )
                if (
                    cute.arch.warp_idx() == self.mma_warp_id
                    and cluster_cta_idx == self.leader_cta_id
                ):
                    # we wait on our shared memory
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
                        tcgen05.commit(mma_done + smem_stage, cluster_mask, cta_group)

            # after all K tiles we commit all our mmas to signal for the accumulator
            if (
                cute.arch.warp_idx() == self.mma_warp_id
                and cluster_cta_idx == self.leader_cta_id
            ):
                with cute.arch.elect_one():
                    tcgen05.commit(
                        acc_ready + acc_stage, mask=cluster_mask, cta_group=cta_group
                    )

            # tmem to reg
            # this happens after full num_k_tiles iterations happens, epilogue warps just spin here forever
            if cute.arch.warp_idx() in self.epilogue_warp_id:
                # wait for buffer to be ready
                cute.arch.mbarrier_wait(acc_ready + acc_stage, acc_phase)

                acc_empty_leader = cute.arch.map_dsmem_ptr(acc_empty + acc_stage, 0)
                # order following tmems after that synchronization above
                fence_after_thread_sync()
                cute.copy(tmem_copy, tDtC, rAcc)
                # signal that the data is fully gone from tmem and signals registers are ready
                cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_warp()
                # signal to mma warps that the accumulator is already empty
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(acc_empty_leader)

                # cast to bf16 in registers
                rC.store(rAcc.load().to(mC.element_type))
                # copy to destination
                cute.autovec_copy(rC, tDgC)

            pipeline_base += num_k_tiles
            tile_idx += num_clusters
            t += 1

        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)
