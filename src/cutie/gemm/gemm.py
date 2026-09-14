import os

import cuda.bindings.driver as cuda_driver
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
    def __init__(self, BM: int, BN: int, BK: int, num_smem_stages: int = 1):
        self.BM = BM
        self.BN = BN
        self.BK = BK
        self.num_smem_stages = num_smem_stages

        self._elements_per_copy = 8

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

        threads_per_k = self.BK // self._elements_per_copy

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            A.element_type,
            num_bits_per_copy=self._elements_per_copy * A.element_type.width,
        )

        tiled_copy = cute.make_tiled_copy_tv(
            copy_atom,
            thr_layout=cute.make_layout(
                (128 // threads_per_k, threads_per_k), stride=(threads_per_k, 1)
            ),
            val_layout=cute.make_layout(
                (1, self._elements_per_copy), stride=(self._elements_per_copy, 1)
            ),
        )

        block = (128, 1, 1)
        grid = (M // self.BM, N // self.BN, 1)

        self.gemm_kernel(A, B, out, tiled_mma, tiled_copy).launch(
            grid=grid, block=block, stream=stream
        )

    @cute.kernel
    def gemm_kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tiled_copy: cute.TiledCopy,
    ):
        salloc = SmemAllocator()

        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

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
        layout_A = cute.tile_to_shape(
            smem_atom, (self.BM, self.BK, self.num_smem_stages), order=(1, 0, 2)
        )
        layout_B = cute.tile_to_shape(
            smem_atom, (self.BN, self.BK, self.num_smem_stages), order=(1, 0, 2)
        )

        # allocate with outer/inner
        sA = salloc.allocate_tensor(
            mA.element_type,
            layout_A.outer,
            byte_alignment=1024,
            swizzle=layout_A.inner,
        )
        sB = salloc.allocate_tensor(
            mB.element_type,
            layout_B.outer,
            byte_alignment=1024,
            swizzle=layout_B.inner,
        )
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

        thr_copy = tiled_copy.get_slice(tidx)

        tAsA = thr_copy.partition_D(sA)
        tBsB = thr_copy.partition_D(sB)
        _print(f"tAsA: {tAsA}")
        _print(f"tBsB: {tBsB}")

        gA = cute.local_tile(mA, (self.BM, self.BK), (bidx, None))
        gB = cute.local_tile(mB, (self.BN, self.BK), (bidy, None))
        tAgA = thr_copy.partition_S(gA)
        tBgB = thr_copy.partition_S(gB)

        # prefetch N-1
        # this will prob break on small K but who cares
        for load_k_tile in range_constexpr(self.num_smem_stages - 1):
            cute.copy(
                tiled_copy,
                tAgA[None, None, None, load_k_tile],
                tAsA[None, None, None, load_k_tile % self.num_smem_stages],
            )
            cute.copy(
                tiled_copy,
                tBgB[None, None, None, load_k_tile],
                tBsB[None, None, None, load_k_tile % self.num_smem_stages],
            )
            cute.arch.cp_async_commit_group()

        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        mma_done = salloc.allocate(cute.Int64, byte_alignment=8)
        phase = 0
        if tidx == 0:
            cute.arch.mbarrier_init(mma_done, 1)

        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        num_k_tiles = cute.size(mA, mode=[1]) // self.BK
        for k_tile in range(num_k_tiles):
            consume_k_tile = k_tile
            load_k_tile = k_tile + self.num_smem_stages - 1
            consume_smem_stage = consume_k_tile % self.num_smem_stages
            load_smem_stage = load_k_tile % self.num_smem_stages
            # prefetch Nth
            if k_tile < num_k_tiles - self.num_smem_stages + 1:
                cute.copy(
                    tiled_copy,
                    tAgA[None, None, None, load_k_tile],
                    tAsA[None, None, None, load_smem_stage],
                )
                cute.copy(
                    tiled_copy,
                    tBgB[None, None, None, load_k_tile],
                    tBsB[None, None, None, load_smem_stage],
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(self.num_smem_stages - 1)
            else:
                cute.arch.cp_async_wait_group(0)

            # signal mma that the data is ready
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()

            if cute.arch.warp_idx() == 0:
                for k_atom in range_constexpr(cute.size(tCrA, mode=[2])):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[None, None, k_atom, consume_smem_stage],
                        tCrB[None, None, k_atom, consume_smem_stage],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                with cute.arch.elect_one():
                    tcgen05.commit(mma_done)

            # wait for MMA to finish before we start the next iteration
            cute.arch.mbarrier_wait(mma_done, phase)
            phase ^= 1
            cute.arch.sync_threads()

        fence_after_thread_sync()
        # tmem to reg
        cute.copy(tmem_copy, tDtC, rAcc)
        # signal that the data is fully gone from tmem
        cute.arch.fence_view_async_tmem_load()
        rC.store(rAcc.load().to(mC.element_type))
        cute.autovec_copy(rC, tDgC)

        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)
