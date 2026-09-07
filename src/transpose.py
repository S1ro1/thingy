import torch

from argparse import ArgumentParser

import cutlass.cute as cute

from cutlass.utils import SmemAllocator
from cutlass.cute.runtime import from_dlpack

ELEMENTS_PER_COPY = 8
WARP_SIZE = 32

class Transpose:
    def __init__(self, M: int, N: int, BM: int = 32, BN: int = 32):
        self.M = M
        self.N = N
        self.BM = BM
        self.BN = BN
    
    @cute.jit
    def __call__(self, S: cute.Tensor, D: cute.Tensor) -> None:
        val_layout = cute.make_layout((1, ELEMENTS_PER_COPY), stride=(ELEMENTS_PER_COPY, 1))
        thr_layout = cute.make_layout((self.BM, self.BN // ELEMENTS_PER_COPY), stride=(self.BN // ELEMENTS_PER_COPY, 1))
        copy_op = cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(copy_op, cute.Float32, num_bits_per_copy=256)
        load_tiled_copy: cute.TiledCopy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        print(f"load_tiled_copy: {load_tiled_copy}")

        block_dim = (cute.size(load_tiled_copy.layout_tv_tiled, mode=[0]), 1, 1)
        grid_dim = ((self.M // self.BM), (self.N // self.BN), 1)

        print(f"block_dim: {block_dim}")
        print(f"grid_dim: {grid_dim}")

        tiled_s = cute.zipped_divide(S, load_tiled_copy.tiler_mn)

        self._kernel(tiled_s, D, load_tiled_copy).launch(block=block_dim, grid=grid_dim)

    @cute.kernel
    def _kernel(self, S: cute.Tensor, D: cute.Tensor, tiled_copy: cute.TiledCopy) -> None:
        salloc = SmemAllocator()

        tidx, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, _ = cute.arch.block_idx()

        tile_s = cute.flatten(S[None, (bid_m, bid_n)])
        tile_d = cute.local_tile(D, (self.BN, self.BM), (bid_n, bid_m))

        smem_layout_base = cute.make_layout((self.BM, self.BN), stride=(self.BN, 1))
        swizzle = cute.make_swizzle(2, 3, 2)
        smem_layout = cute.make_composed_layout(swizzle, 0, smem_layout_base)

        print(f"S: {S}")
        print(f"D: {D}")
        print(f"tile_s: {tile_s}")
        print(f"tile_d: {tile_d}")

        smem_tile_s = salloc.allocate_tensor(cute.Float32, smem_layout, byte_alignment=32)

        thr_load = tiled_copy.get_slice(tidx)

        tSgS = thr_load.partition_S(tile_s)
        tSsS = thr_load.partition_D(smem_tile_s)

        cute.copy(tiled_copy, tSgS, tSsS)
        cute.arch.sync_threads()

        # create a transposed layout, we copy a full row of smem into a column of gmem
        tile_smem_layout_t = cute.composition(
            smem_layout,
            cute.make_layout(
                (self.BN, self.BM),
                stride=(self.BM, 1),
            )
        )
        smem_tile_s_t = cute.make_tensor(smem_tile_s.iterator, tile_smem_layout_t)
        print(f"smem_layout: {smem_layout}")
        print(f"tile_smem_layout_t: {tile_smem_layout_t}")
        print(f"smem_tile_s_t: {smem_tile_s_t}")

        store_val_layout = cute.make_layout((1, ELEMENTS_PER_COPY), stride=(ELEMENTS_PER_COPY, 1))
        store_thr_layout = cute.make_layout((self.BN, self.BM // ELEMENTS_PER_COPY), stride=(self.BM // ELEMENTS_PER_COPY, 1))

        smem_load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cute.Float32,
            num_bits_per_copy=32
        )
        gmem_store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cute.Float32,
            num_bits_per_copy=256,
        )

        smem_tiled_load: cute.TiledCopy = cute.make_tiled_copy_tv(smem_load_atom, store_thr_layout, store_val_layout)
        gmem_tiled_store: cute.TiledCopy = cute.make_tiled_copy_tv(gmem_store_atom, store_thr_layout, store_val_layout)

        thr_smem_load = smem_tiled_load.get_slice(tidx)
        thr_gmem_store = gmem_tiled_store.get_slice(tidx)

        tSsD = thr_smem_load.partition_S(smem_tile_s_t)
        tDgD = thr_gmem_store.partition_D(tile_d)
        rD = cute.make_rmem_tensor_like(tDgD)
        rD_smem_view = smem_tiled_load.retile(rD)

        cute.copy(smem_tiled_load, tSsD, rD_smem_view)
        cute.copy(gmem_tiled_store, rD, tDgD)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=1024)
    args = parser.parse_args()

    S = torch.arange(args.M * args.N, dtype=torch.float32, device="cuda").reshape(args.M, args.N)
    s_ref = S.clone().T
    D = torch.empty((args.N, args.M), dtype=torch.float32, device="cuda")

    sc = from_dlpack(S, assumed_align=32)
    dc = from_dlpack(D, assumed_align=32)

    transpose = Transpose(args.M, args.N, BM=32, BN=64)

    jitted = cute.compile(transpose, sc, dc)
    jitted(sc, dc)

    assert torch.allclose(D, s_ref)

if __name__ == "__main__":
    main()
