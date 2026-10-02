#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#include "utils/tma.cuh"
#include "utils/utils.h"

using namespace kittens;

constexpr int MMA_ROWS = 64;

constexpr int HEADS = 64;
constexpr int D_LATENT = 512, D_ROPE = 64, D_QK = D_LATENT + D_ROPE;
constexpr int TOPK = 2048;
[[maybe_unused]] constexpr float SM_SCALE = 1.0f / 16.0f;
constexpr int NUM_WARPS = 4;

struct globals {
  using st_q_latent = st_bf<MMA_ROWS, D_LATENT>;
  using st_kv_latent = st_bf<MMA_ROWS, D_LATENT>;
  using st_q_rope = st_bf<MMA_ROWS, D_ROPE>;
  using st_kv_rope = st_bf<MMA_ROWS, D_ROPE>;

  CUtensorMap kv_gather_map;

  gl<bf16, 1, -1, HEADS, D_QK, st_q_rope, st_q_latent> q; // [1, S, 64, 576]
  gl<bf16, 1, -1, 1, D_QK> kv;          // [1, S+1, 1, 576]; last row is zero
  gl<int, 1, -1, 1, TOPK> indices;      // [1, S, 1, 2048]; sentinel index is S
  gl<bf16, 1, -1, HEADS, D_LATENT> out; // [1, S, 64, 512]
  gl<float, 1, 1, -1, HEADS> lse;       // [1, S, 64]; log2(sum(exp(scores)))

  __host__ globals(decltype(q) q_, decltype(kv) kv_, decltype(indices) indices_,
                   decltype(out) out_, decltype(lse) lse_)
      : q(q_), kv(kv_), indices(indices_), out(out_), lse(lse_) {
    kv_gather_map =
        ::tma::init_kv_gather<D_LATENT, D_ROPE>(kv_.raw_ptr, kv.depth());
  }

  dim3 grid() const { return {static_cast<unsigned>(q.depth()), 1, 1}; }
  dim3 block() const { return {NUM_WARPS * 32, 1, 1}; }
  int dynamic_shared_memory() const {
    return sizeof(st_q_latent) + sizeof(st_q_rope) + sizeof(st_kv_latent) +
           sizeof(st_kv_rope);
  }
};

__global__ __launch_bounds__(NUM_WARPS * 32) void sparse_mla(
    const __grid_constant__ globals g) {
  extern __shared__ __align__(1024) int smem[];
  tma_swizzle_allocator salloc(smem);
  tensor_allocator<1, 1> talloc;

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;

  auto &sQ_latent = salloc.allocate<globals::st_q_latent>();
  auto &sQ_rope = salloc.allocate<globals::st_q_rope>();
  auto &sK_latent = salloc.allocate<globals::st_kv_latent>();
  auto &sK_rope = salloc.allocate<globals::st_kv_rope>();

  __shared__ semaphore q_ready;
  __shared__ semaphore k_ready;

  if (threadIdx.x == 0) {
    init_semaphore(q_ready, 0, 1);
    init_semaphore(k_ready, 0, 1);
  }
  __syncthreads();

  if (tidx == 0) {
    kittens::tma::expect_bytes(q_ready, sizeof(globals::st_q_latent) +
                                            sizeof(globals::st_q_rope));
    kittens::tma::load_async(sQ_latent, g.q, coord<>{0, bidx, 0, 0},
                             q_ready); // coord<> so we get global coordinates
    kittens::tma::load_async(sQ_rope, g.q, coord<>{0, bidx, 0, D_LATENT},
                             q_ready);
  }

  wait(q_ready, 0);
  for (int k_tile = 0; k_tile < TOPK / MMA_ROWS; ++k_tile) {

    if (tidx == 0) {
      kittens::tma::expect_bytes(k_ready, sizeof(globals::st_kv_latent) +
                                              sizeof(globals::st_kv_rope));
    }
    __syncthreads();

    if (tidx < MMA_ROWS / 4) {
      const int offset = bidx * TOPK + k_tile * MMA_ROWS + tidx * 4;
      const int dst_row = tidx * 4;
      int4 indices4 =
          *reinterpret_cast<const int4 *>(g.indices.raw_ptr + offset);

      ::tma::gather4(sK_rope.data + dst_row * 64, &g.kv_gather_map, k_ready,
                     D_LATENT, indices4);

#pragma unroll
      for (int band = 0; band < D_LATENT / 64; ++band) {
        const int chunk_offset = band * 64;

        ::tma::gather4(sK_latent.data + band * 64 * MMA_ROWS + dst_row * 64,
                       &g.kv_gather_map, k_ready, chunk_offset, indices4);
      }
    }

    wait(k_ready, k_tile & 1);
    __syncthreads(); // All consumers finish before the next barrier phase/load.
  }
}

PYBIND11_MODULE(_sparse_mla, m) {
  py::bind_kernel<sparse_mla>(m, "sparse_mla", &globals::q, &globals::kv,
                              &globals::indices, &globals::out, &globals::lse);
}
