#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#include "utils/tma.cuh"
#include <math_constants.h>

using namespace kittens;

constexpr int MMA_ROWS = 64;

constexpr int HEADS = 64;
constexpr int D_LATENT = 512, D_ROPE = 64, D_QK = D_LATENT + D_ROPE;
constexpr int D_LATENT_TILED = 64;
constexpr int TOPK = 2048;
__device__ constexpr float SM_SCALE = 1.0f / 16.0f;
constexpr int NUM_WARPS = 4;

struct globals {
  using st_q_latent = st_bf<MMA_ROWS, D_LATENT>;
  using st_kv_latent = st_bf<MMA_ROWS, D_LATENT>;
  using st_q_rope = st_bf<MMA_ROWS, D_ROPE>;
  using st_kv_rope = st_bf<MMA_ROWS, D_ROPE>;
  using st_p = st_bf<MMA_ROWS, MMA_ROWS>;

  using rt_p = rt<float, MMA_ROWS / NUM_WARPS, MMA_ROWS>;
  using tt_p = tt<float, MMA_ROWS, MMA_ROWS>;
  using rt_o = rt<float, MMA_ROWS / NUM_WARPS, D_LATENT_TILED>;
  using tt_o = tt<float, MMA_ROWS, D_LATENT>;

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
           sizeof(st_kv_rope) + sizeof(sv_fl<MMA_ROWS>) + sizeof(st_p);
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
  auto &sP = salloc.allocate<globals::st_p>();
  auto &sMask = salloc.allocate<sv_fl<MMA_ROWS>>();

  auto tP = talloc.allocate<globals::tt_p>(0, 0);
  auto tO = talloc.allocate<globals::tt_o>(1, 0);

  __shared__ semaphore q_ready;
  __shared__ semaphore k_ready;
  __shared__ semaphore p_ready;
  __shared__ semaphore o_ready;

  if (threadIdx.x == 0) {
    init_semaphore(q_ready, 0, 1);
    init_semaphore(k_ready, 0, 1);
    init_semaphore(p_ready, 0, 1);
    init_semaphore(o_ready, 0, 1);
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

  globals::rt_p::col_vec tile_max;
  globals::rt_p::col_vec alpha;
  globals::rt_p::col_vec l;

  tile_max = -INFINITY;
  l = 0.f;

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

      sMask[dst_row] = indices4.x < g.q.depth() ? 0.f : -INFINITY;
      sMask[dst_row + 1] = indices4.y < g.q.depth() ? 0.f : -INFINITY;
      sMask[dst_row + 2] = indices4.z < g.q.depth() ? 0.f : -INFINITY;
      sMask[dst_row + 3] = indices4.w < g.q.depth() ? 0.f : -INFINITY;

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
    __syncthreads();

    globals::rt_p::row_vec rMask;
    warp::load(rMask, sMask);

    if (tidx == 0) {
      kittens::mm_ABt(tP, sQ_latent, sK_latent);
      kittens::mma_ABt(tP, sQ_rope, sK_rope);

      detail::tcgen05::commit<1>(p_ready, 0b11);
    }

    globals::rt_p rp; // [num_heads, tokens we're attending to in this tile]

    wait(p_ready, k_tile & 1);
    tensor_after_thread_sync();

    warpgroup::load_async(rp, tP);
    tensor_load_wait();
    tensor_before_thread_sync();
    __syncthreads();
    tensor_after_thread_sync();

    warp::add_col(rp, rp, rMask);
    warpgroup::mul(rp, rp, SM_SCALE);

    auto prev_tile_max = tile_max;

    warpgroup::row_max(tile_max, rp, prev_tile_max);

    warpgroup::sub(alpha, prev_tile_max, tile_max);
    warpgroup::exp(alpha, alpha);

    warpgroup::sub_row(rp, rp, tile_max);
    warpgroup::exp(rp, rp);

    warpgroup::mul(l, l, alpha);
    warpgroup::row_sum(l, rp, l);

    warpgroup::store(sP, rp);
    __syncthreads();

    globals::rt_o rescale_out;
    if (k_tile > 0) {
      for (int rescale_chunk = 0; rescale_chunk < D_LATENT / D_LATENT_TILED;
           ++rescale_chunk) {
        auto tChunk = tO.subtile<tt<float, MMA_ROWS, D_LATENT_TILED>>(
            0, rescale_chunk * D_LATENT_TILED);
        warpgroup::load_async(rescale_out, tChunk);
        tensor_load_wait();
        warpgroup::mul_row(rescale_out, rescale_out, alpha);
        warpgroup::store_async(tChunk, rescale_out);
        tensor_store_wait();
      }
    }

    tensor_before_thread_sync();
    __syncthreads();
    tensor_after_thread_sync();

    if (tidx == 0) {
#pragma unroll
      for (int instruction_idx = 0; instruction_idx < 2; ++instruction_idx) {
        const int instruction_size = D_LATENT / 2;
        auto &sK_latent_chunk =
            sK_latent.subtile<instruction_size>(instruction_idx);
        auto tO_chunk = tO.subtile<tt<float, MMA_ROWS, instruction_size>>(
            0, instruction_idx * instruction_size);
        if (k_tile == 0) {
          kittens::mm_AB(tO_chunk, sP, sK_latent_chunk);
        } else {
          kittens::mma_AB(tO_chunk, sP, sK_latent_chunk);
        }
      }
      detail::tcgen05::commit<1>(o_ready, 0b11);
    }
    wait(o_ready, k_tile & 1);
    tensor_after_thread_sync();
  }

  globals::rt_o rO_chunk;
  for (int scale_chunk = 0; scale_chunk < D_LATENT / D_LATENT_TILED;
       ++scale_chunk) {
    auto tChunk = tO.subtile<tt<float, MMA_ROWS, D_LATENT_TILED>>(
        0, scale_chunk * D_LATENT_TILED);
    warpgroup::load_async(rO_chunk, tChunk);
    tensor_load_wait();
    warpgroup::div_row(rO_chunk, rO_chunk, l);

    warpgroup::store(g.out, rO_chunk,
                     coord<>{0, bidx, 0, scale_chunk * D_LATENT_TILED});
  }

  warpgroup::log2(l, l);
  warpgroup::mul(tile_max, tile_max, CUDART_L2E_F);
  warpgroup::add(tile_max, tile_max, l);
  warpgroup::store(g.lse, tile_max, {0, 0, bidx, 0});
}

PYBIND11_MODULE(_sparse_mla, m) {
  py::bind_kernel<sparse_mla>(m, "sparse_mla", &globals::q, &globals::kv,
                              &globals::indices, &globals::out, &globals::lse);
}
