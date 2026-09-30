#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

// Default: thread 0 of CTA 0. Use DEBUG_PRINT_IF to select another warp/lane.
#ifdef KITTIE_DEBUG
#define DEBUG_PRINT_IF(condition, ...)                                         \
  do {                                                                         \
    if (condition)                                                             \
      printf(__VA_ARGS__);                                                     \
  } while (0)
#define DEBUG_PRINT(...)                                                       \
  DEBUG_PRINT_IF(blockIdx.x == 0 && threadIdx.x == 0, __VA_ARGS__)
#else
#define DEBUG_PRINT_IF(...)                                                    \
  do {                                                                         \
  } while (0)
#define DEBUG_PRINT(...)                                                       \
  do {                                                                         \
  } while (0)
#endif

constexpr int BM = 128, BN = 256, BK = 64;
constexpr int NUM_WARPS = 6; // 0-3 epilogue, 4 producer, 5 issuer

constexpr int NUM_SMEM_STAGES = 4;
constexpr int NUM_ACC_STAGES = 2;
constexpr int SUPER_M = 8;
constexpr int EPI_COLS = 64;
constexpr int NUM_EPI_STAGES = 2;

using a_st = st_bf<BM, BK>;
using b_st = st_bf<BN, BK>;
using c_st = st_bf<BM, 64>;
using epi_tile = st_bf<BM, EPI_COLS>;

using acc_tt = tt<float, BM, BN>;

struct globals {
  gl<bf16, 1, 1, -1, -1, a_st> A;
  gl<bf16, 1, 1, -1, -1, b_st> B; // B is [N, K], so C = A @ B.T.
  gl<bf16, 1, 1, -1, -1, c_st> C;

  const int producer_warp_id = 4;
  const int mma_warp_id = 5;

  dim3 grid() const { return {static_cast<unsigned>(kittens::num_sms()), 1}; }
  dim3 block() const { return {NUM_WARPS * 32, 1, 1}; }

  int dynamic_shared_memory() const {
    return NUM_SMEM_STAGES * (sizeof(a_st) + sizeof(b_st)) +
           NUM_EPI_STAGES * sizeof(epi_tile) + 1024;
  }
};

__device__ inline auto get_tile_ids(const int tile_idx, const int num_m_tiles,
                                    const int num_n_tiles) {

  auto tiles_per_band = SUPER_M * num_n_tiles;
  auto band_idx = tile_idx / tiles_per_band;
  auto first_m = band_idx * SUPER_M;

  auto rows_in_band = min(SUPER_M, num_m_tiles - first_m);
  auto within_band = tile_idx % tiles_per_band;

  auto m_tile_idx = first_m + (within_band % rows_in_band);
  auto n_tile_idx = within_band / rows_in_band;

  return std::make_tuple(m_tile_idx, n_tile_idx);
}

__global__
__launch_bounds__(32 * NUM_WARPS) void gemm(const __grid_constant__ globals g) {
  extern __shared__ int smem[];
  tma_swizzle_allocator alloc(smem);
  tensor_allocator<1, 1> tmem_alloc;

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;

  const int num_m_tiles = g.C.rows() / BM;
  const int num_n_tiles = g.C.cols() / BN;
  const int num_k_tiles = g.A.cols() / a_st::cols;
  const int num_output_tiles = num_m_tiles * num_n_tiles;

  const int warp_id = warp::groupid();

  auto &sA = alloc.allocate<a_st, NUM_SMEM_STAGES>();
  auto &sB = alloc.allocate<b_st, NUM_SMEM_STAGES>();

  auto &sAcc = alloc.allocate<epi_tile, NUM_EPI_STAGES>();

  __shared__ semaphore inputs_ready[NUM_SMEM_STAGES];
  __shared__ semaphore inputs_free[NUM_SMEM_STAGES];

  __shared__ semaphore acc_ready[NUM_ACC_STAGES];
  __shared__ semaphore acc_free[NUM_ACC_STAGES];

  __shared__ semaphore epi_tile_free[NUM_EPI_STAGES];

  if (tidx == 0) {
    for (int stage = 0; stage < NUM_SMEM_STAGES; ++stage) {
      init_semaphore(inputs_ready[stage], 0, 1);
      init_semaphore(inputs_free[stage], 1, 0);
    }
    for (int stage = 0; stage < NUM_ACC_STAGES; ++stage) {
      init_semaphore(acc_ready[stage], 0, 1);
      init_semaphore(acc_free[stage], 4, 0);
    }
    for (int stage = 0; stage < NUM_EPI_STAGES; ++stage) {
      init_semaphore(epi_tile_free[stage], 1, 0);
    }
  }
  __syncthreads();

  // producer mainloop
  int task_idx = 0;
  for (int tile_idx = bidx; tile_idx < num_output_tiles;
       tile_idx += gridDim.x, ++task_idx) {

    const int acc_stage = task_idx % NUM_ACC_STAGES;
    const int acc_phase = (task_idx / NUM_ACC_STAGES) & 1;

    auto [m_tile_idx, n_tile_idx] =
        get_tile_ids(tile_idx, num_m_tiles, num_n_tiles);

    auto tAcc = tmem_alloc.allocate<acc_tt>(acc_stage * BN);

    if (warp_id == g.producer_warp_id) {
      for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_step = task_idx * num_k_tiles + k_tile;

        const int smem_stage = k_step % NUM_SMEM_STAGES;
        const int smem_phase = (k_step / NUM_SMEM_STAGES) & 1;

        wait(inputs_free[smem_stage], smem_phase ^ 1);

        if (warp::laneid() == 0) {
          tma::expect_bytes(inputs_ready[smem_stage],
                            sizeof(a_st) + sizeof(b_st));

          tma::load_async(sA[smem_stage], g.A, {m_tile_idx, k_tile},
                          inputs_ready[smem_stage]);
          tma::load_async(sB[smem_stage], g.B, {n_tile_idx, k_tile},
                          inputs_ready[smem_stage]);
        }
      }
      // consumer mainloop
    } else if (warp_id == g.mma_warp_id) {
      wait(acc_free[acc_stage], acc_phase ^ 1);
      tensor_after_thread_sync();
      for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_step = task_idx * num_k_tiles + k_tile;
        const int mma_stage = k_step % NUM_SMEM_STAGES;
        const int mma_phase = (k_step / NUM_SMEM_STAGES) & 1;

        wait(inputs_ready[mma_stage], mma_phase);

        if (warp::laneid() == 0) {
          if (k_tile == 0) {
            mm_ABt(tAcc, sA[mma_stage], sB[mma_stage], inputs_free[mma_stage]);
          } else {
            mma_ABt(tAcc, sA[mma_stage], sB[mma_stage], inputs_free[mma_stage]);
          }
        }
      }
      if (warp::laneid() == 0) {
        detail::tcgen05::commit<1>(acc_ready[acc_stage]);
      }
    }

    if (warp_id < 4) {
      constexpr int num_epi_tiles = BN / EPI_COLS;
      rt_fl<BM / 4, EPI_COLS> rAcc[num_epi_tiles];

      wait(acc_ready[acc_stage], acc_phase);
      tensor_after_thread_sync();

#pragma unroll
      for (int epi_tile = 0; epi_tile < num_epi_tiles; ++epi_tile) {
        auto epi_col = epi_tile * EPI_COLS;
        auto chunk = tAcc.template subtile<tt<float, BM, EPI_COLS>>(0, epi_col);
        warpgroup::load_async(rAcc[epi_tile], chunk);
      }
      tensor_load_wait();

      tensor_before_thread_sync();
      __syncwarp();
      if (kittens::laneid() == 0) {
        kittens::arrive(acc_free[acc_stage]);
      }

      for (int epi_tile = 0; epi_tile < BN / EPI_COLS; ++epi_tile) {
        const int epi_stage = epi_tile % NUM_EPI_STAGES;

        warpgroup::tma::store_async_wait<NUM_EPI_STAGES - 1>();
        warpgroup::sync(1);
        warpgroup::store(sAcc[epi_stage], rAcc[epi_tile]);
        warpgroup::sync(2);
        warpgroup::tma::store_async(
            g.C, sAcc[epi_stage],
            {m_tile_idx, n_tile_idx * (BN / EPI_COLS) + epi_tile});
      }
    }
  }
  warpgroup::tma::store_async_wait();
  __syncthreads();
}

PYBIND11_MODULE(_C, m) {
  py::bind_kernel<gemm>(m, "gemm", &globals::A, &globals::B, &globals::C);
  m.attr("BM") = BM;
  m.attr("BN") = BN;
  m.attr("BK") = BK;
}
