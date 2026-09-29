#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

constexpr int BM = 128, BN = 256, BK = 64;
constexpr int NUM_WARPS = 6; // 0-3 epilogue, 4 producer, 5 issuer

constexpr int NUM_SMEM_STAGES = 4;

using a_st = st_bf<BM, BK>;
using b_st = st_bf<BN, BK>;
using c_st = st_bf<BM, 64>;

using acc_tt = tt<float, BM, BN>;

struct globals {
  gl<bf16, 1, 1, -1, -1, a_st> A;
  gl<bf16, 1, 1, -1, -1, b_st> B; // B is [N, K], so C = A @ B.T.
  gl<bf16, 1, 1, -1, -1, c_st> C;

  const int producer_warp_id = 4;
  const int mma_warp_id = 5;

  dim3 grid() const {
    return {(unsigned)(C.rows() / BM), (unsigned)(C.cols() / BN), 1};
  }
  dim3 block() const { return {NUM_WARPS * 32, 1, 1}; }

  int dynamic_shared_memory() const {
    return NUM_SMEM_STAGES * (sizeof(a_st) + sizeof(b_st)) + 1024;
  }
};

__global__ __launch_bounds__(NUM_WARPS *
                             32) void gemm(const __grid_constant__ globals g) {
  extern __shared__ int smem[];
  tma_swizzle_allocator alloc(smem);
  tensor_allocator<1, 1> tmem_alloc;

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y;
  const int warp_id = warp::groupid();

  const int num_k_tiles = g.A.cols() / a_st::cols;

  auto &sA = alloc.allocate<a_st, NUM_SMEM_STAGES>();
  auto &sB = alloc.allocate<b_st, NUM_SMEM_STAGES>();

  auto tAcc = tmem_alloc.allocate<acc_tt>(0);

  __shared__ semaphore inputs_ready[NUM_SMEM_STAGES];
  __shared__ semaphore inputs_free[NUM_SMEM_STAGES];

  __shared__ semaphore acc_ready;

  if (tidx == 0) {
    for (int stage = 0; stage < NUM_SMEM_STAGES; ++stage) {
      init_semaphore(inputs_ready[stage], 0, 1);
      init_semaphore(inputs_free[stage], 1, 0);
    }
    init_semaphore(acc_ready, 0, 1);
  }
  __syncthreads();

  // producer mainloop
  if (warp_id == g.producer_warp_id) {
    if (warp::laneid() == 0) {
      for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int smem_stage = k_tile % NUM_SMEM_STAGES;
        const int smem_phase = (k_tile / NUM_SMEM_STAGES) & 1;

        wait(inputs_free[smem_stage], smem_phase ^ 1);

        tma::expect_bytes(inputs_ready[smem_stage],
                          sizeof(a_st) + sizeof(b_st));

        tma::load_async(sA[smem_stage], g.A, {bidx, k_tile},
                        inputs_ready[smem_stage]);
        tma::load_async(sB[smem_stage], g.B, {bidy, k_tile},
                        inputs_ready[smem_stage]);
      }
    }
    // consumer mainloop
  } else if (warp_id == g.mma_warp_id) {
    if (warp::laneid() == 0) {
      for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int mma_stage = k_tile % NUM_SMEM_STAGES;
        const int mma_phase = (k_tile / NUM_SMEM_STAGES) & 1;

        wait(inputs_ready[mma_stage], mma_phase);

        if (k_tile == 0) {
          mm_ABt(tAcc, sA[mma_stage], sB[mma_stage], inputs_free[mma_stage]);
        } else {
          mma_ABt(tAcc, sA[mma_stage], sB[mma_stage], inputs_free[mma_stage]);
        }
      }
      detail::tcgen05::commit<1>(acc_ready);
    }
  }
  wait(acc_ready, 0);

  if (warp_id < 4) {
    rt_bf<BM / 4, BN> rC;

    warpgroup::load_async(rC, tAcc);
    tensor_load_wait();

    warpgroup::store(g.C, rC, {bidx, bidy});
  }

  __syncthreads();
}

PYBIND11_MODULE(_C, m) {
  py::bind_kernel<gemm>(m, "gemm", &globals::A, &globals::B, &globals::C);
  m.attr("BM") = BM;
  m.attr("BN") = BN;
  m.attr("BK") = BK;
}
