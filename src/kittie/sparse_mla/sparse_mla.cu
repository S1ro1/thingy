#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#include "utils/tma.cuh"
#include <math_constants.h>

using namespace kittens;

constexpr int HEADS_PER_MMA = 64;
constexpr int KV_TOKENS_PER_MMA = 64;

constexpr int HEADS = 64;
constexpr int D_LATENT = 512, D_ROPE = 64, D_QK = D_LATENT + D_ROPE;
constexpr int D_LATENT_TILED = 64;
constexpr int TOPK = 2048;
constexpr int NUM_WARPS = 8;
constexpr int NUM_CONSUMER_WARPS = 4;
constexpr int NUM_SMEM_STAGES = 2;

__device__ constexpr float SM_SCALE = 1.0f / 16.0f;

struct globals {
    using st_q_latent = st_bf<HEADS_PER_MMA, D_LATENT>;
    using st_kv_latent = st_bf<HEADS_PER_MMA, D_LATENT>;
    using st_q_rope = st_bf<HEADS_PER_MMA, D_ROPE>;
    using st_kv_rope = st_bf<HEADS_PER_MMA, D_ROPE>;
    using st_p = st_bf<HEADS_PER_MMA, KV_TOKENS_PER_MMA>;

    using rt_p = rt<float, HEADS_PER_MMA / NUM_CONSUMER_WARPS, KV_TOKENS_PER_MMA>;
    using tt_p = tt<float, HEADS_PER_MMA, KV_TOKENS_PER_MMA>;
    using rt_o = rt<float, HEADS_PER_MMA / NUM_CONSUMER_WARPS, D_LATENT_TILED>;
    using tt_o = tt<float, HEADS_PER_MMA, D_LATENT>;
    using tt_o_chunk = tt<float, HEADS_PER_MMA, D_LATENT_TILED>;
    using tt_q_latent = tt<bf16, HEADS_PER_MMA, D_LATENT>;

    CUtensorMap kv_gather_map;

    gl<bf16, 1, -1, HEADS, D_QK, st_q_rope, st_q_latent> q; // [1, S, 64, 576]
    gl<bf16, 1, -1, 1, D_QK> kv;                            // [1, S+1, 1, 576]; last row is zero
    gl<int, 1, -1, 1, TOPK> indices;                        // [1, S, 1, 2048]; sentinel index is S
    gl<bf16, 1, -1, HEADS, D_LATENT> out;                   // [1, S, 64, 512]
    gl<float, 1, 1, -1, HEADS> lse;                         // [1, S, 64]; log2(sum(exp(scores)))

    __host__ globals(decltype(q) q_, decltype(kv) kv_, decltype(indices) indices_, decltype(out) out_, decltype(lse) lse_)
        : q(q_), kv(kv_), indices(indices_), out(out_), lse(lse_) {
        kv_gather_map = ::tma::init_kv_gather<D_LATENT, D_ROPE>(kv_.raw_ptr, kv.depth());
    }

    dim3 grid() const { return {static_cast<unsigned>(q.depth()), 1, 1}; }
    dim3 block() const { return {NUM_WARPS * 32, 1, 1}; }
    int dynamic_shared_memory() const {
        return (sizeof(st_q_rope) + sizeof(st_kv_latent) + sizeof(st_kv_rope) + sizeof(sv_fl<HEADS_PER_MMA>)) * NUM_SMEM_STAGES +
               sizeof(st_p);
    }
};

__global__ __launch_bounds__(NUM_WARPS * 32) void sparse_mla(const __grid_constant__ globals g) {
    extern __shared__ __align__(1024) int smem[];
    tma_swizzle_allocator salloc(smem);
    tensor_allocator<1, 1> talloc;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    auto &sQ_rope = salloc.allocate<globals::st_q_rope>();
    auto &sK_latent = salloc.allocate<globals::st_kv_latent, NUM_SMEM_STAGES>();
    auto &sK_rope = salloc.allocate<globals::st_kv_rope, NUM_SMEM_STAGES>();
    auto &sMask = salloc.allocate<sv_fl<HEADS_PER_MMA>, NUM_SMEM_STAGES>();
    auto &sP = salloc.allocate<globals::st_p>();

    auto tP = talloc.allocate<globals::tt_p>(0, 0);
    auto tO = talloc.allocate<globals::tt_o>(1, 0);
    auto tQ_latent = talloc.allocate<globals::tt_q_latent>(0, 64);

    __shared__ semaphore q_smem_ready;
    __shared__ semaphore q_tmem_ready;
    __shared__ semaphore kv_ready[NUM_SMEM_STAGES];
    __shared__ semaphore p_ready;
    __shared__ semaphore o_ready;

    __shared__ semaphore kv_empty[NUM_SMEM_STAGES];

    globals::rt_p rp; // [num_heads, tokens we're attending to in this tile]
    globals::rt_p::col_vec tile_max;
    globals::rt_p::col_vec alpha;
    globals::rt_p::col_vec l;

    tile_max = -INFINITY;
    l = 0.f;

    if (threadIdx.x == 0) {
        init_semaphore(q_smem_ready, 0, 1);
        init_semaphore(p_ready, 0, 1);
        init_semaphore(o_ready, 0, 1);
        init_semaphore(q_tmem_ready, 0, 128);

        for (int stage = 0; stage < NUM_SMEM_STAGES; ++stage) {
            init_semaphore(kv_ready[stage], 0, 1);
            init_semaphore(kv_empty[stage], 1, 0);
        }
    }
    __syncthreads();

    if (warpgroup::groupid() == 0) {
        // q to smem
        if (tidx == 0) {
            kittens::tma::expect_bytes(q_smem_ready, sizeof(globals::st_q_latent) + sizeof(globals::st_q_rope));
            kittens::tma::load_async(sK_latent[1], g.q, coord<>{0, bidx, 0, 0},
                                     q_smem_ready); // coord<> so we get global coordinates
            kittens::tma::load_async(sQ_rope, g.q, coord<>{0, bidx, 0, D_LATENT}, q_smem_ready);
        }
        wait(q_smem_ready, 0);

        // q to tmem
        for (int chunk_idx = 0; chunk_idx < 8; ++chunk_idx) {
            auto &sChunk = sK_latent[1].subtile<D_LATENT_TILED>(chunk_idx);
            auto tChunk = tQ_latent.subtile<tt_bf<HEADS_PER_MMA, D_LATENT_TILED>>(0, chunk_idx * D_LATENT_TILED);
            rt_bf<16, 64> rQ;
            warpgroup::load(rQ, sChunk);
            warpgroup::store_async(tChunk, rQ);
        }

        for (int k_tile_idx = 0; k_tile_idx < TOPK / KV_TOKENS_PER_MMA; ++k_tile_idx) {
            const int smem_stage = k_tile_idx % NUM_SMEM_STAGES;
            const int smem_phase = (k_tile_idx / NUM_SMEM_STAGES) & 1;

            if (k_tile_idx >= NUM_SMEM_STAGES) {
                wait(kv_empty[smem_stage], smem_phase ^ 1);
            }

            const int offset = bidx * TOPK + k_tile_idx * HEADS_PER_MMA + tidx * 4;
            const int dst_row = tidx * 4;
            int4 indices4;
            if (tidx < HEADS_PER_MMA / 4) {
                indices4 = *reinterpret_cast<const int4 *>(g.indices.raw_ptr + offset);

                sMask[smem_stage][dst_row] = indices4.x < g.q.depth() ? 0.f : -INFINITY;
                sMask[smem_stage][dst_row + 1] = indices4.y < g.q.depth() ? 0.f : -INFINITY;
                sMask[smem_stage][dst_row + 2] = indices4.z < g.q.depth() ? 0.f : -INFINITY;
                sMask[smem_stage][dst_row + 3] = indices4.w < g.q.depth() ? 0.f : -INFINITY;
            }

            // On tile 0 this also ensures all producer warps finished reading
            // Q from stage 1 before any later iteration can reuse that stage.
            warpgroup::sync(2);
            if (tidx == 0) {
                kittens::tma::expect_bytes(kv_ready[smem_stage], sizeof(globals::st_kv_latent) + sizeof(globals::st_kv_rope));
            }
            warpgroup::sync(2);

            if (tidx < HEADS_PER_MMA / 4) {
                ::tma::gather4(sK_rope[smem_stage].data + dst_row * 64, &g.kv_gather_map, kv_ready[smem_stage], D_LATENT, indices4);

#pragma unroll
                for (int band = 0; band < D_LATENT / 64; ++band) {
                    const int chunk_offset = band * 64;

                    ::tma::gather4(sK_latent[smem_stage].data + band * 64 * HEADS_PER_MMA + dst_row * 64, &g.kv_gather_map,
                                   kv_ready[smem_stage], chunk_offset, indices4);
                }
            }
            if (k_tile_idx == 0) {
                tensor_store_wait();
                tensor_before_thread_sync();
                arrive(q_tmem_ready);
                tensor_after_thread_sync();
            }
        }
    }

    if (warpgroup::groupid() == 1) {
        const int tidx = warpgroup::laneid();
        wait(q_tmem_ready, 0);
        tensor_after_thread_sync();
        for (int k_tile_idx = 0; k_tile_idx < TOPK / HEADS_PER_MMA; ++k_tile_idx) {
            const int smem_stage = k_tile_idx % NUM_SMEM_STAGES;
            const int smem_phase = (k_tile_idx / NUM_SMEM_STAGES) & 1;
            wait(kv_ready[smem_stage], smem_phase);

            globals::rt_p::row_vec rMask;
            warp::load(rMask, sMask[smem_stage]);

            if (tidx == 0) {
                kittens::mm_ABt(tP, tQ_latent, sK_latent[smem_stage]);
                kittens::mma_ABt(tP, sQ_rope, sK_rope[smem_stage]);
                detail::tcgen05::commit<1>(p_ready, 0b11);
            }

            wait(p_ready, k_tile_idx & 1);
            tensor_after_thread_sync();

            warpgroup::load_async(rp, tP);
            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);
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
            warpgroup::sync(1);

            globals::rt_o rescale_out;
            if (k_tile_idx > 0) {
                for (int chunk_idx = 0; chunk_idx < D_LATENT / D_LATENT_TILED; ++chunk_idx) {
                    auto tChunk = tO.subtile<globals::tt_o_chunk>(0, chunk_idx * D_LATENT_TILED);
                    warpgroup::load_async(rescale_out, tChunk);
                    tensor_load_wait();
                    warpgroup::mul_row(rescale_out, rescale_out, alpha);
                    warpgroup::store_async(tChunk, rescale_out);
                    tensor_store_wait();
                }
            }

            tensor_before_thread_sync();
            warpgroup::sync(1);
            tensor_after_thread_sync();

            if (tidx == 0) {
#pragma unroll
                for (int instr_idx = 0; instr_idx < 2; ++instr_idx) {
                    const int instruction_size = D_LATENT / 2;
                    auto &sK_latent_chunk = sK_latent[smem_stage].subtile<instruction_size>(instr_idx);
                    auto tO_chunk = tO.subtile<tt<float, HEADS_PER_MMA, instruction_size>>(0, instr_idx * instruction_size);
                    if (k_tile_idx == 0) {
                        kittens::mm_AB(tO_chunk, sP, sK_latent_chunk);
                    } else {
                        kittens::mma_AB(tO_chunk, sP, sK_latent_chunk);
                    }
                }
                detail::tcgen05::commit<1>(o_ready, 0b11);
            }
            wait(o_ready, k_tile_idx & 1);
            tensor_after_thread_sync();

            // One consumer arrival per stage; initialize kv_empty with count 1.
            if (tidx == 0) {
                arrive(kv_empty[smem_stage]);
            }
        }

        globals::rt_o rO_chunk;
        for (int chunk_idx = 0; chunk_idx < D_LATENT / D_LATENT_TILED; ++chunk_idx) {
            auto tChunk = tO.subtile<globals::tt_o_chunk>(0, chunk_idx * D_LATENT_TILED);
            warpgroup::load_async(rO_chunk, tChunk);
            tensor_load_wait();
            warpgroup::div_row(rO_chunk, rO_chunk, l);

            warpgroup::store(g.out, rO_chunk, coord<>{0, bidx, 0, chunk_idx * D_LATENT_TILED});
        }

        warpgroup::log2(l, l);
        warpgroup::mul(tile_max, tile_max, CUDART_L2E_F);
        warpgroup::add(tile_max, tile_max, l);
        warpgroup::store(g.lse, tile_max, {0, 0, bidx, 0});
    }

    // Warp 0 frees TMEM in talloc's destructor; wait for the consumer to finish.
    tensor_before_thread_sync();
    __syncthreads();
}

PYBIND11_MODULE(_sparse_mla, m) {
    py::bind_kernel<sparse_mla>(m, "sparse_mla", &globals::q, &globals::kv, &globals::indices, &globals::out, &globals::lse);
}
