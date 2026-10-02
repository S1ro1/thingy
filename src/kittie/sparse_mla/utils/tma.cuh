#pragma once

#include "cuda.h"
#include "cuda_bf16.h"

#include "kittens.cuh"

#include <cstdint>
#include <stdexcept>

namespace tma {

template <int D_LATENT, int D_ROPE> __host__ CUtensorMap init_kv_gather(void *d_kv, int S) {
    CUtensorMap map{};

    const uint64_t dims[] = {(D_LATENT + D_ROPE), static_cast<uint64_t>(S)};
    const uint64_t strides[] = {(D_LATENT + D_ROPE) * sizeof(nv_bfloat16)};
    const uint32_t box[] = {64, 1};
    const uint32_t element_strides[] = {1, 1};

    auto result = cuTensorMapEncodeTiled(&map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, d_kv, dims, strides, box,
                                         element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                         CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to initialize tensor map for KV gather.");
    }

    return map;
}

__device__ inline void tcgen_cp_4x256b(uint32_t dst_addr, uint64_t src_desc) {
    asm volatile("tcgen05.cp.cta_group::1.4x256b [%0], %1;\n" : : "r"(dst_addr), "l"(src_desc) : "memory");
}

__device__ inline void gather4(void *dst, const CUtensorMap *map, kittens::semaphore &ready, int col, int4 rows) {
    uint32_t dst_addr = __cvta_generic_to_shared(dst);
    uint32_t bar_addr = __cvta_generic_to_shared(&ready);

    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global"
                 ".tile::gather4.mbarrier::complete_tx::bytes.cta_group::1 "
                 "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                 :
                 : "r"(dst_addr), "l"(map), "r"(col), "r"(rows.x), "r"(rows.y), "r"(rows.z), "r"(rows.w), "r"(bar_addr)
                 : "memory");
}

} // namespace tma
