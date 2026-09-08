"""Unmodified per-token quantizer subset from Prime-RL (Apache-2.0).

Source: https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/trainer/models/kernels/fp8_utils.py
Only imports/constants and the three required functions are retained.
See LICENSE alongside this file.
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

GROUP_ALIGNMENT = 128

def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


@triton.jit
def _per_token_fp8_kernel(
    x_ptr,
    out_ptr,
    sf_ptr,
    rows,
    cols,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sk,
    USE_UE8M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_offsets_i64 = row_offsets.to(tl.int64)
    col_offsets_i64 = col_offsets.to(tl.int64)
    mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)
    x = tl.load(
        x_ptr + row_offsets_i64[:, None] * stride_xm + col_offsets_i64[None, :] * stride_xn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    scale = tl.maximum(amax / 448.0, 1e-4)
    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    y = x / scale[:, None]
    tl.store(
        out_ptr + row_offsets_i64[:, None] * stride_ym + col_offsets_i64[None, :] * stride_yn,
        y.to(tl.float8e4nv),
        mask=mask,
    )
    tl.store(sf_ptr + row_offsets_i64 * stride_sm + pid_k * stride_sk, scale, mask=row_offsets < rows)


def per_token_cast_to_fp8_triton(
    x: torch.Tensor, use_ue8m0: bool, gran_k: int = GROUP_ALIGNMENT
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    assert gran_k == GROUP_ALIGNMENT
    rows, cols = x.shape
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    sf = torch.empty((rows, ceil_div(cols, gran_k)), device=x.device, dtype=torch.float32)
    grid = lambda meta: (ceil_div(rows, meta["BLOCK_M"]), ceil_div(cols, meta["BLOCK_K"]))
    _per_token_fp8_kernel[grid](
        x,
        out,
        sf,
        rows,
        cols,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        sf.stride(0),
        sf.stride(1),
        USE_UE8M0=use_ue8m0,
        BLOCK_M=8,
        BLOCK_K=gran_k,
        num_warps=4,
    )
    return out, sf
