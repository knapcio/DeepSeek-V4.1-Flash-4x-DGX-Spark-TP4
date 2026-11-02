// Copyright (c) 2026 ds41 contributors. SPDX-License-Identifier: BSD-3-Clause
// ds41 v41n diagnostic (qualification only, never on the serving path): runs the native readers' conversion code on
// the GPU and dumps the shared-memory rows it builds, in the V4 row ABI (448 e4m3 nope bytes, 64 bf16 rope values,
// 7 ue8m0 bytes + the pad byte = 584 B per token), so a harness can compare them byte for byte with v2.6's V4 rows.
//   MODE 0: the decode kernel's path: v41n_io_issue_chunk (one IO warp, bulk copies) + v41n_convert_chunk (8
//           converter warps, named barrier, in place) -- the code decode_dsv4_kernel.cuh runs with EXTRA_NATIVE;
//   MODE 1: the prefill kernel's path: the same staging + v41n_convert_warp_records per math warp (8 records each).
// One CTA per 64 tokens; token < 0 = masked (converted as the all-zero record, like the kernels).

#include <cuda_runtime.h>

#include <cstdio>
#include <flashinfer_v41c/sparse_mla_sm120/decode_dsv4_kernel.cuh>
#include <flashinfer_v41c/sparse_mla_sm120/v41n_prefill_kernel.cuh>

namespace flashinfer::sparse_mla_sm120 {

namespace {

constexpr int V41N_DIAG_THREADS = 32 + V41N_CONV_THREADS;  // IO warp 0 + 8 converting warps
static_assert(V41N_CONV_THREADS == 256, "diag: 8 converting warps (decode converters / prefill math warps)");
constexpr int V41N_DIAG_ROW = KVCacheTraits<ModelType::DSV4>::KV_SMEM_STRIDE;  // 464
constexpr int V41N_DIAG_SC = 8;
constexpr size_t V41N_DIAG_SMEM = 64 * V41N_DIAG_ROW + 64 * V41N_DIAG_SC + 64 * 64 * sizeof(bf16) +
                                  V41N_ETAB_BYTES + 16 /* mask + mbar */;

template <int MODE>
__global__ void __launch_bounds__(V41N_DIAG_THREADS)
    v41n_convert_dump_kernel(const uint8_t* __restrict__ extra, int pbs, size_t page_stride,
                             const int32_t* __restrict__ tokens, int n, uint8_t* __restrict__ out) {
  extern __shared__ __align__(16) char smem[];
  uint8_t* rows = reinterpret_cast<uint8_t*>(smem);
  uint8_t* sc = rows + 64 * V41N_DIAG_ROW;
  bf16* rope = reinterpret_cast<bf16*>(sc + 64 * V41N_DIAG_SC);
  uint8_t* etab = reinterpret_cast<uint8_t*>(rope + 64 * 64);
  uint64_t* mask = reinterpret_cast<uint64_t*>(etab + V41N_ETAB_BYTES);
  uint64_t* mbar = mask + 1;
  const int base = blockIdx.x * 64;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) mbarrier_init(mbar, 1);
  if (warp > 0) v41n_fill_etab(etab, threadIdx.x - 32, V41N_CONV_THREADS);
  __syncthreads();
  if (warp == 0) {
    int idx_raw[2];
    for (int k = 0; k < 2; ++k) {
      const int t = base + k * 32 + lane;
      idx_raw[k] = t < n ? tokens[t] : -1;
    }
    v41n_io_issue_chunk<V41N_DIAG_ROW>(rows, extra, pbs, page_stride, idx_raw, mask, mbar, lane);
  }
  v41n_mbar_wait_parity(mbar, 0);
  if (warp > 0) {
    if constexpr (MODE == 0) {
      v41n_convert_chunk<V41N_DIAG_ROW, V41N_DIAG_SC>(rows, sc, rope, *mask, etab, threadIdx.x - 32);
    } else {
      const int len = n - base;  // position-valid records of this tile
      v41n_convert_warp_records(rows, sc, rope, tokens + base, 0, len, (warp - 1) * 8, lane, etab);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 64 * 584; i += V41N_DIAG_THREADS) {
    const int e = i / 584, j = i - e * 584;
    if (base + e >= n) continue;
    uint8_t b;
    if (j < 448) b = rows[e * V41N_DIAG_ROW + j];
    else if (j < 576) b = reinterpret_cast<const uint8_t*>(rope + e * 64)[j - 448];
    else b = sc[e * V41N_DIAG_SC + (j - 576)];
    out[(size_t)(base + e) * 584 + j] = b;
  }
}

}  // namespace

bool launch_v41n_convert_dump(int mode, const uint8_t* extra, int pbs, size_t page_stride,
                              const int32_t* tokens, int n, uint8_t* out, cudaStream_t stream) {
  if (n <= 0) return true;
  if (pbs <= 0 || (mode != 0 && mode != 1)) return false;
  const dim3 grid((n + 63) / 64), block(V41N_DIAG_THREADS);
  if (mode == 0) {
    if (cudaFuncSetAttribute(v41n_convert_dump_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)V41N_DIAG_SMEM) != cudaSuccess)
      return false;
    v41n_convert_dump_kernel<0><<<grid, block, V41N_DIAG_SMEM, stream>>>(extra, pbs, page_stride, tokens, n, out);
  } else {
    if (cudaFuncSetAttribute(v41n_convert_dump_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)V41N_DIAG_SMEM) != cudaSuccess)
      return false;
    v41n_convert_dump_kernel<1><<<grid, block, V41N_DIAG_SMEM, stream>>>(extra, pbs, page_stride, tokens, n, out);
  }
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace flashinfer::sparse_mla_sm120
