// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Modified for ds41 v41n (DSV41_V41C_READER=native): FlashInfer's dual-cache MG prefill launcher
// (base: launch_prefill_mg_dual in sparse_mla_sm120_prefill.cu), with the generated native kernel
// (v41n_prefill_kernel.cuh): the extra cache holds SGLang KVLayout.V41_FP4 pages of 128 or 256 records, read in place
// (no 64-token page split, no V4 scratch). Served configuration only: DSV4, SWA top-k 128 over 64-token V4 pages,
// runtime lengths for both sections, 8 or 16 heads (one 16-head group per CTA). Anything else returns false and the
// caller keeps the group reader.

#include <cuda_runtime.h>

#include <cstdio>
#include <flashinfer_v41c/sparse_mla_sm120/v41n_prefill_kernel.cuh>

namespace flashinfer::sparse_mla_sm120 {

namespace {

constexpr int kV41nMaxDevices = 32;

#define V41N_CUDA_OK(call)                                                  \
  do {                                                                      \
    cudaError_t e_ = (call);                                                \
    if (e_ != cudaSuccess) {                                                \
      printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      return false;                                                         \
    }                                                                       \
  } while (0)

template <int NUM_HEADS, int PBS_EXTRA>
bool launch_v41n_prefill_impl(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                              const uint8_t* KV_cache_extra, const int32_t* indices_extra,
                              const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
                              int num_tokens, int topk_extra, size_t stride_kv_block,
                              size_t stride_kv_block_extra, const int* topk_length_ptr,
                              const int* topk_length_extra_ptr, cudaStream_t stream) {
  constexpr ModelType MT = ModelType::DSV4;
  constexpr ComputeMode CM = ComputeMode::BF16;
  constexpr int MG_N_HG_T = 1;
  constexpr size_t smem_bytes = SmemLayoutMG<MT, CM>::TOTAL;  // unchanged: v41n reuses the idle Q group region
  constexpr int REPLICATE_H = (NUM_HEADS + MG_N_HG_T * HPB - 1) / (MG_N_HG_T * HPB);
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = v41n_prefill_mg_dual_kernel<MT, CM, NUM_HEADS, 128, 64, PBS_EXTRA, MG_N_HG_T>;
  static bool configured[kV41nMaxDevices] = {};
  int device = 0;
  V41N_CUDA_OK(cudaGetDevice(&device));
  if (device < 0 || device >= kV41nMaxDevices || !configured[device]) {
    V41N_CUDA_OK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem_bytes)));
    if (device >= 0 && device < kV41nMaxDevices) configured[device] = true;
  }

  PrefillColdParams cold{sm_scale,   num_tokens, stride_kv_block, stride_kv_block_extra,
                         topk_extra, attn_sink,  topk_length_ptr, topk_length_extra_ptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,       (void*)&KV_cache, (void*)&indices,   (void*)&KV_cache_extra,
                  (void*)&indices_extra, (void*)&output, (void*)&out_lse, (void*)&attn_sink,
                  (void*)&cold};
  V41N_CUDA_OK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
  V41N_CUDA_OK(cudaGetLastError());
  return true;
}

}  // namespace

bool v41n_prefill_supported(int num_heads, int topk, int page_block_size, int pbs_extra,
                            bool has_topk_length, bool has_extra_topk_length) {
  return (num_heads == 8 || num_heads == 16) && topk == 128 && page_block_size == 64 &&
         (pbs_extra == 128 || pbs_extra == 256) && has_topk_length && has_extra_topk_length;
}

bool launch_v41n_prefill(int num_heads, int topk, int page_block_size, int pbs_extra,
                         int num_tokens, int topk_extra, const bf16* Q, const uint8_t* KV_cache,
                         const int32_t* indices, const uint8_t* KV_cache_extra,
                         const int32_t* indices_extra, const float* attn_sink, bf16* output,
                         float* out_lse, float sm_scale, size_t stride_kv_block,
                         size_t stride_kv_block_extra, const int* topk_length_ptr,
                         const int* topk_length_extra_ptr, cudaStream_t stream) {
  if (!v41n_prefill_supported(num_heads, topk, page_block_size, pbs_extra,
                              topk_length_ptr != nullptr, topk_length_extra_ptr != nullptr))
    return false;
  if (KV_cache_extra == nullptr || topk_extra <= 0) return false;
#define V41N_PREFILL(NH, PBSX)                                                                    \
  if (num_heads == (NH) && pbs_extra == (PBSX))                                                   \
    return launch_v41n_prefill_impl<(NH), (PBSX)>(Q, KV_cache, indices, KV_cache_extra,            \
                                                  indices_extra, attn_sink, output, out_lse,       \
                                                  sm_scale, num_tokens, topk_extra,                \
                                                  stride_kv_block, stride_kv_block_extra,          \
                                                  topk_length_ptr, topk_length_extra_ptr, stream);
  V41N_PREFILL(16, 256)
  V41N_PREFILL(16, 128)
  V41N_PREFILL(8, 256)
  V41N_PREFILL(8, 128)
#undef V41N_PREFILL
  return false;
}

}  // namespace flashinfer::sparse_mla_sm120
