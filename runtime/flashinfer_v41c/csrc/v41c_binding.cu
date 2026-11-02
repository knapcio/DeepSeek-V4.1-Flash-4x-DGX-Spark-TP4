// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Modified for ds41 (DSV41_KV_LAYOUT=v41c): see runtime/flashinfer_v41c/PROVENANCE.md.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <flashinfer_v41c/sparse_mla_sm120/model/model_type.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace flashinfer::sparse_mla_sm120 {

using bf16 = __nv_bfloat16;

bool launch_v41c_decode_dsv4(ModelType mt, int num_heads, int topk, int page_block_size,
                             int num_tokens, int num_splits, const bf16* Q,
                             const uint8_t* KV_cache, const int32_t* indices, bf16* mid_out,
                             float* mid_lse, bf16* output, float* out_lse,
                             const int* topk_length, const float* attn_sink,
                             const uint8_t* extra_KV_cache, const int32_t* extra_indices,
                             const int* extra_topk_length, int extra_topk, int pbs_extra,
                             size_t stride_extra_kv_block, int chunks_per_block_override,
                             float sm_scale, size_t stride_kv_block, cudaStream_t stream);

// ds41 DSV41_SWA_NATIVE_DECODE: swan_decode_dsv4.cu (stock kernel, SWA pool read in its 256-token pages)
bool launch_swan_decode_dsv4(ModelType mt, int num_heads, int topk, int page_block_size,
                             int num_tokens, int num_splits, const bf16* Q,
                             const uint8_t* KV_cache, const int32_t* indices, bf16* mid_out,
                             float* mid_lse, bf16* output, float* out_lse,
                             const int* topk_length, const float* attn_sink,
                             const uint8_t* extra_KV_cache, const int32_t* extra_indices,
                             const int* extra_topk_length, int extra_topk, int pbs_extra,
                             size_t stride_extra_kv_block, int chunks_per_block_override,
                             float sm_scale, size_t stride_kv_block, cudaStream_t stream);

// ds41 v41n (DSV41_V41C_READER=native): v41n_decode_dsv4.cu / v41n_prefill.cu
bool launch_v41n_decode_dsv4(ModelType mt, int num_heads, int topk, int page_block_size,
                             int num_tokens, int num_splits, const bf16* Q,
                             const uint8_t* KV_cache, const int32_t* indices, bf16* mid_out,
                             float* mid_lse, bf16* output, float* out_lse,
                             const int* topk_length, const float* attn_sink,
                             const uint8_t* extra_KV_cache, const int32_t* extra_indices,
                             const int* extra_topk_length, int extra_topk, int pbs_extra,
                             size_t stride_extra_kv_block, int chunks_per_block_override,
                             float sm_scale, size_t stride_kv_block, cudaStream_t stream,
                             int active_heads, int diag_mode);
bool launch_v41n_prefill(int num_heads, int topk, int page_block_size, int pbs_extra,
                         int num_tokens, int topk_extra, const bf16* Q, const uint8_t* KV_cache,
                         const int32_t* indices, const uint8_t* KV_cache_extra,
                         const int32_t* indices_extra, const float* attn_sink, bf16* output,
                         float* out_lse, float sm_scale, size_t stride_kv_block,
                         size_t stride_kv_block_extra, const int* topk_length_ptr,
                         const int* topk_length_extra_ptr, cudaStream_t stream);
bool launch_v41n_convert_dump(int mode, const uint8_t* extra, int pbs, size_t page_stride,
                              const int32_t* tokens, int n, uint8_t* out, cudaStream_t stream);

namespace {

struct PagedKVLayout {
  int page_block_size;
  size_t stride_kv_block;
};

inline PagedKVLayout parse_paged_kv_layout(const TensorView& kv, int bpt, const char* name) {
  const size_t elem_bytes = static_cast<size_t>(kv.dtype().bits / 8);
  if (kv.ndim() == 2) {
    const size_t block_bytes = static_cast<size_t>(kv.size(1)) * elem_bytes;
    TVM_FFI_ICHECK_EQ(block_bytes % static_cast<size_t>(bpt), 0)
        << name << " 2D block width " << block_bytes
        << " is not divisible by bytes_per_token=" << bpt;
    return {static_cast<int>(block_bytes / static_cast<size_t>(bpt)), block_bytes};
  }
  if (kv.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(kv.size(-1), bpt)
        << name << " 3D form must be [num_pages, page_block_size, " << bpt << "]";
    return {static_cast<int>(kv.size(1)), static_cast<size_t>(kv.stride(0)) * elem_bytes};
  }
  TVM_FFI_ICHECK_EQ(kv.ndim(), 4) << name << " must be 2D [num_pages, page_bytes], 3D "
                                  << "[num_pages, page_block_size, bytes_per_token], HND "
                                  << "[num_pages, 1, page_block_size, bytes_per_token], or NHD "
                                  << "[num_pages, page_block_size, 1, bytes_per_token]";
  TVM_FFI_ICHECK_EQ(kv.size(-1), bpt)
      << name << " last dim must be bytes_per_token=" << bpt << ", got " << kv.size(-1);
  if (kv.size(1) == 1) {
    return {static_cast<int>(kv.size(2)), static_cast<size_t>(kv.stride(0)) * elem_bytes};
  }
  if (kv.size(2) == 1) {
    return {static_cast<int>(kv.size(1)), static_cast<size_t>(kv.stride(0)) * elem_bytes};
  }
  TVM_FFI_ICHECK(false) << name << " 4D form must have singleton KV-head axis at dim 1 "
                        << "(HND) or dim 2 (NHD)";
  return {0, 0};
}

}  // namespace

// ds41 v41c: FlashInfer's decode-dsv4 TVM-FFI wrapper (base: base/sparse_mla_sm120_jit_binding.cu) with the
// extra cache parsed as 288-byte V4.1 FP4 records. Exported as v41c_decode_dsv4 from its own JIT module. The caller passes
// already-sized scratch tensors mid_out + mid_lse plus the output and lse.
// Handles DSV4 decode with page_block_size=64 and the supported head/top-k
// instantiation grid in sparse_mla_sm120_decode_dsv4.cu.
void V41cDecodeDsv4(TensorView q, TensorView kv_cache, TensorView indices,
                              TensorView mid_out, TensorView mid_lse, TensorView output,
                              TensorView out_lse, int64_t num_splits, double sm_scale,
                              Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
                              Optional<TensorView> extra_kv_cache,
                              Optional<TensorView> extra_indices,
                              Optional<TensorView> extra_topk_length,
                              int64_t chunks_per_block_override) {
  TVM_FFI_ICHECK_EQ(q.ndim(), 3) << "q must be [T, H, D_QK]";
  TVM_FFI_ICHECK_GE(kv_cache.ndim(), 2);
  // indices may be 2D [T, topk] or 3D [T, s_q=1, topk] (some callers keep
  // the s_q singleton dim through the call stack). The kernel walks
  // `indices + t_idx * stride_per_t`, where stride is captured below from
  // .stride(0).
  TVM_FFI_ICHECK_GE(indices.ndim(), 2)
      << "indices must have at least 2 dims; got ndim=" << indices.ndim();
  if (indices.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(indices.size(1), 1)
        << "indices 3D form requires size(1) == 1; got " << indices.size(1);
  }

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int topk = static_cast<int>(indices.size(-1));
  const int d_qk = static_cast<int>(q.size(2));
  ModelType mt = (d_qk == 512) ? ModelType::DSV4 : ModelType::DSV3_2;
  // Currently the kernel only supports DSV4.
  TVM_FFI_ICHECK_EQ(static_cast<int>(mt), static_cast<int>(ModelType::DSV4))
      << "decode-dsv4 currently DSV4-only";

  constexpr int BPT_DSV4 = 584;
  const PagedKVLayout kv_layout = parse_paged_kv_layout(kv_cache, BPT_DSV4, "kv_cache");
  const int page_block_size = kv_layout.page_block_size;

  const int* topk_len_ptr =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;
  const float* attn_sink_ptr =
      attn_sink.has_value() ? static_cast<const float*>(attn_sink.value().data_ptr()) : nullptr;
  const uint8_t* extra_kv_ptr = extra_kv_cache.has_value()
                                    ? static_cast<const uint8_t*>(extra_kv_cache.value().data_ptr())
                                    : nullptr;
  const int32_t* extra_indices_ptr =
      extra_indices.has_value() ? static_cast<const int32_t*>(extra_indices.value().data_ptr())
                                : nullptr;
  const int* extra_topk_len_ptr =
      extra_topk_length.has_value() ? static_cast<const int*>(extra_topk_length.value().data_ptr())
                                    : nullptr;
  // extra_topk and stride_extra_kv_block are derived from the optional tensors.
  int extra_topk_arg = 0;
  int pbs_extra_arg = 0;
  size_t stride_extra_kv_block = 0;
  if (extra_kv_cache.has_value()) {
    TVM_FFI_ICHECK(extra_indices.has_value()) << "extra_kv_cache requires extra_indices";
    const auto& ekv = extra_kv_cache.value();
    extra_topk_arg = static_cast<int>(extra_indices.value().size(-1));
    // v41c: the extra (compressed) cache holds 288-byte V4.1 FP4 records (SGLang KVLayout.V41_FP4).
    constexpr int BPT_V41_FP4 = 288;
    const PagedKVLayout extra_layout = parse_paged_kv_layout(ekv, BPT_V41_FP4, "extra_kv_cache");
    pbs_extra_arg = extra_layout.page_block_size;
    stride_extra_kv_block = extra_layout.stride_kv_block;
  }

  cudaStream_t stream = get_stream(q.device());
  bool ok = launch_v41c_decode_dsv4(
      mt, num_heads, topk, page_block_size, num_tokens, static_cast<int>(num_splits),
      static_cast<const bf16*>(q.data_ptr()), static_cast<const uint8_t*>(kv_cache.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()), static_cast<bf16*>(mid_out.data_ptr()),
      static_cast<float*>(mid_lse.data_ptr()), static_cast<bf16*>(output.data_ptr()),
      static_cast<float*>(out_lse.data_ptr()), topk_len_ptr, attn_sink_ptr, extra_kv_ptr,
      extra_indices_ptr, extra_topk_len_ptr, extra_topk_arg, pbs_extra_arg, stride_extra_kv_block,
      static_cast<int>(chunks_per_block_override), static_cast<float>(sm_scale),
      kv_layout.stride_kv_block, stream);
  TVM_FFI_ICHECK(ok) << "v41c decode-dsv4 launch failed (unsupported shape, no FP4 extra cache, or kernel error)";
}

// ds41 DSV41_SWA_NATIVE_DECODE (opt-in): FlashInfer's decode-dsv4 TVM-FFI wrapper (base:
// base/sparse_mla_sm120_jit_binding.cu SparseMlaSm120DecodeDsv4, = V41cDecodeDsv4 with the stock 584-byte extra
// cache) calling launch_swan_decode_dsv4: kv_cache is SGLang's SWA pool viewed with its own 256-token pages.
void SwanDecodeDsv4(TensorView q, TensorView kv_cache, TensorView indices,
                              TensorView mid_out, TensorView mid_lse, TensorView output,
                              TensorView out_lse, int64_t num_splits, double sm_scale,
                              Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
                              Optional<TensorView> extra_kv_cache,
                              Optional<TensorView> extra_indices,
                              Optional<TensorView> extra_topk_length,
                              int64_t chunks_per_block_override) {
  TVM_FFI_ICHECK_EQ(q.ndim(), 3) << "q must be [T, H, D_QK]";
  TVM_FFI_ICHECK_GE(kv_cache.ndim(), 2);
  // indices may be 2D [T, topk] or 3D [T, s_q=1, topk] (some callers keep
  // the s_q singleton dim through the call stack). The kernel walks
  // `indices + t_idx * stride_per_t`, where stride is captured below from
  // .stride(0).
  TVM_FFI_ICHECK_GE(indices.ndim(), 2)
      << "indices must have at least 2 dims; got ndim=" << indices.ndim();
  if (indices.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(indices.size(1), 1)
        << "indices 3D form requires size(1) == 1; got " << indices.size(1);
  }

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int topk = static_cast<int>(indices.size(-1));
  const int d_qk = static_cast<int>(q.size(2));
  ModelType mt = (d_qk == 512) ? ModelType::DSV4 : ModelType::DSV3_2;
  // Currently the kernel only supports DSV4.
  TVM_FFI_ICHECK_EQ(static_cast<int>(mt), static_cast<int>(ModelType::DSV4))
      << "decode-dsv4 currently DSV4-only";

  constexpr int BPT_DSV4 = 584;
  const PagedKVLayout kv_layout = parse_paged_kv_layout(kv_cache, BPT_DSV4, "kv_cache");
  const int page_block_size = kv_layout.page_block_size;

  const int* topk_len_ptr =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;
  const float* attn_sink_ptr =
      attn_sink.has_value() ? static_cast<const float*>(attn_sink.value().data_ptr()) : nullptr;
  const uint8_t* extra_kv_ptr = extra_kv_cache.has_value()
                                    ? static_cast<const uint8_t*>(extra_kv_cache.value().data_ptr())
                                    : nullptr;
  const int32_t* extra_indices_ptr =
      extra_indices.has_value() ? static_cast<const int32_t*>(extra_indices.value().data_ptr())
                                : nullptr;
  const int* extra_topk_len_ptr =
      extra_topk_length.has_value() ? static_cast<const int*>(extra_topk_length.value().data_ptr())
                                    : nullptr;
  // extra_topk and stride_extra_kv_block are derived from the optional tensors.
  int extra_topk_arg = 0;
  int pbs_extra_arg = 0;
  size_t stride_extra_kv_block = 0;
  if (extra_kv_cache.has_value()) {
    TVM_FFI_ICHECK(extra_indices.has_value()) << "extra_kv_cache requires extra_indices";
    const auto& ekv = extra_kv_cache.value();
    extra_topk_arg = static_cast<int>(extra_indices.value().size(-1));
    // swan: the extra (compressed) cache in the stock V4 layout (584 bytes per token, 64 / 128 / 256-token pages).
    const PagedKVLayout extra_layout = parse_paged_kv_layout(ekv, BPT_DSV4, "extra_kv_cache");
    pbs_extra_arg = extra_layout.page_block_size;
    stride_extra_kv_block = extra_layout.stride_kv_block;
  }

  cudaStream_t stream = get_stream(q.device());
  TVM_FFI_ICHECK_EQ(page_block_size, 256) << "swan: kv_cache must be the SWA pool's 256-token pages";
  bool ok = launch_swan_decode_dsv4(
      mt, num_heads, topk, page_block_size, num_tokens, static_cast<int>(num_splits),
      static_cast<const bf16*>(q.data_ptr()), static_cast<const uint8_t*>(kv_cache.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()), static_cast<bf16*>(mid_out.data_ptr()),
      static_cast<float*>(mid_lse.data_ptr()), static_cast<bf16*>(output.data_ptr()),
      static_cast<float*>(out_lse.data_ptr()), topk_len_ptr, attn_sink_ptr, extra_kv_ptr,
      extra_indices_ptr, extra_topk_len_ptr, extra_topk_arg, pbs_extra_arg, stride_extra_kv_block,
      static_cast<int>(chunks_per_block_override), static_cast<float>(sm_scale),
      kv_layout.stride_kv_block, stream);
  TVM_FFI_ICHECK(ok) << "swan decode-dsv4 launch failed (unsupported shape or kernel error)";
}

// ds41 v41n (DSV41_V41C_READER=native): V41cDecodeDsv4 with the native-reader launcher (raw records copied into
// the V4 rows, converted in shared memory by the kernel's converter warps).
void V41nDecodeDsv4(TensorView q, TensorView kv_cache, TensorView indices,
                              TensorView mid_out, TensorView mid_lse, TensorView output,
                              TensorView out_lse, int64_t num_splits, double sm_scale,
                              Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
                              Optional<TensorView> extra_kv_cache,
                              Optional<TensorView> extra_indices,
                              Optional<TensorView> extra_topk_length,
                              int64_t chunks_per_block_override, int64_t active_heads, int64_t diag_mode) {
  TVM_FFI_ICHECK_EQ(q.ndim(), 3) << "q must be [T, H, D_QK]";
  TVM_FFI_ICHECK_GE(kv_cache.ndim(), 2);
  // indices may be 2D [T, topk] or 3D [T, s_q=1, topk] (some callers keep
  // the s_q singleton dim through the call stack). The kernel walks
  // `indices + t_idx * stride_per_t`, where stride is captured below from
  // .stride(0).
  TVM_FFI_ICHECK_GE(indices.ndim(), 2)
      << "indices must have at least 2 dims; got ndim=" << indices.ndim();
  if (indices.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(indices.size(1), 1)
        << "indices 3D form requires size(1) == 1; got " << indices.size(1);
  }

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int topk = static_cast<int>(indices.size(-1));
  const int d_qk = static_cast<int>(q.size(2));
  ModelType mt = (d_qk == 512) ? ModelType::DSV4 : ModelType::DSV3_2;
  // Currently the kernel only supports DSV4.
  TVM_FFI_ICHECK_EQ(static_cast<int>(mt), static_cast<int>(ModelType::DSV4))
      << "decode-dsv4 currently DSV4-only";

  constexpr int BPT_DSV4 = 584;
  const PagedKVLayout kv_layout = parse_paged_kv_layout(kv_cache, BPT_DSV4, "kv_cache");
  const int page_block_size = kv_layout.page_block_size;

  const int* topk_len_ptr =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;
  const float* attn_sink_ptr =
      attn_sink.has_value() ? static_cast<const float*>(attn_sink.value().data_ptr()) : nullptr;
  const uint8_t* extra_kv_ptr = extra_kv_cache.has_value()
                                    ? static_cast<const uint8_t*>(extra_kv_cache.value().data_ptr())
                                    : nullptr;
  const int32_t* extra_indices_ptr =
      extra_indices.has_value() ? static_cast<const int32_t*>(extra_indices.value().data_ptr())
                                : nullptr;
  const int* extra_topk_len_ptr =
      extra_topk_length.has_value() ? static_cast<const int*>(extra_topk_length.value().data_ptr())
                                    : nullptr;
  // extra_topk and stride_extra_kv_block are derived from the optional tensors.
  int extra_topk_arg = 0;
  int pbs_extra_arg = 0;
  size_t stride_extra_kv_block = 0;
  if (extra_kv_cache.has_value()) {
    TVM_FFI_ICHECK(extra_indices.has_value()) << "extra_kv_cache requires extra_indices";
    const auto& ekv = extra_kv_cache.value();
    extra_topk_arg = static_cast<int>(extra_indices.value().size(-1));
    // v41n: the extra (compressed) cache holds 288-byte V4.1 FP4 records (SGLang KVLayout.V41_FP4).
    constexpr int BPT_V41_FP4 = 288;
    const PagedKVLayout extra_layout = parse_paged_kv_layout(ekv, BPT_V41_FP4, "extra_kv_cache");
    pbs_extra_arg = extra_layout.page_block_size;
    stride_extra_kv_block = extra_layout.stride_kv_block;
  }

  cudaStream_t stream = get_stream(q.device());
  bool ok = launch_v41n_decode_dsv4(
      mt, num_heads, topk, page_block_size, num_tokens, static_cast<int>(num_splits),
      static_cast<const bf16*>(q.data_ptr()), static_cast<const uint8_t*>(kv_cache.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()), static_cast<bf16*>(mid_out.data_ptr()),
      static_cast<float*>(mid_lse.data_ptr()), static_cast<bf16*>(output.data_ptr()),
      static_cast<float*>(out_lse.data_ptr()), topk_len_ptr, attn_sink_ptr, extra_kv_ptr,
      extra_indices_ptr, extra_topk_len_ptr, extra_topk_arg, pbs_extra_arg, stride_extra_kv_block,
      static_cast<int>(chunks_per_block_override), static_cast<float>(sm_scale),
      kv_layout.stride_kv_block, stream, static_cast<int>(active_heads), static_cast<int>(diag_mode));
  TVM_FFI_ICHECK(ok) << "v41n decode-dsv4 launch failed (unsupported shape, no FP4 extra cache, or kernel error)";
}

// ds41 v41n: dual-cache prefill (> 64 rows) with the extra cache in V4.1 FP4 pages (128 / 256 records), read in place.
// q [T, H, 512] bf16; kv_cache: V4 64-token pages (584 B per token); indices [T, 128] (or [T, 1, 128]); output
// [T, H, 512] bf16; out_lse [T, H] f32; topk_length / extra_topk_length [T] int32 (required).
// DLPack dtype codes: 0 int, 1 uint, 2 float, 4 bfloat
inline void v41n_check_dtype(const TensorView& t, int code, int bits, const char* name) {
  TVM_FFI_ICHECK(t.dtype().code == code && t.dtype().bits == bits)
      << "v41n: " << name << " has dtype code " << static_cast<int>(t.dtype().code) << " bits "
      << static_cast<int>(t.dtype().bits) << ", want code " << code << " bits " << bits;
}

void V41nPrefill(TensorView q, TensorView kv_cache, TensorView indices, TensorView output,
                 TensorView out_lse, double sm_scale, Optional<TensorView> topk_length,
                 Optional<TensorView> attn_sink, TensorView extra_kv_cache, TensorView extra_indices,
                 Optional<TensorView> extra_topk_length) {
  TVM_FFI_ICHECK_EQ(q.ndim(), 3) << "q must be [T, H, D_QK]";
  TVM_FFI_ICHECK_EQ(q.size(2), 512) << "v41n prefill is DSV4-only (D_QK 512)";
  TVM_FFI_ICHECK_GE(indices.ndim(), 2);
  if (indices.ndim() == 3) TVM_FFI_ICHECK_EQ(indices.size(1), 1);
  TVM_FFI_ICHECK_GE(extra_indices.ndim(), 2);
  if (extra_indices.ndim() == 3) TVM_FFI_ICHECK_EQ(extra_indices.size(1), 1);
  const int64_t T = q.size(0), H = q.size(1);
  v41n_check_dtype(q, 4, 16, "q");
  v41n_check_dtype(indices, 0, 32, "indices");
  v41n_check_dtype(extra_indices, 0, 32, "extra_indices");
  v41n_check_dtype(output, 4, 16, "output");
  v41n_check_dtype(out_lse, 2, 32, "out_lse");
  v41n_check_dtype(kv_cache, 1, 8, "kv_cache");
  v41n_check_dtype(extra_kv_cache, 1, 8, "extra_kv_cache");
  TVM_FFI_ICHECK(indices.size(0) == T && extra_indices.size(0) == T) << "v41n: index rows != q rows";
  // the IO warps read whole 64-entry tiles of the extra indices: the width must be a multiple of 64
  TVM_FFI_ICHECK(extra_indices.size(-1) > 0 && extra_indices.size(-1) % 64 == 0)
      << "v41n: extra_indices width " << extra_indices.size(-1) << " is not a positive multiple of 64";
  TVM_FFI_ICHECK(output.ndim() == 3 && output.size(0) == T && output.size(1) == H && output.size(2) == 512)
      << "v41n: output must be [T, H, 512]";
  TVM_FFI_ICHECK(out_lse.ndim() == 2 && out_lse.size(0) == T && out_lse.size(1) == H)
      << "v41n: out_lse must be [T, H]";
  TVM_FFI_ICHECK(topk_length.has_value() && extra_topk_length.has_value()) << "v41n: both length tensors required";
  v41n_check_dtype(topk_length.value(), 0, 32, "topk_length");
  v41n_check_dtype(extra_topk_length.value(), 0, 32, "extra_topk_length");
  TVM_FFI_ICHECK(topk_length.value().size(0) == T && extra_topk_length.value().size(0) == T)
      << "v41n: length tensors must have T entries";
  if (attn_sink.has_value()) {
    v41n_check_dtype(attn_sink.value(), 2, 32, "attn_sink");
    TVM_FFI_ICHECK_EQ(attn_sink.value().size(0), H);
  }
  constexpr int BPT_DSV4 = 584;
  constexpr int BPT_V41_FP4 = 288;
  const PagedKVLayout kv_layout = parse_paged_kv_layout(kv_cache, BPT_DSV4, "kv_cache");
  const PagedKVLayout extra_layout = parse_paged_kv_layout(extra_kv_cache, BPT_V41_FP4, "extra_kv_cache");
  const int num_tokens = static_cast<int>(q.size(0));
  bool ok = launch_v41n_prefill(
      static_cast<int>(q.size(1)), static_cast<int>(indices.size(-1)), kv_layout.page_block_size,
      extra_layout.page_block_size, num_tokens, static_cast<int>(extra_indices.size(-1)),
      static_cast<const bf16*>(q.data_ptr()), static_cast<const uint8_t*>(kv_cache.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()),
      static_cast<const uint8_t*>(extra_kv_cache.data_ptr()),
      static_cast<const int32_t*>(extra_indices.data_ptr()),
      attn_sink.has_value() ? static_cast<const float*>(attn_sink.value().data_ptr()) : nullptr,
      static_cast<bf16*>(output.data_ptr()), static_cast<float*>(out_lse.data_ptr()),
      static_cast<float>(sm_scale), kv_layout.stride_kv_block, extra_layout.stride_kv_block,
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr,
      extra_topk_length.has_value() ? static_cast<const int*>(extra_topk_length.value().data_ptr())
                                    : nullptr,
      get_stream(q.device()));
  TVM_FFI_ICHECK(ok) << "v41n prefill launch failed (unsupported shape: heads 8/16, SWA top-k 128 over 64-token "
                        "pages, FP4 pages of 128/256 records, both length tensors) or kernel error";
}

// ds41 v41n qualification diagnostic: the native conversion code on the GPU, rows dumped in the V4 row ABI.
// extra_kv_cache: V4.1 FP4 pages; tokens [n] int32; out [n, 584] uint8; mode 0 = decode path, 1 = prefill path.
void V41nConvertDump(TensorView extra_kv_cache, TensorView tokens, TensorView out, int64_t mode) {
  constexpr int BPT_V41_FP4 = 288;
  const PagedKVLayout layout = parse_paged_kv_layout(extra_kv_cache, BPT_V41_FP4, "extra_kv_cache");
  v41n_check_dtype(extra_kv_cache, 1, 8, "extra_kv_cache");
  v41n_check_dtype(tokens, 0, 32, "tokens");
  v41n_check_dtype(out, 1, 8, "out");
  TVM_FFI_ICHECK_EQ(tokens.ndim(), 1);
  TVM_FFI_ICHECK(out.ndim() == 2 && out.size(0) == tokens.size(0) && out.size(1) == 584)
      << "v41n: out must be [n, 584]";
  const bool ok = launch_v41n_convert_dump(
      static_cast<int>(mode), static_cast<const uint8_t*>(extra_kv_cache.data_ptr()), layout.page_block_size,
      layout.stride_kv_block, static_cast<const int32_t*>(tokens.data_ptr()), static_cast<int>(tokens.size(0)),
      static_cast<uint8_t*>(out.data_ptr()), get_stream(out.device()));
  TVM_FFI_ICHECK(ok) << "v41n convert dump launch failed";
}

}  // namespace flashinfer::sparse_mla_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(v41c_decode_dsv4, flashinfer::sparse_mla_sm120::V41cDecodeDsv4);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(v41n_decode_dsv4, flashinfer::sparse_mla_sm120::V41nDecodeDsv4);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(v41n_prefill, flashinfer::sparse_mla_sm120::V41nPrefill);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(v41n_convert_dump, flashinfer::sparse_mla_sm120::V41nConvertDump);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(swan_decode_dsv4, flashinfer::sparse_mla_sm120::SwanDecodeDsv4);
