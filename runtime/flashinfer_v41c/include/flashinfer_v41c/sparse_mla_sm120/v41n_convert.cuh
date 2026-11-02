// Copyright (c) 2026 ds41 contributors. SPDX-License-Identifier: BSD-3-Clause (same terms as the vendored
// FlashInfer sparse_mla_sm120 sources this header extends).
//
// ds41 v41n (DSV41_V41C_READER=native): in-kernel conversion of SGLang KVLayout.V41_FP4 records (288 B: 256 bytes of
// e2m1 codes, even index in the low nibble, then 32 e4m3 scales, one per 16 values) into the V4 shared-memory row the
// stock sparse_mla_sm120 math reads (448 e4m3 nope bytes with 7 ue8m0 scales, one per 64 values; 64 bf16 rope values).
// The bytes are, for every record the SGLang writer can produce, the bytes of SGLang's V4 store rule (c1 / c2 V4
// branch, v41c_store_v4_row): ue8m0 = ceil-exponent of __fdiv_rn(max(1e-4, amax), 448), value times the exact
// power-of-two inverse, RN satfinite e4m3; rope = bf16 of the exact product. The proof and the exhaustive CPU check are
// diagnostics/v41n-native-reader-20261029/DESIGN.md section 5 and tests/test_v41n_native.py.
//
// Design idea credit: Christopher Owen (christopherowen), spark-ds41f (Apache-2.0): the FP4 records are copied raw
// into shared memory by the asynchronous copy engine while the previous chunk computes, and the dequantization is
// spread over many warps instead of one IO warp. No spark-ds41f code is used here (its kernel is TileLang and
// dequantizes to BF16; this reader must reproduce the V4 FP8 operand bytes).
//
// Work units (all lanes of a warp take the same kind, so no divergence between the two paths):
//   nope tile  (record, t < 7): 64 values = 8 code words + 4 scale bytes  -> 64 e4m3 bytes + 1 ue8m0 byte
//   rope block (record, b < 4): 16 values = 2 code words + 1 scale byte   -> 16 bf16 values
// Per nope tile: the tile exponent e is the max over its 4 blocks of E[maxcode][|scale|] (a 1 KB shared table built
// once per CTA with the reference arithmetic; e is monotone in amax, so the max of the block exponents is the tile
// exponent), then per block an 8-entry e4m3 magnitude table (4 f16x2 products + 4 conversions) and one byte permute per
// 4 values, the e2m1 sign xor-ed into bit 7. Per rope block: an 8-entry bf16 table split in low / high bytes, two byte
// permutes per 4 values, the sign xor-ed into bit 15.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace flashinfer::sparse_mla_sm120 {

constexpr int V41N_DATA_BYTES = 256;
constexpr int V41N_SCALE_BYTES = 32;
constexpr int V41N_RECORD_BYTES = V41N_DATA_BYTES + V41N_SCALE_BYTES;  // 288
constexpr int V41N_NOPE_TILES = 7;                                      // 448 / 64
constexpr int V41N_ROPE_BLOCKS = 4;                                     // 64 / 16
constexpr int V41N_ETAB_BYTES = 8 * 128;                                // [maxcode 0..7][|scale byte| 0..127]

// ── Synchronization helpers of the native paths ──────────────────────────────────────────────────
// The vendored FlashInfer mbarrier / bulk-copy helpers (arch/barrier.cuh, arch/cp_async.cuh, approved closure, not
// edited) have no "memory" clobber, so the compiler may move ordinary shared-memory accesses across them. The native
// paths use these clobbered equivalents (same PTX).
__device__ __forceinline__ void v41n_mbar_arrive(uint64_t* mbar) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("{\n .reg .b64 state;\n mbarrier.arrive.shared::cta.b64 state, [%0];\n}\n" ::"r"(addr)
               : "memory");
}

__device__ __forceinline__ void v41n_mbar_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "{\n .reg .b64 state;\n mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;\n}\n" ::"r"(addr),
      "r"(tx_bytes)
      : "memory");
}

__device__ __forceinline__ void v41n_mbar_wait_parity(uint64_t* mbar, uint32_t phase) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done)
        : "r"(addr), "r"(phase)
        : "memory");
  }
}

__device__ __forceinline__ void v41n_bulk_g2s(void* smem_dst, const void* gmem_src, uint32_t bytes,
                                              uint64_t* mbar) {
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  const uint32_t bar = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
      "l"(gmem_src), "r"(bytes), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void v41n_bulk_g2s_l2hint(void* smem_dst, const void* gmem_src, uint32_t bytes,
                                                     uint64_t* mbar, uint64_t policy) {
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  const uint32_t bar = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::
          "r"(dst),
      "l"(gmem_src), "r"(bytes), "r"(bar), "l"(policy)
      : "memory");
}

// The converters write the staging rows with ordinary (generic-proxy) stores; the bulk copy that later refills the
// same rows writes through the async proxy. Each converting thread orders its stores before any later async-proxy
// access with this fence, executed after its last store and before the arrival / barrier that lets the buffer be
// reused (PTX memory model, proxy fences).
__device__ __forceinline__ void v41n_fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t v41n_prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ float v41n_e4m3_value(uint32_t byte) {
  __nv_fp8_e4m3 s;
  s.__x = static_cast<__nv_fp8_storage_t>(byte & 0xFFu);
  return static_cast<float>(s);
}

__device__ __forceinline__ float v41n_e2m1_magnitude(int code) {
  // 0, 0.5, 1, 1.5, 2, 3, 4, 6 (exact)
  return code < 4 ? 0.5f * static_cast<float>(code)
                  : (code == 4 ? 2.0f : (code == 5 ? 3.0f : (code == 6 ? 4.0f : 6.0f)));
}

// The V4 store rule's exponent byte for a tile whose |value| maximum is amax (the IO-warp reader's arithmetic).
__device__ __forceinline__ uint32_t v41n_exponent_byte(float amax) {
  const float sc = __fdiv_rn(fmaxf(1e-4f, amax), 448.0f);
  const uint32_t u = __float_as_uint(sc);
  return ((u >> 23) & 0xFFu) + ((u & 0x7FFFFFu) != 0u ? 1u : 0u);
}

// E[mc][s] for every e2m1 magnitude code mc (0..7) and every scale magnitude byte s (0..127). Called once per CTA
// by `nthreads` threads; the caller synchronizes before the first lookup.
__device__ __forceinline__ void v41n_fill_etab(uint8_t* etab, int tid, int nthreads) {
  for (int i = tid; i < V41N_ETAB_BYTES; i += nthreads) {
    const float amax = v41n_e2m1_magnitude(i >> 7) * v41n_e4m3_value(static_cast<uint32_t>(i & 127));
    etab[i] = static_cast<uint8_t>(v41n_exponent_byte(amax));
  }
}

// Largest magnitude code (0..7) among the 16 e2m1 codes of two words: each nibble's code becomes the thermometer byte
// (1 << code) - 1 by a byte permute, the bytes are or-ed, the population count is the largest code.
__device__ __forceinline__ int v41n_max_code(uint32_t w0, uint32_t w1) {
  constexpr uint32_t TH_LO = 0x07030100u, TH_HI = 0x7F3F1F0Fu;
  const uint32_t m0 = w0 & 0x77777777u, m1 = w1 & 0x77777777u;
  uint32_t t = v41n_prmt(TH_LO, TH_HI, m0) | v41n_prmt(TH_LO, TH_HI, m0 >> 16) |
               v41n_prmt(TH_LO, TH_HI, m1) | v41n_prmt(TH_LO, TH_HI, m1 >> 16);
  t |= t >> 16;
  t |= t >> 8;
  return __popc(t & 0xFFu);
}

__device__ __forceinline__ uint32_t v41n_e4m3x2(__half2 h) {
  return static_cast<uint32_t>(
      __nv_cvt_halfraw2_to_fp8x2(static_cast<__half2_raw>(h), __NV_SATFINITE, __NV_E4M3));
}

// One nope tile. w: the tile's 8 code words (64 values), sw: its 4 scale bytes (block b in byte b). out: 64 e4m3
// bytes in value order; returns the ue8m0 byte.
__device__ __forceinline__ uint32_t v41n_nope_tile(const uint32_t (&w)[8], uint32_t sw,
                                                   const uint8_t* __restrict__ etab,
                                                   uint32_t (&out)[16]) {
  uint32_t e = 0;
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const int mc = v41n_max_code(w[2 * b], w[2 * b + 1]);
    e = max(e, static_cast<uint32_t>(etab[(mc << 7) | ((sw >> (8 * b)) & 0x7Fu)]));
  }
  const float inv = __uint_as_float((254u - e) << 23);
  // f16x2 magnitude pairs (low half = even code): {0, 0.5}, {1, 1.5}, {2, 3}, {4, 6}
  const __half2 c01 = __halves2half2(__ushort_as_half(0x0000), __ushort_as_half(0x3800));
  const __half2 c23 = __halves2half2(__ushort_as_half(0x3C00), __ushort_as_half(0x3E00));
  const __half2 c45 = __halves2half2(__ushort_as_half(0x4000), __ushort_as_half(0x4200));
  const __half2 c67 = __halves2half2(__ushort_as_half(0x4400), __ushort_as_half(0x4600));
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    // m = scale x 2^(127 - e): exact in fp32; clamped to the f16 range (only an all-zero block can exceed it,
    // and its codes are 0, so the clamp never changes a used table entry).
    const float m =
        fminf(fmaxf(v41n_e4m3_value(sw >> (8 * b)) * inv, -65504.0f), 65504.0f);
    const __half2 m2 = __float2half2_rn(m);
    const uint32_t t_lo = v41n_e4m3x2(__hmul2(c01, m2)) | (v41n_e4m3x2(__hmul2(c23, m2)) << 16);
    const uint32_t t_hi = v41n_e4m3x2(__hmul2(c45, m2)) | (v41n_e4m3x2(__hmul2(c67, m2)) << 16);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const uint32_t x = w[2 * b + k];
      const uint32_t mag = x & 0x77777777u;
      const uint32_t x4 = x << 4;
      // byte i of the result: magnitude of nibble i; bit 7: sign bit (bit 3) of nibble i
      out[4 * b + 2 * k] =
          v41n_prmt(t_lo, t_hi, mag) ^ (v41n_prmt(x, x4, 0x1504u) & 0x80808080u);
      out[4 * b + 2 * k + 1] =
          v41n_prmt(t_lo, t_hi, mag >> 16) ^ (v41n_prmt(x, x4, 0x3726u) & 0x80808080u);
    }
  }
  return e;
}

// One rope block: 16 values (2 code words, scale byte s) -> 16 bf16 (8 words, value order).
__device__ __forceinline__ void v41n_rope_block(uint32_t w0, uint32_t w1, uint32_t s,
                                                uint32_t (&out)[8]) {
  const float sv = v41n_e4m3_value(s);
  uint32_t p[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __nv_bfloat162 pr = __floats2bfloat162_rn(v41n_e2m1_magnitude(2 * j) * sv,
                                                    v41n_e2m1_magnitude(2 * j + 1) * sv);
    p[j] = *reinterpret_cast<const uint32_t*>(&pr);
  }
  const uint32_t lo_lo = v41n_prmt(p[0], p[1], 0x6420u), lo_hi = v41n_prmt(p[2], p[3], 0x6420u);
  const uint32_t hi_lo = v41n_prmt(p[0], p[1], 0x7531u), hi_hi = v41n_prmt(p[2], p[3], 0x7531u);
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const uint32_t x = k ? w1 : w0;
    const uint32_t mag = x & 0x77777777u;
    const uint32_t x4 = x << 4;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t sel = h ? (mag >> 16) : mag;
      const uint32_t l = v41n_prmt(lo_lo, lo_hi, sel);
      const uint32_t g = v41n_prmt(hi_lo, hi_hi, sel);
      // values 4h..4h+3 of word k: bf16 pairs (l0 h0 l1 h1), (l2 h2 l3 h3); signs into bits 15 and 31
      out[4 * k + 2 * h] =
          v41n_prmt(l, g, 0x5140u) ^ (v41n_prmt(x, x4, h ? 0x2060u : 0x0040u) & 0x80008000u);
      out[4 * k + 2 * h + 1] =
          v41n_prmt(l, g, 0x7362u) ^ (v41n_prmt(x, x4, h ? 0x3070u : 0x1050u) & 0x80008000u);
    }
  }
}

// One IO warp stages a 64-record chunk: entry e = k * 32 + lane (k = 0, 1) holds index idx_raw[k] (< 0 = masked).
// Lane 0 publishes the valid mask, then arrives with the chunk's transaction bytes; every row gets two bulk copies
// (256 code bytes -> row byte 0, 32 scale bytes -> row byte 256). Masked rows copy record 0.
template <int ROW_STRIDE>
__device__ __forceinline__ void v41n_io_issue_chunk(uint8_t* rows, const uint8_t* __restrict__ extra, int pbs,
                                                    size_t page_stride, const int (&idx_raw)[2],
                                                    uint64_t* mask_slot, uint64_t* mbar, int lane) {
  const uint32_t b0 = __ballot_sync(0xffffffffu, idx_raw[0] >= 0);
  const uint32_t b1 = __ballot_sync(0xffffffffu, idx_raw[1] >= 0);
  if (lane == 0) {
    *mask_slot = (uint64_t)b0 | ((uint64_t)b1 << 32);
    v41n_mbar_arrive_expect_tx(mbar, 64u * V41N_RECORD_BYTES);
  }
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int idx = idx_raw[k] >= 0 ? idx_raw[k] : 0;
    const int page = idx / pbs;
    const int local = idx - page * pbs;
    const uint8_t* page_base = extra + (size_t)page * page_stride;
    uint8_t* row = rows + (size_t)(k * 32 + lane) * ROW_STRIDE;
    v41n_bulk_g2s(row, page_base + (size_t)local * V41N_DATA_BYTES, V41N_DATA_BYTES, mbar);
    v41n_bulk_g2s(row + V41N_DATA_BYTES,
                  page_base + (size_t)pbs * V41N_DATA_BYTES + (size_t)local * V41N_SCALE_BYTES,
                  V41N_SCALE_BYTES, mbar);
  }
}

// ── Work-unit loops over one 64-record chunk staged in place ───────────────────────────────────
// Records are staged at row r of `rows` (row stride ROW_STRIDE bytes): bytes [0, 256) codes, [256, 288) scales.
// The conversion writes the V4 row in place: nope bytes [0, 448) of the same row, the ue8m0 bytes to
// sc[r * SC_STRIDE + t], the rope values to rope[r * 64 + ...]. Every staged byte a unit needs is loaded before any
// unit stores (the caller's barrier between v41n_load_units and v41n_store_units). Records with valid bit 0 convert
// as an all-zero record (codes 0, scales 0), as the IO-warp reader does.
constexpr int V41N_NOPE_UNITS_PER_REC = V41N_NOPE_TILES;
constexpr int V41N_ROPE_UNITS_PER_REC = V41N_ROPE_BLOCKS;

struct V41nUnitIn {
  uint32_t w[8];
  uint32_t s;
};

template <int ROW_STRIDE>
__device__ __forceinline__ void v41n_load_nope(const uint8_t* rows, int rec, int t, bool valid,
                                               V41nUnitIn& in) {
  const uint8_t* row = rows + (size_t)rec * ROW_STRIDE;
  const uint4 a = *reinterpret_cast<const uint4*>(row + 32 * t);
  const uint4 b = *reinterpret_cast<const uint4*>(row + 32 * t + 16);
  const uint32_t s = *reinterpret_cast<const uint32_t*>(row + V41N_DATA_BYTES + 4 * t);
  in.w[0] = valid ? a.x : 0u;
  in.w[1] = valid ? a.y : 0u;
  in.w[2] = valid ? a.z : 0u;
  in.w[3] = valid ? a.w : 0u;
  in.w[4] = valid ? b.x : 0u;
  in.w[5] = valid ? b.y : 0u;
  in.w[6] = valid ? b.z : 0u;
  in.w[7] = valid ? b.w : 0u;
  in.s = valid ? s : 0u;
}

template <int ROW_STRIDE>
__device__ __forceinline__ void v41n_load_rope(const uint8_t* rows, int rec, int blk, bool valid,
                                               V41nUnitIn& in) {
  const uint8_t* row = rows + (size_t)rec * ROW_STRIDE;
  const uint2 a = *reinterpret_cast<const uint2*>(row + 224 + 8 * blk);
  const uint32_t s = *reinterpret_cast<const uint32_t*>(row + V41N_DATA_BYTES + 28);
  in.w[0] = valid ? a.x : 0u;
  in.w[1] = valid ? a.y : 0u;
  in.s = valid ? ((s >> (8 * blk)) & 0xFFu) : 0u;
}

template <int ROW_STRIDE, int SC_STRIDE>
__device__ __forceinline__ void v41n_store_nope(uint8_t* rows, uint8_t* sc, int rec, int t,
                                                const V41nUnitIn& in,
                                                const uint8_t* __restrict__ etab) {
  uint32_t out[16];
  const uint32_t e = v41n_nope_tile(in.w, in.s, etab, out);
  uint4* dst = reinterpret_cast<uint4*>(rows + (size_t)rec * ROW_STRIDE + 64 * t);
  dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
  dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
  dst[2] = make_uint4(out[8], out[9], out[10], out[11]);
  dst[3] = make_uint4(out[12], out[13], out[14], out[15]);
  sc[rec * SC_STRIDE + t] = static_cast<uint8_t>(e);
}

template <int SC_STRIDE>
__device__ __forceinline__ void v41n_store_rope(__nv_bfloat16* rope, uint8_t* sc, int rec, int blk,
                                                const V41nUnitIn& in) {
  uint32_t out[8];
  v41n_rope_block(in.w[0], in.w[1], in.s, out);
  uint4* dst = reinterpret_cast<uint4*>(rope + (size_t)rec * 64 + 16 * blk);
  dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
  dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
  if (blk == 0) sc[rec * SC_STRIDE + V41N_NOPE_TILES] = 0;  // the V4 footer's pad byte
}

}  // namespace flashinfer::sparse_mla_sm120
