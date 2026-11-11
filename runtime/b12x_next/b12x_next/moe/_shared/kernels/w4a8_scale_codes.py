"""Coded W4A8 N64 block scales (ds41 patch b12x_next-scale-codes; idea: Christopher Owen, spark-ds41f
experiments/2026-10-18-scale-codes).

The N64 in-place repack stores, per expert, ``rows * K/32`` UE8M0 bytes in the order the kernels read them.
The coded form keeps that order and replaces every byte by a 4-bit delta from one base per expert:

    coded expert block = [16-byte header: base in byte 0, zeros] + [nibbles, two per byte, low nibble first]
    native byte at expert-local offset o  ==  base + ((block[16 + o // 2] >> (4 * (o % 2))) & 0xF)

so ``coded_expert_bytes = 16 + native_expert_bytes / 2`` and EVERY reader's address arithmetic carries over: a
16-byte native transfer at expert-local word w is the 8 code bytes at 16 + 2 w. The expansion writes the same
bytes into the same shared-memory slots, so the MMAs, their order and every reduction see identical operands.

Exactness needs each expert's stored bytes to span at most 16 values; the host converter refuses otherwise.
The helpers below hold no dynamic control flow (plain trace-time Python over DSL values).

Readers (2026-10-30, parallel row expansion; idea credited to Christopher Owen, spark-ds41f expert-codes-v2 /
coded-arrive: every thread expands its own rows with the base in a register, after its own copies landed):
  FC1 (w4a8_phase1), cp.async path: ``coded_cp_async_16b`` copies a native 16-byte transfer's 8 code bytes and the
      expert's header word ASYNCHRONOUSLY into that transfer's own 16-byte shared slot (header -> +0, codes -> +8);
      after the thread's ``cp_async_wait_group`` (which covers its OWN copies) ``coded_expand_16b`` rewrites the slot
      in place with the 16 native bytes, before the stage's CTA barrier publishes it. No synchronous global load on
      the critical path and no extra barrier: the copier is the expander.
  FC2 (w4a8_phase2), byte path: ``coded_row4_u32`` / ``coded_row2_u32`` give one thread a whole N row of the K128
      tile (4 native bytes, or 2 + 2 zero bytes on the K tail) from one 16-bit (8-bit) code load and a base loaded
      once per tile, stored as one u32 where the native branch stores the same 4 bytes one by one.
  ``coded_stage_16b`` (synchronous) / ``coded_byte`` stay for the compact micro projection (not selected by the
  production plans: the adapter pins <= 8 rows to dynamic) and the monolithic dynamic kernel (refused with coded
  scales by _impl._get_dynamic_kernel).
"""
from __future__ import annotations

import os

from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint32

from b12x_next._lib.intrinsics import (
    cp_async_u32_shared_global,
    cp_async_u64_shared_global,
    get_ptr_as_int64,
    ld_global_b16,
    ld_global_nc_u32,
    ld_global_nc_v2_u32,
    ld_shared_u32,
    ld_shared_v2_u32,
    st_shared_u32,
    st_shared_v4_u32,
)

CODE_HEADER_BYTES = 16
ENV = "B12X_NEXT_W4A8_SCALE_CODES"


def scale_codes_enabled() -> bool:
    """Process-wide switch read where plans compile their compact launches (the dynamic path keys on the
    tensors' dtype instead). The ds41 adapter sets it from DSV41_MOE_SCALE_CODES before preparing plans."""
    return os.environ.get(ENV, "0").strip() not in ("", "0", "off", "false")


def coded_expert_bytes(native_expert_bytes: int) -> int:
    if native_expert_bytes % 16:
        raise ValueError(f"native expert scale bytes must be a multiple of 16, got {native_expert_bytes}")
    return CODE_HEADER_BYTES + native_expert_bytes // 2


def _spread16(c):
    """Four nibbles of a 16-bit code -> four bytes."""
    return (
        (c & Uint32(0xF))
        | ((c & Uint32(0xF0)) << Uint32(4))
        | ((c & Uint32(0xF00)) << Uint32(8))
        | ((c & Uint32(0xF000)) << Uint32(12))
    )


def coded_stage_16b(sfb_u32, dst_addr, expert_idx, native_expert_words, local_word):
    """Replace ``cp_async4_shared_global(dst_addr, &native_u32[expert * native_expert_words + local_word])``.

    ``local_word`` is a multiple of 4 (one 16-byte transfer). Synchronous load + expand + four st.shared: the
    bytes land in the same slots before the stage barrier the consumer already waits on."""
    block_words = Int64(expert_idx) * (Int64(native_expert_words) // Int64(2) + Int64(CODE_HEADER_BYTES // 4))
    base = ld_global_nc_u32(get_ptr_as_int64(sfb_u32, block_words)) & Uint32(0xFF)
    b4 = base * Uint32(0x01010101)
    c0, c1 = ld_global_nc_v2_u32(
        get_ptr_as_int64(sfb_u32, block_words + Int64(CODE_HEADER_BYTES // 4) + Int64(local_word) // Int64(2))
    )
    st_shared_u32(dst_addr, _spread16(c0 & Uint32(0xFFFF)) + b4)
    st_shared_u32(dst_addr + Int32(4), _spread16(c0 >> Uint32(16)) + b4)
    st_shared_u32(dst_addr + Int32(8), _spread16(c1 & Uint32(0xFFFF)) + b4)
    st_shared_u32(dst_addr + Int32(12), _spread16(c1 >> Uint32(16)) + b4)


def coded_byte(sfb_u8, expert_idx, native_expert_bytes, local_byte):
    """The native byte ``sfb_u8[expert * native_expert_bytes + local_byte]`` from the coded buffer."""
    block = Int64(expert_idx) * (Int64(native_expert_bytes) // Int64(2) + Int64(CODE_HEADER_BYTES))
    lb = Int64(local_byte)
    base = Int32(sfb_u8[block])
    code = Int32(sfb_u8[block + Int64(CODE_HEADER_BYTES) + (lb >> Int64(1))])
    shift = Int32(lb & Int64(1)) * Int32(4)
    return Uint8(base + ((code >> shift) & Int32(15)))


def coded_cp_async_16b(sfb_u32, dst_addr, expert_idx, native_expert_words, local_word):
    """Asynchronous half of a coded 16-byte transfer (FC1). Replaces
    ``cp_async4_shared_global(dst_addr, &native_u32[expert * native_expert_words + local_word])`` with two cp.async
    copies into the same 16-byte slot: the expert's header word (base in byte 0) -> dst_addr, and the 8 code bytes of
    that transfer -> dst_addr + 8 (8-byte aligned in global memory: the expert block is 16 + native/2 bytes with
    native a multiple of 16, and local_word is a multiple of 4). The slot holds no native bytes until
    ``coded_expand_16b`` runs after this thread's cp_async_wait_group."""
    block_words = Int64(expert_idx) * (Int64(native_expert_words) // Int64(2) + Int64(CODE_HEADER_BYTES // 4))
    cp_async_u32_shared_global(dst_addr, get_ptr_as_int64(sfb_u32, block_words))
    cp_async_u64_shared_global(
        dst_addr + Int32(8),
        get_ptr_as_int64(sfb_u32, block_words + Int64(CODE_HEADER_BYTES // 4) + Int64(local_word) // Int64(2)),
    )


def coded_expand_16b(dst_addr):
    """In-place expansion of a slot filled by ``coded_cp_async_16b`` (the same thread, after its own copies landed):
    reads the base (+0) and the two code words (+8, +12) first, then writes the 16 native bytes in one v4 store, the
    exact bytes the native cp.async would have written."""
    b4 = (ld_shared_u32(dst_addr) & Uint32(0xFF)) * Uint32(0x01010101)
    c0, c1 = ld_shared_v2_u32(dst_addr + Int32(8))
    st_shared_v4_u32(
        dst_addr,
        _spread16(c0 & Uint32(0xFFFF)) + b4,
        _spread16(c0 >> Uint32(16)) + b4,
        _spread16(c1 & Uint32(0xFFFF)) + b4,
        _spread16(c1 >> Uint32(16)) + b4,
    )


def coded_block_base(sfb_u8, expert_idx, native_expert_bytes):
    """(first byte of the expert's coded block, its base) for the FC2 row readers; the caller loads the base once."""
    block = Int64(expert_idx) * (Int64(native_expert_bytes) // Int64(2) + Int64(CODE_HEADER_BYTES))
    return block, Uint32(sfb_u8[block])


def coded_row4_u32(sfb_u8, block, base, local_byte):
    """One N row of a K128 FC2 scale tile (phase 2, tile_cols == 4): the native bytes at expert-local offsets
    local_byte .. local_byte + 3 as the u32 the native branch's four byte stores produce. local_byte = 4 * row + an
    even tile offset, so the 4 codes are one aligned 16-bit load."""
    code = ld_global_b16(get_ptr_as_int64(sfb_u8, block + Int64(CODE_HEADER_BYTES) + (Int64(local_byte) >> Int64(1))))
    return _spread16(code & Uint32(0xFFFF)) + base * Uint32(0x01010101)


def coded_row2_u32(sfb_u8, block, base, local_byte):
    """The N64 K tail (tile_cols == 2): the two native bytes at local_byte, local_byte + 1 and two zero bytes above
    them (the native branch stores 0 for kb >= tile_cols); one 8-bit code load."""
    code = Uint32(sfb_u8[block + Int64(CODE_HEADER_BYTES) + (Int64(local_byte) >> Int64(1))])
    return _spread16(code & Uint32(0xFF)) + base * Uint32(0x0101)
