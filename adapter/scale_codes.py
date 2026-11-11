"""Routed-expert block scales as 4-bit codes (gate DSV41_MOE_SCALE_CODES=1, default OFF; needs DSV41_MOE_B12X_NEXT).

Idea: Christopher Owen, spark-ds41f experiments/2026-10-18-scale-codes (r6x): UE8M0 scales as a base exponent plus
4-bit codes, expanded in the GEMM. Prior art he credits: Local Inference Lab's B12X MXFP4-CSF.

b12x_next repacks each layer's E8M0 scales IN PLACE into its N64 W4A8 storage (``rows * K/32`` bytes per expert, no
padding, in the order the kernels read them). This module turns that storage into

    coded expert block = [16-byte header: base (the expert's smallest byte), 15 zero bytes]
                         + [the expert's bytes - base, as nibbles, two per byte, low nibble first, SAME ORDER]

so the native byte at expert-local offset o is ``base + nibble(o)`` and every kernel reader keeps its address
arithmetic (16-byte transfer at word w -> 8 code bytes at 16 + 2w). The patched kernels
(scripts/b12x_next-scale-codes.patch) expand into the same shared-memory bytes.

Contract: lossless relative to 1059dfe's POST-REPACK operands (b12x clamps bytes > 247, including the E8M0 NaN
encoding 0xff, before this code sees them; 0x00 is the exponent 2^-127, not a float zero). moe_b12x_next refuses
checkpoint bytes > 247 before the repack, so with the gate on the codes also preserve the original checkpoint bytes.
Output bit identity on GB10 is the GPU gate's claim (tests/gpu_test_scale_codes.py), not this module's.

Exact iff every expert's stored bytes span <= 16 values. DeepSeek-V4.1-Flash-1017 at 1bae0e1 (scanned in full, all 43
MoE layers incl. the three DSpark stages): every layer spans <= 16 values (mtp.2 exactly 16: 112..127); per expert and
TP4 slice the maximum is 13. A layer that does not fit is refused at load (the boot fails; nothing is approximated).

Memory at TP4 / EP1 / N = 576: 4.05 GiB of native scales per rank -> 2.03 GiB of codes (+ 16 B per expert).
"""
import os

import torch

HEADER = 16
ENABLED = os.environ.get("DSV41_MOE_SCALE_CODES", "0").strip() not in ("", "0", "off", "false")
MAX_CHUNK = 64          # experts per conversion chunk: <= 64 x 184,320 B native + int32 unpack scratch (~60 MB)


def chunk_experts(value=None):
    """The validated chunk size: an integer in [1, MAX_CHUNK] (REVIEW_sol_scalecodes R2: a negative value used to
    skip every loop and return all-zero codes as success). Malformed, zero, negative or oversized values refuse."""
    raw = os.environ.get("DSV41_MOE_SCALE_CODES_CHUNK", "32") if value is None else value
    try:
        c = int(str(raw).strip())
    except ValueError:
        raise CodeError(f"DSV41_MOE_SCALE_CODES_CHUNK={raw!r} is not an integer") from None
    if not 1 <= c <= MAX_CHUNK:
        raise CodeError(f"DSV41_MOE_SCALE_CODES_CHUNK={c} outside [1, {MAX_CHUNK}]")
    return c


class CodeError(ValueError):
    pass


def coded_bytes(native_expert_bytes):
    if native_expert_bytes % 16:
        raise CodeError(f"native expert scale bytes {native_expert_bytes} are not a multiple of 16")
    return HEADER + native_expert_bytes // 2


def pack(native, chunk=None):
    """native: uint8 [E, B] (an expert's stored scale bytes per row). Returns (coded uint8 [E, 16 + B/2],
    max span over experts). Works on any device; scratch is O(chunk * B). Every expert is packed and checked;
    anything else refuses."""
    chunk = chunk_experts(chunk)
    if native.dtype != torch.uint8 or native.dim() != 2:
        raise CodeError(f"expected uint8 [E, B], got {native.dtype} {tuple(native.shape)}")
    E, B = native.shape
    if E < 1 or B < 1:
        raise CodeError(f"empty scale storage [E={E}, B={B}]")
    out = torch.zeros((E, coded_bytes(B)), dtype=torch.uint8, device=native.device)
    worst, done = 0, 0
    for e0 in range(0, E, chunk):
        x = native[e0:e0 + chunk]
        base = x.amin(dim=1)
        span = (x.amax(dim=1).to(torch.int32) - base.to(torch.int32) + 1)
        s = int(span.max())
        if s > 16:
            e = e0 + int(span.argmax())
            raise CodeError(f"expert {e}: stored scales span {s} values; 4-bit codes hold 16")
        worst = max(worst, s)
        nib = x - base.unsqueeze(1)                       # uint8, x >= base
        out[e0:e0 + chunk, 0] = base
        out[e0:e0 + chunk, HEADER:] = nib[:, 0::2] | (nib[:, 1::2] << 4)
        done += x.shape[0]
    if done != E or worst < 1:
        raise CodeError(f"packed {done} of {E} experts (max span {worst})")
    return out, worst


def unpack(coded, native_bytes):
    """Inverse of pack (tests and the boot self-check)."""
    base = coded[:, :1].to(torch.int32)
    c = coded[:, HEADER:].to(torch.int32)
    nib = torch.stack([c & 0xF, c >> 4], dim=-1).reshape(coded.shape[0], -1)[:, :native_bytes]
    return (nib + base).to(torch.uint8)


def convert(rep_sfb, label=""):
    """rep_sfb: the prepared representation's scale tensor (int32 words, [E, words] or [E, ...]).
    Returns (coded uint8 [E, 16 + B/2], max span); verifies the round trip byte for byte on the device."""
    chunk = chunk_experts()
    E = int(rep_sfb.shape[0])
    native = rep_sfb.contiguous().view(torch.uint8).reshape(E, -1)
    coded, span = pack(native, chunk)
    checked = 0
    for e0 in range(0, E, chunk):
        if not torch.equal(unpack(coded[e0:e0 + chunk], native.shape[1]), native[e0:e0 + chunk]):
            raise CodeError(f"{label}: round trip differs in experts {e0}..{e0 + chunk - 1}")
        checked += coded[e0:e0 + chunk].shape[0]
    if checked != E:
        raise CodeError(f"{label}: round trip checked {checked} of {E} experts")
    return coded, span
