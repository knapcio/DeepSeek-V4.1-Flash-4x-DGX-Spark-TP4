# Copy of diagnostics/ds-v41kv-20261027/kit/v41w_encode.py, the CPU reference the v41w window (2026-10-27 12:45, PASS) checked
# against SGLang's V41_FP4 writer and V4 store on sm_121a. Test oracle for runtime/flashinfer_v41c/v41c.py.
"""CPU reference of the V4.1 FP4 compressed-KV record in SGLang's page ABI (torch, no GPU).

The model's own rounding of a ratio-1/2 compressed latent (SGLang torch_quant.fake_quant_compressed_kv):
per 16 values, scale = e4m3(clamp(amax * f32(1/6), 2^-9, 448)), each value divided (IEEE) by the decoded
scale, RN-even e2m1 (saturating at 6). SGLang's page ABI (kv_layout.py / kv_layout.cuh, V41_FP4): a page of
`page_size` tokens holds page_size x 256 data bytes (512 e2m1 codes, even index in the low nibble), then
page_size x 32 e4m3 scale bytes, padded to a multiple of 256 bytes.
"""
import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
DATA, SCALE = 256, 32


def page_bytes(page_size):
    raw = page_size * (DATA + SCALE)
    return -(-raw // 256) * 256


def e2m1_codes(v):
    """fp32 values already divided by the scale -> 4-bit codes, RN-even, saturating, sign kept."""
    mag = v.abs().clamp(max=6.0)
    d = (mag.unsqueeze(-1) - E2M1).abs()
    best = d.min(dim=-1).values.unsqueeze(-1)
    tied = d == best
    idx = torch.arange(8).expand_as(tied)
    even = torch.where(tied & (idx % 2 == 0), idx, torch.full_like(idx, 99)).min(-1).values
    anyc = torch.where(tied, idx, torch.full_like(idx, 99)).min(-1).values
    code = torch.where(even < 99, even, anyc)
    return code | (torch.signbit(v).to(torch.int64) << 3)


def encode_rows(x):
    """x [n, 512] bf16 -> (data [n, 256] u8, scales [n, 32] u8): the model's rounding."""
    b = x.float().unflatten(-1, (-1, 16))
    amax = b.abs().amax(-1, keepdim=True)
    s8 = (amax * (1.0 / 6.0)).clamp(min=2.0 ** -9, max=448.0).to(torch.float8_e4m3fn)
    s = s8.float()
    code = e2m1_codes((b / s).clamp(-6.0, 6.0)).reshape(x.shape[0], 512).to(torch.uint8)
    data = code[:, 0::2] | (code[:, 1::2] << 4)
    return data, s8.reshape(x.shape[0], 32).view(torch.uint8)


def decode_rows(data, scales):
    """Inverse: [n, 256] + [n, 32] u8 -> [n, 512] bf16 (exact: <= 6 significant bits)."""
    codes = torch.stack((data & 15, data >> 4), dim=-1).reshape(data.shape[0], 512).long()
    val = E2M1[codes & 7]
    val = torch.where((codes & 8) != 0, -val, val)
    s = scales.view(torch.float8_e4m3fn).float()
    return (val.reshape(-1, 32, 16) * s.reshape(-1, 32, 1)).reshape(-1, 512).to(torch.bfloat16)


def expected_pages(x, slots, num_pages, page_size):
    """The page buffer a correct writer leaves after writing rows x at slots (zeros elsewhere)."""
    pb = page_bytes(page_size)
    out = torch.zeros(num_pages, pb, dtype=torch.uint8)
    data, scales = encode_rows(x)
    flat = out.view(-1)
    for i, s in enumerate(slots.tolist()):
        p, o = divmod(int(s), page_size)
        base = p * pb
        flat[base + o * DATA: base + (o + 1) * DATA] = data[i]
        sb = base + page_size * DATA + o * SCALE
        flat[sb: sb + SCALE] = scales[i]
    return out


def latents(n, seed, tail=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 512, generator=g)
    if tail:
        x = x * torch.exp(torch.randn(n, 512, generator=g) * 1.5)
    x[:, 448:] *= 3.0
    return x.to(torch.bfloat16)


def value_ties():
    """336 rows: an exact e2m1 midpoint times a non-trivial e4m3 scale (the first value pins the scale)."""
    rows = []
    for s_byte in range(0x30, 0x60):
        s = torch.tensor([s_byte], dtype=torch.uint8).view(torch.float8_e4m3fn).float().item()
        for mid in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
            row = torch.zeros(512)
            row[0] = 6.0 * s
            row[1] = mid * s
            rows.append(row)
    x = torch.stack(rows)
    keep = (x.to(torch.bfloat16).float() == x).all(-1)
    return x[keep].to(torch.bfloat16)


def scale_ties():
    """Rows whose amax / 6 is an exact midpoint between two adjacent e4m3 values (bf16-representable)."""
    vals = torch.arange(0, 0x7F, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()
    vals = vals[torch.isfinite(vals)]
    mids = ((vals[:-1] + vals[1:]) / 2)
    amax = (mids * 6.0)
    amax = amax[(amax.to(torch.bfloat16).float() == amax) & (amax >= 6 * 2.0 ** -9) & (amax <= 6 * 448.0)]
    x = torch.zeros(amax.numel(), 512)
    x[:, 0] = amax
    x[:, 1] = amax * 0.3
    x[:, 17] = -amax
    return x.to(torch.bfloat16)


# ---- the reader design: an FP4 record transcoded to the 584-byte V4 row v2.6 stores ---------------------------------
V4_DATA, V4_SCALE = 576, 8


def ue8m0(x):
    """SGLang fp8::cast_to_ue8m0: the biased exponent of x, plus one when x has a mantissa (x > 0, finite)."""
    bits = x.contiguous().view(torch.int32)
    return ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)


def v4_rows(values):
    """The 584-byte V4 row SGLang's c1 / c2 V4 path stores for fake-quantized latents `values` [n, 512] (fp32 or
    bf16, exact): nope 448 as e4m3 with one ue8m0 per 64 (cast_to_ue8m0(max(1e-4, amax) / 448), values times the
    exact power-of-two inverse, RN, clipped to +-448), rope 64 as bf16, then 7 scale bytes and one pad byte (left 0).
    Returns (data [n, 576] u8, scales [n, 8] u8)."""
    v = values.float()
    n = v.shape[0]
    nope = v[:, :448].reshape(n, 7, 64)
    amax = nope.abs().amax(-1, keepdim=True)
    e = ue8m0(torch.clamp_min(amax, 1e-4) / 448.0)
    inv = ((254 - e) << 23).view(torch.float32)
    q = (nope * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8).reshape(n, 448)
    rope = v[:, 448:].to(torch.bfloat16).contiguous().view(torch.uint8).reshape(n, 128)
    scales = torch.zeros(n, V4_SCALE, dtype=torch.uint8)
    scales[:, :7] = e.reshape(n, 7).to(torch.uint8)
    return torch.cat([q, rope], dim=-1), scales


def transcode_fp4_to_v4(data, scales):
    """What the reader's producer stage would write into the V4 shared-memory row from one FP4 record: decode
    (exact) then the V4 row rule above. Equal to v4_rows(fake_quant_compressed_kv(x)) when the record is the model's
    rounding of x, i.e. the attention kernel sees exactly v2.6's bytes."""
    return v4_rows(decode_rows(data, scales).float())
