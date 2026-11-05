"""V4.1 KV record numerics on CPU (DSV41_KV_LAYOUT=v41 gate, step 0).

Question: is storing the ratio-1 / ratio-2 compressed latents as 288-byte V4.1 FP4
records (e2m1 + per-16 e4m3 scales) the model's own rounding, i.e. lossless with
respect to what the model computes, and is the 528-byte V4.1 SWA record lossless
with respect to today's 584-byte V4 SWA record?

Ground truth for the compressed latent is SGLang's `fake_quant_compressed_kv`
(staged canary `kernels/ops/attention/dsv4/torch_quant.py`): the model itself rounds
each compressed latent to e2m1 with one e4m3 scale per 16 values, by IEEE division.
The fp8 cache layouts then store that fake-quantized value a second time.

Checks:
  1. A division-arithmetic V4.1 FP4 encoder reproduces the model's values exactly
     (decode(encode(x)) == fake_quant_compressed_kv(x), bit for bit): FP4 storage is
     lossless for the model.
  2. The vendored b12x_next reference / device arithmetic (scale by IEEE division,
     values multiplied by the rounded reciprocal of the decoded scale) is measured
     against the model on random and on constructed-tie inputs. Any difference is a
     deviation from the model's rounding; the test records the rate and requires the
     division encoder (1) to be the one shipped.
  3. Today's 584-byte V4 storage of the same fake-quantized values (448 fp8 nope with
     one power-of-two scale per 64, 64 bf16 rope) is measured against the model: its
     error is the double rounding V4.1 FP4 removes.
  4. SWA: the 528-byte V4.1 record (512 e4m3 incl. rope, ue8m0 per 32) against the
     584-byte V4 record (448 e4m3 per 64 + bf16 rope) on the same bf16 latents: NOT a
     model rounding (the model does not fake-quantize the SWA latent), so V4.1 SWA is a
     storage-format change; its error vs bf16 is reported for the quality gate.

Needs torch and the staged SGLang canary (SGLANG_CANARY_PY, default
../ds41-issue16/runtime/sglang-canary/python); SKIPs without them.
"""

import importlib.util
import os
import sys
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANARY = os.environ.get(
    "SGLANG_CANARY_PY",
    os.path.join(os.path.dirname(ROOT), "ds41-issue16", "runtime", "sglang-canary", "python"),
)
TQ_PATH = os.path.join(CANARY, "sglang", "kernels", "ops", "attention", "dsv4", "torch_quant.py")
REF_PATH = os.path.join(ROOT, "runtime", "b12x_next", "b12x_next", "attention", "_shared", "mla",
                        "compressed_reference.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e2m1_codes(v):
    """fp32 values (already scaled, |v| <= 6 after clamp) -> 4-bit codes, RN-even."""
    lut = torch.tensor(E2M1)
    mag = v.abs().clamp(max=6.0)
    # nearest grid value, ties to the even code (codes 0,2,4,6 are 'even' mantissa)
    d = (mag.unsqueeze(-1) - lut).abs()
    best = d.min(dim=-1).values.unsqueeze(-1)
    cand = (d == best)
    # among tied candidates prefer the even code
    idx = torch.arange(8).expand_as(cand)
    even_first = torch.where(cand & (idx % 2 == 0), idx, torch.full_like(idx, 99)).min(-1).values
    any_first = torch.where(cand, idx, torch.full_like(idx, 99)).min(-1).values
    code = torch.where(even_first < 99, even_first, any_first)
    return code | (torch.signbit(v).to(torch.int64) << 3)


def decode_codes(code, scale):
    lut = torch.tensor(E2M1)
    val = lut[code & 7]
    val = torch.where((code & 8) != 0, -val, val)
    return val * scale


def encode_fp4_division(x):
    """V4.1 FP4 record values by the model's arithmetic: scale = e4m3(max(amax/6, 2^-9)),
    values divided (IEEE) by the decoded scale, RN-even e2m1."""
    b = x.float().unflatten(-1, (-1, 16))
    amax = b.abs().amax(-1, keepdim=True)
    s = (amax * (1.0 / 6.0)).clamp(min=2.0 ** -9, max=448.0).to(torch.float8_e4m3fn).float()
    code = e2m1_codes((b / s).clamp(-6, 6))
    return code, s


def encode_fp4_b12x(x):
    """The b12x_next dsv41 writer arithmetic BEFORE this branch (kept to document the
    deviation): scale = e4m3(max(amax, 6*2^-9) / 6), values * fp32(1 / decoded) (a rounded
    reciprocal), RN-even e2m1."""
    b = x.float().unflatten(-1, (-1, 16))
    amax = b.abs().amax(-1, keepdim=True)
    s = (amax.clamp_min(6 * 2.0 ** -9) / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    inv = 1.0 / s
    code = e2m1_codes((b * inv).clamp(-6, 6))
    return code, s


def v4_store(values):
    """584-byte V4 record of a [n, 512] latent: nope 448 as e4m3 with one power-of-two
    scale per 64 (ceil_pow2(amax / 448)), rope 64 kept bf16. Returns the decoded fp32."""
    nope, rope = values[:, :448].float(), values[:, 448:]
    t = nope.unflatten(-1, (7, 64))
    amax = t.abs().amax(-1, keepdim=True).clamp_min(2.0 ** -126)
    e = torch.ceil(torch.log2(amax / 448.0))
    s = torch.pow(2.0, e)
    q = (t / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
    return torch.cat([q.flatten(-2), rope.to(torch.bfloat16).float()], dim=-1)


def v41_swa_store(values):
    """528-byte V4.1 SWA record: 512 e4m3 incl. rope, ue8m0 per 32 (SGLang
    quantize_k_cache_v41 rule: 2^ceil(log2(max(amax / 448, 1e-4))))."""
    t = values.float().unflatten(-1, (16, 32))
    s = torch.clamp_min(t.abs().amax(-1, keepdim=True) / 448.0, 1e-4)
    s = torch.pow(2.0, torch.ceil(torch.log2(s)))
    q = (t / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
    return q.flatten(-2)


def latents(n, seed, tail=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 512, generator=g)
    if tail:
        x = x * torch.exp(torch.randn(n, 512, generator=g) * 1.5)
    x[:, 448:] *= 3.0          # rope tail runs larger after rotation
    return x.to(torch.bfloat16)


@unittest.skipIf(torch is None or not os.path.exists(TQ_PATH), "needs torch and the staged SGLang canary")
class V41Numerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tq = _load("dsv4_torch_quant", TQ_PATH)

    def model(self, x):
        return self.tq.fake_quant_compressed_kv(x).float()

    def test_division_encoder_is_the_models_rounding(self):
        for seed, tail in ((1, False), (2, True), (3, True)):
            x = latents(4096, seed, tail)
            code, s = encode_fp4_division(x)
            got = decode_codes(code, s).flatten(-2)
            want = self.model(x)
            self.assertTrue(torch.equal(got, want), (seed, (got != want).sum().item()))

    @staticmethod
    def tie_latents():
        """336 latents whose second value is an exact e2m1 midpoint times a non-trivial e4m3
        scale (the first value pins amax / 6 to that scale)."""
        rows = []
        for s_byte in range(0x30, 0x60):                       # e4m3 scales around 2^-3..2^3
            s = torch.tensor([s_byte], dtype=torch.uint8).view(torch.float8_e4m3fn).float().item()
            for mid in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
                row = torch.zeros(16)
                row[0] = 6.0 * s                               # pins amax / 6 = s exactly
                row[1] = mid * s
                rows.append(row)
        x = torch.stack(rows)
        x = torch.cat([x, torch.zeros(x.shape[0], 512 - 16)], dim=-1)
        exact = x.to(torch.bfloat16).float() == x               # keep rows bf16 represents exactly
        keep = exact.all(-1)
        return x[keep].to(torch.bfloat16)

    def test_ties_division_vs_reciprocal(self):
        """Constructed exact ties: division lands on the midpoint and rounds to even like the
        model; the old rounded-reciprocal arithmetic can land off the midpoint."""
        x = self.tie_latents()
        keep = torch.ones(x.shape[0], dtype=torch.bool)
        want = self.model(x)
        d_code, d_s = encode_fp4_division(x)
        r_code, r_s = encode_fp4_b12x(x)
        div = decode_codes(d_code, d_s).flatten(-2)
        rec = decode_codes(r_code, r_s).flatten(-2)
        self.assertTrue(torch.equal(div, want))
        self.__class__.tie_rows = int(keep.sum())
        self.__class__.tie_mismatch_rows = int((rec != want).any(-1).sum())
        print(f"\n[v41 numerics] constructed ties: {self.tie_rows} rows; "
              f"b12x reciprocal arithmetic differs from the model on {self.tie_mismatch_rows}")

    def test_random_b12x_rate(self):
        x = latents(65536, 7, tail=True)
        want = self.model(x)
        r_code, r_s = encode_fp4_b12x(x)
        rec = decode_codes(r_code, r_s).flatten(-2)
        n = int((rec != want).sum())
        print(f"\n[v41 numerics] random bf16 latents: b12x reciprocal arithmetic differs on {n} of "
              f"{want.numel()} values")
        self.assertLess(n / want.numel(), 1e-3)

    def test_v4_double_rounding(self):
        x = latents(8192, 11, tail=True)
        want = self.model(x)                                   # the model's compressed latent
        v4 = v4_store(want)                                    # what today's 584-B pool returns
        diff = (v4 != want)
        rel = ((v4 - want).abs().sum() / want.abs().sum()).item()
        print(f"\n[v41 numerics] V4 (584 B) storage of the model's compressed latents: "
              f"{int(diff.sum())} of {want.numel()} values changed, relative L1 {rel:.2e}; "
              f"V4.1 FP4 (288 B): 0 (test_division_encoder_is_the_models_rounding)")
        self.assertGreater(int(diff.sum()), 0)

    def test_swa_formats_vs_bf16(self):
        x = latents(8192, 13, tail=True).float()
        v4 = v4_store(x)
        v41 = v41_swa_store(x)
        e4 = ((v4 - x).abs().sum() / x.abs().sum()).item()
        e41 = ((v41 - x).abs().sum() / x.abs().sum()).item()
        r4 = ((v4[:, 448:] - x[:, 448:]).abs().sum() / x[:, 448:].abs().sum()).item()
        r41 = ((v41[:, 448:] - x[:, 448:]).abs().sum() / x[:, 448:].abs().sum()).item()
        print(f"\n[v41 numerics] SWA vs bf16, relative L1: V4 584 B {e4:.2e} (rope {r4:.2e}), "
              f"V4.1 528 B {e41:.2e} (rope {r41:.2e})")
        self.assertGreater(r41, r4)        # V4.1 SWA quantizes the rope: a format change, not lossless

    def ref_records(self, x):
        ref = _load("b12x_compressed_reference", REF_PATH)
        n = x.shape[0]
        cache = ref.pack_deepseek_v41_cache_reference(x, page_size=256, cache_kind="indexed")
        self.assertEqual(cache.shape[1], 256 * 288)
        recs = cache.view(-1, 288)[:n]
        vals = ref.unpack_deepseek_v41_cache_reference(cache, page_size=256, cache_kind="indexed")[:n]
        return recs, vals

    def division_records(self, x):
        code, s = encode_fp4_division(x)
        code = code.reshape(x.shape[0], 512).to(torch.uint8)
        payload = code[:, 0::2] | (code[:, 1::2] << 4)
        scales = s.reshape(x.shape[0], 32).to(torch.float8_e4m3fn).view(torch.uint8)
        return torch.cat([payload, scales], dim=-1)

    def test_b12x_reference_is_the_models_rounding(self):
        """Step 1 of the v41c build: b12x_next's V4.1 indexed record encoder (CPU reference, the
        arithmetic its dsv41 device writer now mirrors: mul.rn amax*f32(1/6), clamp, e4m3,
        div.rn, e2m1) decodes to the model's fake-quantized latent bit for bit, and its bytes
        equal the division encoder's, on random latents and on the constructed ties."""
        xs = [latents(4096, 31, False), latents(4096, 32, True), self.tie_latents()]
        for x in xs:
            recs, vals = self.ref_records(x)
            self.assertTrue(torch.equal(vals, self.model(x)))
            self.assertTrue(torch.equal(recs, self.division_records(x)))

    def test_writer_source_uses_model_arithmetic(self):
        """The device writer's dsv41 branch spells the model's arithmetic (static check; the
        GPU parity test of the window kit proves it on hardware)."""
        with open(os.path.join(ROOT, "runtime", "b12x_next", "b12x_next", "attention", "_shared",
                               "mla", "kv_cache.py")) as fh:
            src = fh.read()
        branch = src[src.index("The model's own rounding of a V4.1 ratio-1/2"):]
        branch = branch[:branch.index("st_global_u64(dst + tid.to(Int64) * Int64(8), packed64)")]
        for needle in ("mul_rn_f32(group_amax, Float32(1.0 / 6.0))", "fmax_f32(scale_f32, Float32(2.0**-9))",
                       "fmin_f32(scale_f32, Float32(448.0))", "div_rn_f32(vals[i], decoded_scale)",
                       "quantize_and_pack_16_fast(vals, Float32(1.0))"):
            self.assertIn(needle, branch)
        self.assertNotIn("rcp_approx_ftz(decoded_scale)", branch.split("else:")[0])


if __name__ == "__main__":
    unittest.main()
