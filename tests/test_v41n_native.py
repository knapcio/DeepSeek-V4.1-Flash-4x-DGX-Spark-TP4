"""CPU checks of the v41n native reader (DSV41_V41C_READER=native; 2026-10-29).

The native reader builds the V4 operand bytes inside the attention kernels from the raw 288-byte V4.1 FP4 records
(runtime/flashinfer_v41c/include/flashinfer_v41c/sparse_mla_sm120/v41n_convert.cuh). The math warps then run the
stock code on those bytes, so the outputs are v2.6's iff the bytes are the V4 store rule's bytes. Checked here:

- arithmetic (exhaustive): a bit-level numpy emulation of v41n_nope_tile / v41n_rope_block (byte permutes with the
  exact PTX prmt semantics, f16 products with IEEE RN, satfinite RN-even e4m3, the exponent table) equals the IO-warp
  reader's rule (cuda_rule_rows / the per-element reference) for EVERY e2m1 code x EVERY finite e4m3 scale byte x
  EVERY tile exponent in the reachable range, for the nope bytes and the rope bf16; with and without f16 flush-to-zero;
- exponent table: E[maxcode][|s|] equals the reference exponent of every block amax, is monotone in amax (so the
  tile exponent is the max of its block exponents), and the thermometer max-code trick equals max(code & 7) on
  every pair of code bytes and on random words;
- records: the whole converter (unit decomposition, in-place staging in a 464-byte row, load-all-then-store order,
  masked records) equals the IO-warp reader's shared-memory rows on random, heavy-tail, e2m1-tie and e4m3-scale-tie
  records, and the torch reference (v4_rows o decode_records);
- layout / ABI: the decode shared-memory layout keeps every stock offset and fits 99 KB; the raw record fits a V4 row;
  the prefill rope buffer + table fit the idle Q group region; the bulk-copy addresses are the FP4 page ABI the
  writer uses (gather_records); every copy is 16-byte aligned;
- dispatch: DSV41_V41C_READER=native routes decode / prefill to the native entries (prefill only for the shapes
  v41n_prefill serves, otherwise the group path); unset / group / io never touch them;
- flag off == 942f0ec: reader(), decode(), prefill() minus the native branch and every other function of v41c.py are
  942f0ec's text; the binding change is insert-only; the generated prefill header is the generator's output and the
  generator's edits are the documented six (the io decode SASS identity is in tests/v41n_nvcc_check.sh).

usage: python3 tests/test_v41n_native.py      (torch + numpy: ds-b3-cutecheck:1; git for the 942f0ec checks)
"""
import ast
import difflib
import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V41C = ROOT / "runtime" / "flashinfer_v41c"
BASE_COMMIT = "942f0ec"

try:
    import numpy as np
    import torch
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v = _load("v41c_native_under_test", V41C / "v41c.py")
P = _load("v41c_pool_native_under_test", ROOT / "adapter" / "v41c_pool.py")
if HAVE_TORCH:
    os.environ.setdefault("TRITON_INTERPRET", "1")
    F = _load("test_v41c_fast_helpers", ROOT / "tests" / "test_v41c_fast.py")


def _env(**kw):
    for k, val in kw.items():
        if val is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = val
    v.reader.cache_clear()
    v.enabled.cache_clear()
    v.native_receipt.cache_clear()


def _git_show(rel):
    try:
        return subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:{rel}"], capture_output=True,
                              text=True, check=True).stdout
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ bit emulation
U32 = np.uint32 if HAVE_TORCH else None


def prmt(a, b, sel):
    """PTX prmt.b32 (default mode): byte n of the result = byte (sel nibble n & 7) of {b:a}; nibble bit 3 replicates
    that byte's sign bit. numpy arrays (broadcast)."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    sel = np.asarray(sel, dtype=np.uint64)
    src = (b << np.uint64(32)) | a
    out = np.zeros(np.broadcast(a, b, sel).shape, dtype=np.uint64)
    for n in range(4):
        s = (sel >> np.uint64(4 * n)) & np.uint64(15)
        byte = (src >> ((s & np.uint64(7)) * np.uint64(8))) & np.uint64(0xFF)
        byte = np.where(s & np.uint64(8), np.where(byte & np.uint64(0x80), np.uint64(0xFF), np.uint64(0)), byte)
        out |= byte << np.uint64(8 * n)
    return out.astype(np.uint32)


def e4m3_satfinite_from_f16(h):
    """cvt.rn.satfinite.e4m3x2.f16x2 on one half: RN-even, |x| >= 448 (incl. inf) -> +-448, NaN -> 0x7F."""
    h = np.asarray(h, dtype=np.float16)
    x = h.astype(np.float32)
    nan = np.isnan(x)
    x = np.where(nan, np.float32(0), np.clip(x, np.float32(-448.0), np.float32(448.0)))
    b = F.e4m3_oracle(x)
    return np.where(nan, np.uint8(0x7F), b).astype(np.uint8)


def f16_ftz(h):
    h = np.asarray(h, dtype=np.float16)
    return np.where(np.abs(h.astype(np.float32)) < np.float32(2.0 ** -14), np.copysign(np.float16(0), h), h).astype(
        np.float16)


def bf16_bits(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint32) & 0xFFFF


MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32) if HAVE_TORCH else None


def e4m3_decode(b):
    """e4m3fn byte -> fp32 with the device's encodings: 0x7F / 0xFF are NaN (no infinity in e4m3fn)."""
    b = np.asarray(b, dtype=np.int64)
    v = F.e4m3_value(b)
    return np.where((b & 127) == 127, np.float32(np.nan), v).astype(np.float32)


def ref_exponent(amax):
    """The IO-warp reader's tile exponent byte (v41c_store_v4_row): fp32 IEEE division, ceil to a power of two.
    fmaxf semantics: a NaN amax yields max(1e-4, NaN) = 1e-4."""
    amax = np.asarray(amax, dtype=np.float32)
    sc = (np.fmax(np.float32(1e-4), amax) / np.float32(448.0)).astype(np.float32)
    u = sc.view(np.uint32).astype(np.int64)
    return (((u >> 23) & 0xFF) + ((u & 0x7FFFFF) != 0)).astype(np.int64)


def etab():
    """V41N exponent table [8, 128] (v41n_fill_etab), NaN entry 0x7F included as the device computes it."""
    mc = np.arange(8)[:, None]
    s = np.arange(128)[None, :]
    with np.errstate(invalid="ignore"):
        return ref_exponent(MAG[mc] * e4m3_decode(s)).astype(np.int64)


def therm_max_code(w0, w1):
    """v41n_max_code: thermometer bytes by byte permute, or-ed, folded, popcount."""
    th_lo, th_hi = np.uint32(0x07030100), np.uint32(0x7F3F1F0F)
    m0 = np.asarray(w0, dtype=np.uint32) & np.uint32(0x77777777)
    m1 = np.asarray(w1, dtype=np.uint32) & np.uint32(0x77777777)
    t = prmt(th_lo, th_hi, m0) | prmt(th_lo, th_hi, m0 >> np.uint32(16)) | prmt(th_lo, th_hi, m1) | \
        prmt(th_lo, th_hi, m1 >> np.uint32(16))
    t = t | (t >> np.uint32(16))
    t = t | (t >> np.uint32(8))
    t = (t & np.uint32(0xFF)).astype(np.int64)
    return np.array([bin(int(x)).count("1") for x in t.ravel()], dtype=np.int64).reshape(t.shape)


C16 = (np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float16) if HAVE_TORCH else None)


def nope_tables(sbyte, e, ftz=False):
    """(t_lo, t_hi) u32 tables of v41n_nope_tile for scale byte(s) and tile exponent(s) (broadcast)."""
    inv = ((254 - np.asarray(e, dtype=np.int64)) << 23).astype(np.uint32).view(np.float32)
    # fminf(fmaxf(x, -65504), 65504): fmax / fmin semantics (a NaN x gives -65504, as on the device)
    m = np.fmin(np.fmax((e4m3_decode(sbyte) * inv).astype(np.float32), np.float32(-65504)),
                np.float32(65504)).astype(np.float32)
    m16 = m.astype(np.float16)
    if ftz:
        m16 = f16_ftz(m16)
    prods = (C16[None, :].astype(np.float16) * np.asarray(m16).reshape(-1, 1)).astype(np.float16)
    if ftz:
        prods = f16_ftz(prods)
    b = e4m3_satfinite_from_f16(prods).astype(np.uint32)                       # [n, 8]
    t_lo = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)
    t_hi = b[:, 4] | (b[:, 5] << 8) | (b[:, 6] << 16) | (b[:, 7] << 24)
    return t_lo.astype(np.uint32), t_hi.astype(np.uint32)


def nope_words(x, t_lo, t_hi):
    """The two output words of one code word x (v41n_nope_tile inner loop)."""
    x = np.asarray(x, dtype=np.uint32)
    mag = x & np.uint32(0x77777777)
    x4 = (x.astype(np.uint64) << np.uint64(4)).astype(np.uint64) & np.uint64(0xFFFFFFFF)
    x4 = x4.astype(np.uint32)
    lo = prmt(t_lo, t_hi, mag) ^ (prmt(x, x4, 0x1504) & np.uint32(0x80808080))
    hi = prmt(t_lo, t_hi, mag >> np.uint32(16)) ^ (prmt(x, x4, 0x3726) & np.uint32(0x80808080))
    return lo, hi


def nope_tile(w, sw, E, ftz=False):
    """v41n_nope_tile: w [n, 8] u32, sw [n] u32 -> (bytes [n, 64] u8, e [n])."""
    n = w.shape[0]
    e = np.zeros(n, dtype=np.int64)
    for b in range(4):
        mc = therm_max_code(w[:, 2 * b], w[:, 2 * b + 1])
        e = np.maximum(e, E[mc, (sw >> np.uint32(8 * b)) & np.uint32(0x7F)])
    out = np.zeros((n, 16), dtype=np.uint32)
    for b in range(4):
        sb = (sw >> np.uint32(8 * b)) & np.uint32(0xFF)
        t_lo, t_hi = nope_tables(sb, e, ftz)
        for k in range(2):
            lo, hi = nope_words(w[:, 2 * b + k], t_lo, t_hi)
            out[:, 4 * b + 2 * k], out[:, 4 * b + 2 * k + 1] = lo, hi
    return out.view(np.uint8).reshape(n, 64), e


def rope_block(w0, w1, s):
    """v41n_rope_block: (w0, w1, s) [n] -> 16 bf16 bit patterns [n, 16]."""
    sv = F.e4m3_value(s).astype(np.float32)
    p = []
    for j in range(4):
        lo = bf16_bits(MAG[2 * j] * sv)
        hi = bf16_bits(MAG[2 * j + 1] * sv)
        p.append((lo | (hi << 16)).astype(np.uint32))
    lo_lo, lo_hi = prmt(p[0], p[1], 0x6420), prmt(p[2], p[3], 0x6420)
    hi_lo, hi_hi = prmt(p[0], p[1], 0x7531), prmt(p[2], p[3], 0x7531)
    out = []
    for x in (np.asarray(w0, dtype=np.uint32), np.asarray(w1, dtype=np.uint32)):
        mag = x & np.uint32(0x77777777)
        x4 = ((x.astype(np.uint64) << np.uint64(4)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        for h in range(2):
            sel = (mag >> np.uint32(16)) if h else mag
            l = prmt(lo_lo, lo_hi, sel)
            g = prmt(hi_lo, hi_hi, sel)
            out.append(prmt(l, g, 0x5140) ^ (prmt(x, x4, 0x2060 if h else 0x0040) & np.uint32(0x80008000)))
            out.append(prmt(l, g, 0x7362) ^ (prmt(x, x4, 0x3070 if h else 0x1050) & np.uint32(0x80008000)))
    words = np.stack(out, axis=1).astype(np.uint32)                            # [n, 8]
    return np.stack([words & np.uint32(0xFFFF), words >> np.uint32(16)], axis=2).reshape(-1, 16)


def convert_rows(data, scales, valid=None, ftz=False):
    """The whole converter on staged records, emulating the 464-byte shared-memory rows in place: every unit loads
    first (from the staged bytes), then stores. Returns the V4 row (data [n, 576], scales [n, 8]) as the math reads
    it (nope bytes at row 0..447 + rope from the rope buffer, ue8m0 bytes 0..6, pad byte 7)."""
    n = data.shape[0]
    stride = 464
    rows = np.zeros((n, stride), dtype=np.uint8)
    rows[:, :256] = data
    rows[:, 256:288] = scales
    if valid is None:
        valid = np.ones(n, dtype=bool)
    E = etab()
    # load phase (all units)
    nope_in = []
    for t in range(7):
        w = rows[:, 32 * t:32 * t + 32].copy().view(np.uint32).reshape(n, 8)
        sw = rows[:, 256 + 4 * t:256 + 4 * t + 4].copy().view(np.uint32).reshape(n)
        nope_in.append((np.where(valid[:, None], w, 0).astype(np.uint32), np.where(valid, sw, 0).astype(np.uint32)))
    rope_in = []
    s_last = rows[:, 284:288].copy().view(np.uint32).reshape(n)
    for blk in range(4):
        w = rows[:, 224 + 8 * blk:224 + 8 * blk + 8].copy().view(np.uint32).reshape(n, 2)
        s = (s_last >> np.uint32(8 * blk)) & np.uint32(0xFF)
        rope_in.append((np.where(valid[:, None], w, 0).astype(np.uint32), np.where(valid, s, 0).astype(np.uint32)))
    # store phase
    sc = np.zeros((n, 8), dtype=np.uint8)
    rope = np.zeros((n, 64), dtype=np.uint32)
    for t, (w, sw) in enumerate(nope_in):
        b, e = nope_tile(w, sw, E, ftz)
        rows[:, 64 * t:64 * t + 64] = b
        sc[:, t] = e.astype(np.uint8)
    for blk, (w, s) in enumerate(rope_in):
        rope[:, 16 * blk:16 * blk + 16] = rope_block(w[:, 0], w[:, 1], s)
    rope_bytes = np.stack([rope & 255, rope >> 8], axis=2).reshape(n, 128).astype(np.uint8)
    return np.concatenate([rows[:, :448], rope_bytes], axis=1), sc


# ------------------------------------------------------------------------------------------------ arithmetic
FINITE = [b for b in range(256) if b not in (0x7F, 0xFF)] if HAVE_TORCH else []


@unittest.skipUnless(HAVE_TORCH, "needs torch + numpy")
class Arithmetic(unittest.TestCase):
    def test_exhaustive_nope_element(self):
        """Every e2m1 code x every finite e4m3 scale byte x every tile exponent the table can produce: the native
        byte == the V4 rule's byte (clip(value * 2^(127-e), +-448), RN-even e4m3). With and without f16 FTZ."""
        E = etab()
        es = np.arange(int(E.min()), int(E.max()) + 1)
        codes = np.arange(16)
        S, EE = np.meshgrid(np.array(FINITE), es, indexing="ij")
        S, EE = S.ravel().astype(np.uint32), EE.ravel()
        for ftz in (False, True):
            t_lo, t_hi = nope_tables(S, EE, ftz)
            for code in codes:
                word = np.full(S.shape, code * 0x11111111, dtype=np.uint64).astype(np.uint32)
                lo, _ = nope_words(word, t_lo, t_hi)
                got = (lo & np.uint32(0xFF)).astype(np.uint8)
                val = (np.where(code & 8, -MAG[code & 7], MAG[code & 7]).astype(np.float32)
                       * F.e4m3_value(S).astype(np.float32)).astype(np.float32)
                inv = ((254 - EE) << 23).astype(np.uint32).view(np.float32)
                want = F.e4m3_oracle(np.clip((val * inv).astype(np.float32), np.float32(-448), np.float32(448)))
                bad = np.nonzero(got != want)[0]
                with self.subTest(code=int(code), ftz=ftz):
                    self.assertEqual(bad.size, 0, f"first: s={S[bad[:3]]} e={EE[bad[:3]]} got {got[bad[:3]]} "
                                                  f"want {want[bad[:3]]}")
        self.assertGreater(len(es), 20)

    def test_exhaustive_rope_element(self):
        """Every e2m1 code x every finite e4m3 scale byte: rope bf16 == bf16(value) (RN; exact here), sign of zero
        included."""
        S = np.array(FINITE, dtype=np.uint32)
        for code in range(16):
            word = np.full(S.shape, code * 0x11111111, dtype=np.uint64).astype(np.uint32)
            got = rope_block(word, word, S)
            val = (np.where(code & 8, -MAG[code & 7], MAG[code & 7]).astype(np.float32)
                   * F.e4m3_value(S).astype(np.float32)).astype(np.float32)
            want = bf16_bits(val)
            with self.subTest(code=code):
                self.assertTrue(np.array_equal(got, np.repeat(want[:, None], 16, axis=1)))

    def test_nan_scale_bytes_are_outside_the_contract(self):
        """Finite-only contract: the scale bytes 0x7F / 0xFF (e4m3fn NaN) are not equivalent (the IO rule's fminf turns
        NaN into +448; the native table into -448 / -0), and the model's writer never emits them (satfinite e4m3 of a
        non-negative amax / 6: bytes 0x00..0x7E). Documented in DESIGN.md section 5."""
        E = etab()
        e = int(E[7, 0x7F])
        t_lo, t_hi = nope_tables(np.array([0x7F], dtype=np.uint32), np.array([e]))
        lo, _ = nope_words(np.array([0x77777777], dtype=np.uint32), t_lo, t_hi)
        self.assertEqual(int(lo[0]) & 0xFF, 0xFE)                       # native: -448
        r = F._ref()
        x = torch.cat([r.latents(512, 1), r.latents(512, 2, tail=True), r.value_ties(), r.scale_ties()])
        _, scales = r.encode_rows(x)
        self.assertLessEqual(int(scales.max()), 0x7E)

    def test_exponent_table_monotone(self):
        """E[mc][s] == the reference exponent of amax = e2m1(mc) * |e4m3(s)|, for all 1024 entries, and the exponent
        is monotone in amax: the tile exponent is the max of the block exponents."""
        E = etab()
        amax = (MAG[:, None] * np.abs(F.e4m3_value(np.arange(127))[None, :])).astype(np.float32)
        self.assertTrue(np.array_equal(E[:, :127], ref_exponent(amax)))     # every finite entry
        self.assertTrue(np.all(E[:, 127] == ref_exponent(np.float32(0.0))))  # NaN entry: fmaxf(1e-4, NaN)
        order = np.argsort(amax.ravel(), kind="stable")
        self.assertTrue(np.all(np.diff(E[:, :127].ravel()[order]) >= 0))
        # negative scale bytes index the table by |s|: same exponent as the IO reader's |value| amax
        neg = ref_exponent(MAG[:, None] * np.abs(F.e4m3_value(np.arange(128, 255))[None, :]))
        self.assertTrue(np.array_equal(neg, E[:, :127]))

    def test_max_code(self):
        """The thermometer max-code trick == max(code & 7) on every pair of code bytes and 2^18 random word pairs."""
        b = np.arange(65536, dtype=np.uint32)
        w = (b & np.uint32(0xFF)) | ((b >> np.uint32(8)) << np.uint32(16))
        codes = np.stack([(w >> np.uint32(4 * i)) & np.uint32(7) for i in range(8)], axis=1).max(axis=1)
        self.assertTrue(np.array_equal(therm_max_code(w, np.zeros_like(w)), codes.astype(np.int64)))
        rng = np.random.default_rng(3)
        w0 = rng.integers(0, 2 ** 32, 1 << 18, dtype=np.uint64).astype(np.uint32)
        w1 = rng.integers(0, 2 ** 32, 1 << 18, dtype=np.uint64).astype(np.uint32)
        want = np.maximum(np.stack([(w0 >> np.uint32(4 * i)) & np.uint32(7) for i in range(8)], 1).max(1),
                          np.stack([(w1 >> np.uint32(4 * i)) & np.uint32(7) for i in range(8)], 1).max(1))
        self.assertTrue(np.array_equal(therm_max_code(w0, w1), want.astype(np.int64)))

    def test_random_tiles_exponent(self):
        """Random tiles with mixed-magnitude blocks (incl. all-zero blocks and scale 0): tile exponent == reference."""
        rng = np.random.default_rng(7)
        n = 1 << 16
        w = rng.integers(0, 2 ** 32, (n, 8), dtype=np.uint64).astype(np.uint32)
        w[rng.random((n, 8)) < 0.2] = 0
        sb = rng.choice(np.array(FINITE, dtype=np.uint32), (n, 4))
        sw = (sb[:, 0] | (sb[:, 1] << 8) | (sb[:, 2] << 16) | (sb[:, 3] << 24)).astype(np.uint32)
        _, e = nope_tile(w, sw, etab())
        codes = np.stack([(w[:, i // 8] >> np.uint32(4 * (i % 8))) & np.uint32(15) for i in range(64)], axis=1)
        vals = (np.where(codes & 8, -MAG[codes & 7], MAG[codes & 7])
                * np.repeat(F.e4m3_value(sb), 16, axis=1)).astype(np.float32)
        self.assertTrue(np.array_equal(e, ref_exponent(np.abs(vals).max(axis=1))))


# ------------------------------------------------------------------------------------------------ records
@unittest.skipUnless(HAVE_TORCH, "needs torch + numpy")
class Records(unittest.TestCase):
    def _sets(self):
        r = F._ref()
        return {"random": r.latents(512, 1), "tail": r.latents(512, 2, tail=True), "value_ties": r.value_ties(),
                "scale_ties": r.scale_ties()}, r

    def test_converter_equals_io_reader_and_reference(self):
        sets, r = self._sets()
        for name, x in sets.items():
            data, scales = r.encode_rows(x)
            d_ref, s_ref = v.v4_rows(v.decode_records(data, scales))
            d_io, s_io = F.cuda_rule_rows(data.numpy(), scales.numpy())
            for ftz in (False, True):
                d_n, s_n = convert_rows(data.numpy(), scales.numpy(), ftz=ftz)
                with self.subTest(set=name, ftz=ftz):
                    self.assertTrue(np.array_equal(d_n, d_io))
                    self.assertTrue(np.array_equal(s_n[:, :7], s_io[:, :7]))
                    self.assertTrue(np.array_equal(d_n, d_ref.numpy()))
                    self.assertTrue(np.array_equal(s_n, s_ref.numpy()))

    def test_arbitrary_bytes(self):
        """Records of arbitrary bytes (every finite scale byte, either sign; codes uniform) == the IO-warp rule."""
        rng = np.random.default_rng(11)
        n = 4096
        data = rng.integers(0, 256, (n, 256), dtype=np.uint8)
        scales = rng.choice(np.array(FINITE, dtype=np.uint8), (n, 32))
        d_io, s_io = F.cuda_rule_rows(data, scales)
        d_n, s_n = convert_rows(data, scales)
        self.assertTrue(np.array_equal(d_n, d_io))
        self.assertTrue(np.array_equal(s_n[:, :7], s_io[:, :7]))

    def test_masked_records(self):
        """A masked record converts as the all-zero record: zeros, ue8m0 of max(1e-4, 0) / 448 (the IO reader's
        masked row), whatever bytes were staged (the IO warp copies record 0 for it)."""
        rng = np.random.default_rng(5)
        data = rng.integers(0, 256, (8, 256), dtype=np.uint8)
        scales = rng.choice(np.array(FINITE, dtype=np.uint8), (8, 32))
        valid = np.array([1, 0, 1, 0, 0, 1, 1, 0], dtype=bool)
        d_n, s_n = convert_rows(data, scales, valid=valid)
        d_ok, s_ok = convert_rows(data[valid], scales[valid])
        self.assertTrue(np.array_equal(d_n[valid], d_ok))
        self.assertTrue(np.array_equal(s_n[valid], s_ok))
        e0 = int(ref_exponent(np.float32(0.0)))
        self.assertTrue(np.all(d_n[~valid] == 0))
        self.assertTrue(np.all(s_n[~valid][:, :7] == e0) and np.all(s_n[~valid][:, 7] == 0))


# ------------------------------------------------------------------------------------------------ layout / ABI
class Layout(unittest.TestCase):
    def test_decode_smem(self):
        """DecodeDsv4Smem: stock offsets unchanged (v41n fields appended), total + static <= 99 KB, raw record fits a
        464-byte row, copies 16-byte aligned."""
        HPB, D_ROPE, QNS, NS, BI, KVS, SC = 16, 64, 464, 7, 64, 464, 8
        off_q_fp8 = HPB * D_ROPE * 2
        off_q_sc = off_q_fp8 + HPB * QNS
        off_kv_fp8 = off_q_sc + HPB * NS * 4
        off_kv_sc = off_kv_fp8 + 2 * BI * KVS
        off_kv_rope = off_kv_sc + 2 * BI * SC
        off_mbar_full = -(-(off_kv_rope + 2 * BI * D_ROPE * 2) // 16) * 16
        off_reduce = off_mbar_full + 32
        off_w_head = off_reduce + 2 * 8 * HPB * 4
        off_w_fp8 = off_w_head + NS * HPB * 4
        end_stock = off_w_fp8 + 2 * HPB * (BI + 16)
        off_conv = -(-end_stock // 16) * 16
        total = off_conv + 16 + 16 + 1024
        self.assertGreaterEqual(off_conv, end_stock)
        self.assertLessEqual(total + HPB * BI * 2, 101376)
        self.assertGreaterEqual(KVS, 288)
        for e in range(64):
            self.assertEqual((off_kv_fp8 + e * KVS) % 16, 0)
            self.assertEqual((off_kv_fp8 + e * KVS + 256) % 16, 0)
        src = (V41C / "include" / "flashinfer_v41c" / "sparse_mla_sm120" / "decode_dsv4_kernel.cuh").read_text()
        self.assertIn("OFF_V41N_MBAR_CONV = (OFF_W_FP8 + 2 * SMEM_W_FP8_BUF + 15) / 16 * 16", src)

    def test_emulated_constants_are_the_kernel_constants(self):
        """The permute selectors, thermometer bytes and f16 magnitude pairs the emulation uses are the ones in
        v41n_convert.cuh (the emulation is only evidence for the kernel if they agree)."""
        src = (V41C / "include" / "flashinfer_v41c" / "sparse_mla_sm120" / "v41n_convert.cuh").read_text()
        for lit in ("0x1504u", "0x3726u", "0x5140u", "0x7362u", "0x0040u", "0x1050u", "0x2060u", "0x3070u",
                    "0x6420u", "0x7531u", "0x07030100u", "0x7F3F1F0Fu", "0x77777777u", "0x80808080u", "0x80008000u",
                    "__ushort_as_half(0x0000), __ushort_as_half(0x3800)",
                    "__ushort_as_half(0x3C00), __ushort_as_half(0x3E00)",
                    "__ushort_as_half(0x4000), __ushort_as_half(0x4200)",
                    "__ushort_as_half(0x4400), __ushort_as_half(0x4600)",
                    "fminf(fmaxf(v41n_e4m3_value(sw >> (8 * b)) * inv, -65504.0f), 65504.0f)",
                    "__fdiv_rn(fmaxf(1e-4f, amax), 448.0f)", "(254u - e) << 23", "__NV_SATFINITE, __NV_E4M3",
                    "etab[(mc << 7) | ((sw >> (8 * b)) & 0x7Fu)]", "row + 224 + 8 * blk", "row + V41N_DATA_BYTES + 28",
                    "row + 32 * t", "row + V41N_DATA_BYTES + 4 * t"):
            with self.subTest(lit=lit):
                self.assertIn(lit, src)
        if HAVE_TORCH:
            self.assertEqual([float(x) for x in C16], [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    def test_prefill_smem(self):
        """The rope buffer (64 x 64 bf16) + exponent table fit the idle second Q group region (BF16 MG layout)."""
        q_nope = 16 * (448 + 8) * 2
        self.assertLessEqual(64 * 64 * 2 + 1024, q_nope)

    def test_fp4_page_abi(self):
        """The bulk-copy source addresses (codes: page * stride + local * 256; scales: + pbs * 256 + local * 32) are
        the writer's page ABI (gather_records) for both page sizes, and 16-byte aligned."""
        if not HAVE_TORCH:
            self.skipTest("needs torch")
        for pbs in (128, 256):
            pages = 3
            raw = torch.randint(0, 256, (pages, pbs * 288), dtype=torch.uint8)
            view = torch.as_strided(raw, (pages, pbs, 1, 288), (pbs * 288, 288, 288, 1))
            tok = torch.arange(pages * pbs)
            d, s = v.gather_records(view, tok)
            flat = raw.reshape(-1)
            for t in (0, 1, pbs - 1, pbs, 2 * pbs + 17):
                page, local = t // pbs, t % pbs
                a = page * pbs * 288 + local * 256
                b = page * pbs * 288 + pbs * 256 + local * 32
                self.assertTrue(torch.equal(flat[a:a + 256], d[t]))
                self.assertTrue(torch.equal(flat[b:b + 32], s[t]))
                self.assertEqual(a % 16, 0)
                self.assertEqual(b % 16, 0)

    def test_prefill_generator(self):
        """The committed native prefill header is the generator's output; the generator's six edits each match the
        vendored FlashInfer source exactly once."""
        gen = _load("gen_v41n_prefill_t", V41C / "gen_v41n_prefill.py")
        out = gen.generate(gen.SRC.read_text())
        self.assertEqual(out, gen.DST.read_text())
        self.assertEqual(sorted({tag for tag, _, _ in gen.EDITS}), ["E1", "E2", "E3", "E4", "E5", "E6"])
        self.assertEqual([(t, n) for t, _, _, n in gen.REPLACE_ALL], [("E7", 3)])
        self.assertNotIn(" mbarrier_wait_parity(", out.split("v41n_prefill_mg_impl(", 1)[1])

    def test_sync_and_proxy_fence_in_source(self):
        """P1-1: every native conversion ends with fence.proxy.async.shared::cta before the release that lets a bulk
        copy refill the rows; the native sync points use the clobbered helpers."""
        inc = V41C / "include" / "flashinfer_v41c" / "sparse_mla_sm120"
        conv = (inc / "v41n_convert.cuh").read_text()
        self.assertIn('asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");', conv)
        for helper in ("v41n_mbar_arrive(", "v41n_mbar_arrive_expect_tx(", "v41n_mbar_wait_parity(", "v41n_bulk_g2s("):
            body = conv.split(f"__device__ __forceinline__ void {helper}", 1)[1].split("\n}\n", 1)[0]
            self.assertIn('"memory"', body, helper)
        dec = (inc / "decode_dsv4_kernel.cuh").read_text()
        i = dec.index("v41n_convert_chunk<KV_SMEM_STRIDE, SCALE_BYTES_PER_TOKEN>(")
        j = dec.index("v41n_mbar_arrive(sm.v41n_mbar_conv(conv_idx));")
        self.assertIn("v41n_fence_proxy_async_smem();", dec[i:j])
        pf = (inc / "v41n_prefill_kernel.cuh").read_text()
        body = pf.split("__device__ __forceinline__ void v41n_convert_warp_records(", 1)[1].split("\n}\n", 1)[0]
        self.assertLess(body.rindex("v41n_store_nope"), body.index("v41n_fence_proxy_async_smem();"))


# ------------------------------------------------------------------------------------------------ dispatch
class _T:
    """Shape / dtype tensor stand-in."""

    def __init__(self, shape, contiguous=True, dtype="torch.int32"):
        self.shape = tuple(shape)
        self._c = contiguous
        self.dtype = dtype

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self._c


class Dispatch(unittest.TestCase):
    def tearDown(self):
        _env(DSV41_V41C_READER=None, DSV41_KV_LAYOUT=None)

    def test_reader_values(self):
        for raw, want in (("", "group"), ("group", "group"), ("io", "io"), ("native", "native")):
            _env(DSV41_V41C_READER=raw)
            self.assertEqual(v.reader(), want)
            os.environ["DSV41_V41C_READER"] = raw
            self.assertEqual(P.reader_mode(), want)
        _env(DSV41_V41C_READER="nativ")
        with self.assertRaises(ValueError):
            v.reader()
        with self.assertRaises(ValueError):
            P.reader_mode()

    def _routes(self, mode):
        calls = []
        saved = {k: getattr(v, k) for k in ("native_decode", "native_prefill")}
        v.native_decode = lambda *a, **k: calls.append("native_decode") or "N"
        v.native_prefill = lambda *a, **k: calls.append("native_prefill") or "N"
        try:
            _env(DSV41_V41C_READER=mode)
            kw = dict(topk_length=None, attn_sink=None, extra_kv=None, extra_indices=None, extra_topk_length=None)
            try:
                v.decode(None, None, None, None, None, None, None, 1.0, **kw)
            except Exception:  # noqa: BLE001  (group / io need flashinfer: only the routing matters here)
                pass
            try:
                v.prefill(None, None, None, 1.0, topk_length=None, attn_sink=None, extra_fp4=None, extra_idx=None,
                          extra_topk_length=None, head_dim_v=512)
            except Exception:  # noqa: BLE001
                pass
        finally:
            for k, f in saved.items():
                setattr(v, k, f)
        return calls

    def test_routing(self):
        self.assertEqual(self._routes("native"), ["native_decode", "native_prefill"])
        for mode in ("", "group", "io"):
            with self.subTest(mode=mode):
                self.assertEqual(self._routes(mode), [])

    def test_prefill_supported(self):
        U8, BF = "torch.uint8", "torch.bfloat16"
        ok = dict(topk_length=_T((96,)), extra_fp4=_T((10, 256, 1, 288), dtype=U8), extra_idx=_T((96, 512)),
                  extra_topk_length=_T((96,)))
        q, kv, idx = _T((96, 16, 512), dtype=BF), _T((40, 64, 1, 584), dtype=U8), _T((96, 128))
        self.assertTrue(v.native_prefill_supported(q, kv, idx, **ok))
        q8k = _T((8192, 8, 512), dtype=BF)
        self.assertTrue(v.native_prefill_supported(q8k, kv, _T((8192, 128)),
                                                   **{**ok, "extra_fp4": _T((10, 128, 1, 288), dtype=U8),
                                                      "extra_idx": _T((8192, 512))}))
        bad = [
            (_T((96, 32, 512), dtype=BF), kv, idx, ok),                           # heads
            (q, kv, _T((96, 192)), ok),                                           # SWA width
            (q, _T((10, 256, 1, 584), dtype=U8), idx, ok),                        # main pages not split
            (q, kv, idx, {**ok, "extra_fp4": _T((10, 64, 1, 288), dtype=U8)}),    # FP4 page size
            (q, kv, idx, {**ok, "extra_fp4": _T((10, 256, 1, 584), dtype=U8)}),   # not FP4
            (q, kv, idx, {**ok, "topk_length": None}),
            (q, kv, idx, {**ok, "extra_topk_length": None}),
            (q, kv, _T((96, 128), contiguous=False), ok),
            (q, kv, idx, {**ok, "extra_idx": _T((96, 512), contiguous=False)}),
            (q, kv, idx, {**ok, "extra_idx": _T((96, 500))}),                     # not a multiple of 64
            (q, kv, idx, {**ok, "extra_idx": _T((95, 512))}),                     # rows
            (q, kv, idx, {**ok, "head_dim_v": 448}),                              # head_dim_v
            (_T((96, 16, 512), dtype="torch.float16"), kv, idx, ok),              # q dtype
            (q, kv, idx, {**ok, "extra_idx": _T((96, 512), dtype="torch.int64")}),
            (q, kv, idx, {**ok, "extra_topk_length": _T((96,), dtype="torch.int64")}),
            (q, kv, idx, {**ok, "extra_fp4": _T((10, 256, 1, 288), dtype="torch.int8")}),
        ]
        for i, (a, b, c, kw) in enumerate(bad):
            with self.subTest(case=i):
                self.assertFalse(v.native_prefill_supported(a, b, c, **kw))


class _FakeTensor:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    def dim(self):
        return len(self.shape)


class NativeDecodeSpeedPath(unittest.TestCase):
    """Night-run fixes (2026-10-29): padded head blocks skipped, conversion-skipped / all-heads timing diagnostics,
    SWA chunks waited on mbar_full directly."""

    def setUp(self):
        import types
        self.calls = []
        test = self

        class Runner:
            def get_cache_key_extras(self, inputs):
                return ()

            def __call__(self, inputs, tactic, **kw):
                test.calls.append({"tactic": tactic, "active_heads": kw.get("active_heads"),
                                   "diag_mode": kw.get("diag_mode")})

        S = types.SimpleNamespace(_decode_dsv4_map_to_token_bucket=lambda n: n)
        fi = types.ModuleType("flashinfer")
        mla = types.ModuleType("flashinfer.mla")
        mla._sparse_mla_sm120 = S
        fi.mla = mla
        self._mods = {k: sys.modules.get(k) for k in ("flashinfer", "flashinfer.mla")}
        sys.modules["flashinfer"], sys.modules["flashinfer.mla"] = fi, mla
        self._saved = {k: getattr(v, k) for k in ("_native_runner", "native_tactic", "native_receipt")}
        v._native_runner = lambda: Runner()
        v.native_tactic = lambda S_, inputs, kw, key: -1
        v.native_receipt = lambda: "ok"

    def tearDown(self):
        for k, f in self._saved.items():
            setattr(v, k, f)
        for k, m in self._mods.items():
            if m is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = m
        v.clear_heads()
        os.environ.pop("DSV41_V41N_DIAG", None)
        v.native_diag.cache_clear()

    def _decode(self, heads_tag=None, diag=None):
        if diag is None:
            os.environ.pop("DSV41_V41N_DIAG", None)
        else:
            os.environ["DSV41_V41N_DIAG"] = diag
        v.native_diag.cache_clear()
        v.tag_heads(heads_tag)
        BF, I32, U8 = "torch.bfloat16", "torch.int32", "torch.uint8"
        q = _FakeTensor((6, 64, 512), BF)
        v.native_decode(q, _FakeTensor((10, 64, 1, 584), U8), _FakeTensor((6, 128), I32), None, None,
                        _FakeTensor((6, 64, 512), BF), None, 0.04, topk_length=None, attn_sink=None,
                        extra_kv=_FakeTensor((10, 256, 1, 288), U8), extra_indices=_FakeTensor((6, 512), I32),
                        extra_topk_length=_FakeTensor((6,), I32))
        return self.calls[-1]

    def test_head_skip_and_diagnostics(self):
        self.assertEqual(self._decode(None), {"tactic": -1, "active_heads": 0, "diag_mode": 0})
        self.assertEqual(self._decode(16), {"tactic": -1, "active_heads": 16, "diag_mode": 0})
        self.assertEqual(self._decode(64)["active_heads"], 0)             # nothing padded
        self.assertEqual(self._decode(16, "allheads")["active_heads"], 0)
        self.assertEqual(self._decode(16, "noconv"), {"tactic": -1, "active_heads": 16, "diag_mode": 2})
        os.environ["DSV41_V41N_DIAG"] = "nope"
        v.native_diag.cache_clear()
        with self.assertRaises(ValueError):
            v.native_diag()

    def test_kernel_and_launcher_text(self):
        inc = V41C / "include" / "flashinfer_v41c" / "sparse_mla_sm120"
        dec = (inc / "decode_dsv4_kernel.cuh").read_text()
        # math: SWA chunks wait mbar_full, extra chunks the converters' barrier with a per-buffer parity bit
        self.assertIn("v41n_mbar_wait_parity(sm.v41n_mbar_conv(cons_idx), (conv_bits >> cons_idx) & 1u);", dec)
        self.assertIn("v41n_mbar_wait_parity(sm.mbar_full(cons_idx), cons_phase);", dec)
        # converters arrive only for extra chunks
        blk = dec.split("if (chunk_idx >= num_orig_chunks) {\n          if constexpr (!V41N_SKIP_CONVERT)", 1)[1]
        self.assertLess(blk.index("v41n_mbar_arrive(sm.v41n_mbar_conv(conv_idx));"), blk.index("++conv_idx;"))
        self.assertLess(blk.index("v41n_mbar_arrive(sm.v41n_mbar_conv(conv_idx));"),
                        blk.index("        }\n        ++conv_idx;"))
        launch = (V41C / "csrc" / "v41n_decode_dsv4.cu").read_text()
        # the split heuristic sees the full head count; only grid.y and the merge use the active heads
        self.assertIn("const int per_token_head = num_tokens * H_BLOCKS;", launch)
        self.assertIn("dim3 grid1(num_tokens, active_blocks < H_BLOCKS ? active_blocks : H_BLOCKS, num_splits);", launch)
        self.assertIn("dim3 grid2(num_tokens, active);", launch)

    def test_adapter_tags_heads_for_native(self):
        src = (ROOT / "adapter" / "v41c_pool.py").read_text()
        self.assertIn('rd.tag_heads(getattr(layer, "tp_q_head_num", None))', src)
        self.assertIn("rd.clear_heads()", src)


# ------------------------------------------------------------------------------------------------ flag off == 942f0ec
def _funcs(src):
    tree = ast.parse(src)
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def _strip_native_branch(src, name):
    """Function `name` without its leading `if reader() == "native": return ...` statement and docstring tail."""
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = [s for s in fn.body if not (isinstance(s, ast.If) and "native" in ast.dump(s.test))]
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]                                                      # docstring changed (mentions native)
    fn.body = body
    return ast.dump(fn)


class FlagOff(unittest.TestCase):
    def setUp(self):
        self.base = _git_show("runtime/flashinfer_v41c/v41c.py")
        if self.base is None:
            self.skipTest(f"git {BASE_COMMIT} not available")
        self.cur = (V41C / "v41c.py").read_text()

    def test_functions_unchanged(self):
        """Every function / class of 942f0ec's v41c.py is unchanged, except reader() (one new value), check_sources
        (new non-FlashInfer headers), decode / prefill (a leading native branch)."""
        a, b = _funcs(self.base), _funcs(self.cur)
        changed = sorted(k for k in a if a[k] != b.get(k))
        self.assertEqual(changed, ["check_sources", "decode", "prefill", "reader"])
        for name in ("decode", "prefill"):
            with self.subTest(fn=name):
                self.assertEqual(_strip_native_branch(self.base, name), _strip_native_branch(self.cur, name))

    def test_binding_insert_only(self):
        base = _git_show("runtime/flashinfer_v41c/csrc/v41c_binding.cu")
        cur = (V41C / "csrc" / "v41c_binding.cu").read_text()
        ops = {op for op, *_ in difflib.SequenceMatcher(a=base.splitlines(), b=cur.splitlines(),
                                                         autojunk=False).get_opcodes()}
        self.assertTrue(ops <= {"equal", "insert"}, ops)

    def test_io_sources_unchanged(self):
        """The io decode launcher is 942f0ec's file; the patched header only adds (the io SASS identity is checked
        by tests/v41n_nvcc_check.sh)."""
        rel = "runtime/flashinfer_v41c/csrc/v41c_decode_dsv4.cu"
        self.assertEqual(_git_show(rel), (ROOT / rel).read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
