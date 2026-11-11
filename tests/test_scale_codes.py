"""CPU checks for routed-expert scale codes (adapter/scale_codes.py + scripts/b12x_next-scale-codes.patch).

1. pack / unpack: byte-exact round trip on the production TP4 shapes, adversarial spans (exactly 16, base 0,
   base 247), refusal of an expert spanning 17 values.
2. Layout: b12x_next's N64 in-place repack (`_grid_to_w4a8_n64_sfb_kernel`, ported) of real-shape scales, then
   every address each patched kernel reads, through a line-for-line emulation of its CODED branch
   (`w4a8_scale_codes.coded_stage_16b` / `coded_byte` with the reader's own index arithmetic), must return the
   native bytes / words the native branch reads:
     w4a8_phase1._stage_n64_sfb         (FC1, 16-byte transfers)
     w4a8_compact_projection._stage_n64_sfb (decode FC1, 16-byte transfers)
     w4a8_phase2._stage_n64_sfb         (FC2, bytes, 2-column K tail)
     dynamic._w4a8_stage_n64_sfb_tile   (bytes; whatever layout it assumes, coded == native per offset)
3. The adapter hook swaps the prepared representation and the canonical scale handles to the codes and drops
   the checkpoint-layout scale storage (fake b12x objects), and refuses a layer that does not fit.
4. Default OFF: without DSV41_MOE_SCALE_CODES the adapter neither sets B12X_NEXT_W4A8_SCALE_CODES nor converts.
5. The patch removes no native statement except four rewritten signatures / keys (the coded branch is
   const_expr-gated); compiled native PTX identity is checked separately (sm_121a compile check).

usage: python3 tests/test_scale_codes.py      (torch; no GPU)
"""
import dataclasses
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "adapter"))
import scale_codes as sc  # noqa: E402

N, K, E = 576, 5120, 8                 # TP4 slice, hidden, experts (384 in production; 8 keep the test fast)
K32_W13, K32_W2 = K // 32, N // 32
FAILS = []


def check(c, m):
    if not c:
        FAILS.append(m)


def n64_source(rows, scale_cols, group_rows=None):
    """(row, col) of each stored byte of one expert: `_grid_to_w4a8_n64_sfb_kernel`'s index math."""
    group_rows = rows if group_rows is None else group_rows
    idx = torch.arange(rows * scale_cols)
    gb = group_rows * scale_cols
    group, gl = idx // gb, idx % gb
    full_n_tiles = group_rows // 128
    full_n_bytes = full_n_tiles * 128 * scale_cols
    n_tail = gl >= full_n_bytes
    nt = torch.where(n_tail, torch.full_like(gl, full_n_tiles), gl // (128 * scale_cols))
    n_rows = torch.where(n_tail, torch.full_like(gl, group_rows % 128 or 128), torch.full_like(gl, 128))
    wn = torch.where(n_tail, gl - full_n_bytes, gl - nt * 128 * scale_cols)
    fkt = scale_cols // 4
    fkb = fkt * n_rows * 4
    k_tail = wn >= fkb
    kt = torch.where(k_tail, torch.full_like(gl, fkt), wn // (n_rows * 4))
    tc = torch.where(k_tail, torch.full_like(gl, scale_cols % 4 or 4), torch.full_like(gl, 4))
    wk = torch.where(k_tail, wn - fkb, wn - kt * n_rows * 4)
    return group * group_rows + nt * 128 + wk // tc, kt * 4 + wk % tc


def repack(logical, group_rows=None):
    e, rows, cols = logical.shape
    r, c = n64_source(rows, cols, group_rows)
    return logical.clamp(max=247).reshape(e, -1)[:, r * cols + c].contiguous()


def scales(e, rows, cols, gen, lo=111, hi=127):
    base = torch.randint(lo, hi - 7, (e, 1, 1), generator=gen)
    return (base + torch.randint(0, 9, (e, rows, cols), generator=gen)).to(torch.uint8)


# --- line-for-line emulations of the coded branches (python ints, u32 arithmetic) ------------------------
def _u32(b, off):
    return int(b[off]) | int(b[off + 1]) << 8 | int(b[off + 2]) << 16 | int(b[off + 3]) << 24


def _spread16(c):
    return (c & 0xF) | ((c & 0xF0) << 4) | ((c & 0xF00) << 8) | ((c & 0xF000) << 12)


def coded_stage_16b(flat, expert_idx, native_expert_words, local_word):
    block_words = expert_idx * (native_expert_words // 2 + 4)
    base = _u32(flat, block_words * 4) & 0xFF
    b4 = (base * 0x01010101) & 0xFFFFFFFF
    w = block_words + 4 + local_word // 2
    c0, c1 = _u32(flat, w * 4), _u32(flat, w * 4 + 4)
    return [(_spread16(x) + b4) & 0xFFFFFFFF for x in (c0 & 0xFFFF, c0 >> 16, c1 & 0xFFFF, c1 >> 16)]


def coded_byte(flat, expert_idx, native_expert_bytes, local_byte):
    block = expert_idx * (native_expert_bytes // 2 + 16)
    base = int(flat[block])
    code = int(flat[block + 16 + (local_byte >> 1)])
    return (base + ((code >> ((local_byte & 1) * 4)) & 15)) & 0xFF


def coded_async_slot(flat, expert_idx, native_expert_words, local_word):
    """phase 1, 2026-10-30: coded_cp_async_16b fills the 16-byte slot (header word -> +0, 8 code bytes -> +8; bytes
    4..7 are whatever the slot held: poisoned here), then coded_expand_16b rewrites it in place. Returns the slot's
    four words after the expansion and the two copies' source byte addresses (alignment)."""
    block_words = expert_idx * (native_expert_words // 2 + 4)
    hdr, codes = block_words * 4, (block_words + 4 + local_word // 2) * 4
    slot = [flat[hdr + j] for j in range(4)] + [0xA5] * 4 + [flat[codes + j] for j in range(8)]
    b4 = ((_u32(slot, 0) & 0xFF) * 0x01010101) & 0xFFFFFFFF
    c0, c1 = _u32(slot, 8), _u32(slot, 12)
    return [(_spread16(x) + b4) & 0xFFFFFFFF for x in (c0 & 0xFFFF, c0 >> 16, c1 & 0xFFFF, c1 >> 16)], hdr, codes


def coded_row(flat, expert_idx, native_expert_bytes, local_byte, full_row):
    """phase 2, 2026-10-30: coded_block_base + coded_row4_u32 / coded_row2_u32 -> (u32, code byte address)."""
    block = expert_idx * (native_expert_bytes // 2 + 16)
    base = int(flat[block])
    addr = block + 16 + (local_byte >> 1)
    if full_row:
        code = int(flat[addr]) | int(flat[addr + 1]) << 8
        return (_spread16(code & 0xFFFF) + base * 0x01010101) & 0xFFFFFFFF, addr
    return (_spread16(int(flat[addr]) & 0xFF) + base * 0x0101) & 0xFFFFFFFF, addr


def test_pack_roundtrip():
    gen = torch.Generator().manual_seed(0)
    for rows, cols, group in ((2 * N, K32_W13, N), (K, K32_W2, None)):
        native = repack(scales(E, rows, cols, gen), group)
        coded, span = sc.pack(native, chunk=3)
        check(coded.shape == (E, 16 + native.shape[1] // 2), f"coded shape {tuple(coded.shape)}")
        check(torch.equal(sc.unpack(coded, native.shape[1]), native), "round trip")
        check(bool((coded[:, 1:16] == 0).all()), "header padding not zero")
        x = torch.randint(0, 16, (4, rows * cols), generator=gen).to(torch.uint8)
        x[0] += 0
        x[1] += 232                                         # base 232, max 247
        x[2, 0], x[2, 1] = 0, 15                            # span exactly 16, base 0
        x[3] = 247
        c, s = sc.pack(x)
        check(torch.equal(sc.unpack(c, x.shape[1]), x) and s == 16, f"adversarial round trip, span {s}")
        y = x.clone()
        y[1, 5] = 200                                       # 200..247: span 48
        try:
            sc.pack(y)
            check(False, "span > 16 accepted")
        except sc.CodeError:
            pass


def test_readers():
    gen = torch.Generator().manual_seed(1)
    # ---- FC1 storage: w13 [E, 2N, K/32], two row groups of N (W13 layout: up, gate)
    w13 = repack(scales(E, 2 * N, K32_W13, gen), N)
    c13, _ = sc.pack(w13)
    flat13, nat13 = c13.reshape(-1).tolist(), w13.reshape(-1).tolist()
    expert_words = 2 * N * K32_W13 // 4
    projection_words = N * K32_W13 // 4
    tiles = -(-N // 128)
    words = 0
    for e in range(E):
        for projection in range(2):
            for tile in range(tiles):
                n_rows = 64 if tile == tiles - 1 else 128
                for k128 in range(K32_W13 // 4):
                    for transfer in range(n_rows // 4):
                        # phase1 and compact projection share this arithmetic (tile_base = tile * 128 * K32 / 4)
                        local = projection * projection_words + tile * (128 * K32_W13 // 4) + k128 * n_rows + transfer * 4
                        got = coded_stage_16b(flat13, e, expert_words, local)
                        want = [_u32(nat13, (e * expert_words + local + j) * 4) for j in range(4)]
                        if got != want:
                            FAILS.append(f"FC1 e{e} p{projection} t{tile} k{k128} x{transfer}: {got} != {want}")
                            return
                        slot, hdr, codes = coded_async_slot(flat13, e, expert_words, local)   # phase1's async path
                        if slot != want or hdr % 4 or codes % 8:
                            FAILS.append(f"FC1 async e{e} p{projection} t{tile} k{k128} x{transfer}: {slot} != {want}"
                                         f" (header @{hdr}, codes @{codes})")
                            return
                        words += 4
    check(words == E * 2 * N * K32_W13 // 4, f"FC1 coverage {words}")
    # ---- FC2 storage: w2 [E, K, N/32]
    w2 = repack(scales(E, K, K32_W2, gen))
    c2, _ = sc.pack(w2)
    flat2, nat2 = c2.reshape(-1).tolist(), w2.reshape(-1).tolist()
    expert_bytes = K * K32_W2
    seen = 0
    for e in range(E):
        for tile in range(K // 128):
            base_off = tile * 128 * K32_W2
            for k128 in range(-(-K32_W2 // 4)):
                tile_cols = 4 if k128 < K32_W2 // 4 else K32_W2 % 4
                for idx in range(4 * 128):
                    row, kb = idx & 127, idx >> 7
                    if kb < tile_cols:
                        local = base_off + k128 * 512 + row * tile_cols + kb          # phase2's arithmetic
                        if coded_byte(flat2, e, expert_bytes, local) != nat2[e * expert_bytes + local]:
                            FAILS.append(f"FC2 e{e} t{tile} k{k128} r{row} kb{kb}")
                            return
                        seen += 1
    check(seen == E * expert_bytes, f"FC2 coverage {seen}")
    # ---- phase2's coded branch (2026-10-30): one u32 per N row, the same 4 smem bytes as the native byte stores
    rows_seen = 0
    for e in range(E):
        for tile in range(K // 128):
            base_off = tile * 128 * K32_W2
            for k128 in range(-(-K32_W2 // 4)):
                tile_cols = 4 if k128 < K32_W2 // 4 else K32_W2 % 4
                for row in range(128):
                    local = base_off + k128 * 512 + row * tile_cols
                    want = [nat2[e * expert_bytes + local + kb] if kb < tile_cols else 0 for kb in range(4)]
                    got, addr = coded_row(flat2, e, expert_bytes, local, tile_cols == 4)
                    if [(got >> (8 * kb)) & 0xFF for kb in range(4)] != want or (tile_cols == 4 and addr % 2):
                        FAILS.append(f"FC2 row e{e} t{tile} k{k128} r{row}: {got:#010x} != {want} (code @{addr})")
                        return
                    rows_seen += 1
    check(rows_seen == E * (K // 128) * -(-K32_W2 // 4) * 128, f"FC2 row coverage {rows_seen}")
    # ---- dynamic._w4a8_stage_n64_sfb_tile: its own (64-row tile) arithmetic, any offset it forms
    total_n64, total_k32 = 2 * N // 64, K32_W13
    eb = total_n64 * total_k32 * 64
    full_k = total_k32 // 4
    for e in range(E):
        for n64 in range(total_n64):
            for k32 in range(total_k32):
                for row in range(64):
                    kt, kb = k32 >> 2, k32 & 3
                    tile_cols, tile_base = 4, kt * 256
                    if kt == full_k:
                        tile_cols, tile_base = total_k32 - full_k * 4, full_k * 256
                    local = n64 * (total_k32 * 64) + tile_base + row * tile_cols + kb
                    if local >= eb:
                        continue
                    if coded_byte(flat13, e, eb, local) != nat13[e * eb + local]:
                        FAILS.append(f"dynamic e{e} n64 {n64} k32 {k32} row {row}")
                        return


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Val:
    w13_rp: torch.Tensor
    w13_sfb: torch.Tensor
    w2_rp: torch.Tensor
    w2_sfb: torch.Tensor
    n64_repack: bool = True


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Rep:
    value: object


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Impl:
    w1_blockscale: torch.Tensor
    w2_blockscale: torch.Tensor
    representation: object


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Experts:
    plan: object
    _impl: object


def _hook_env(env):
    code = (
        "import os, sys, types, torch\n"
        f"sys.path.insert(0, {os.path.join(ROOT, 'adapter')!r})\n"
        "import moe_b12x_next as m\n"
        "print('SCALE_CODES', m.SCALE_CODES, 'ENV', os.environ.get('B12X_NEXT_W4A8_SCALE_CODES'))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**{k: v for k, v in os.environ.items() if not k.startswith(("DSV41_", "B12X_"))}, **env})
    return r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:]


def test_adapter_hook():
    rc, out = _hook_env({})
    check(rc == 0 and out == ["SCALE_CODES False ENV None"], f"default OFF: {rc} {out}")
    rc, out = _hook_env({"DSV41_MOE_SCALE_CODES": "release"})
    check(rc == 0 and out == ["SCALE_CODES True ENV 1"], f"release mode: {rc} {out}")
    # v2.7.5: =1 is retain mode (codes beside the native scales, plans native): the process switch stays unset
    rc, out = _hook_env({"DSV41_MOE_SCALE_CODES": "1"})
    check(rc == 0 and out == ["SCALE_CODES True ENV None"], f"retain mode: {rc} {out}")
    sys.path.insert(0, os.path.join(ROOT, "adapter"))
    import moe_b12x_next as m
    gen = torch.Generator().manual_seed(2)
    s13 = repack(scales(4, 2 * N, K32_W13, gen), N)
    s2 = repack(scales(4, K, K32_W2, gen))
    p13 = torch.nn.Parameter(s13.clone().view(torch.uint8), requires_grad=False)
    p2 = torch.nn.Parameter(s2.clone().view(torch.uint8), requires_grad=False)
    val = _Val(w13_rp=torch.zeros(1), w13_sfb=p13.data.view(torch.int32), w2_rp=torch.zeros(1),
               w2_sfb=p2.data.view(torch.int32))
    experts = _Experts(plan=None, _impl=_Impl(w1_blockscale=p13.data, w2_blockscale=p2.data,
                                              representation=_Rep(value=val)))
    layer = type("L", (), {})()
    layer.w13_weight_scale_inv, layer.w2_weight_scale_inv = p13, p2
    out, _meta = m._code_scales(experts, layer, torch.device("cpu"))
    v = out._impl.representation.value
    check(v.w13_sfb.dtype == torch.uint8 and torch.equal(sc.unpack(v.w13_sfb, s13.shape[1]), s13), "w13 not coded")
    check(v.w2_sfb.dtype == torch.uint8 and torch.equal(sc.unpack(v.w2_sfb, s2.shape[1]), s2), "w2 not coded")
    check(out._impl.w1_blockscale is v.w13_sfb and out._impl.w2_blockscale is v.w2_sfb, "canonical handles not swapped")
    check(layer.w13_weight_scale_inv.numel() == 0 and layer.w2_weight_scale_inv.numel() == 0, "native storage kept")
    bad = s13.clone()
    bad[0, 0], bad[0, 1] = 100, 140
    q = torch.nn.Parameter(bad, requires_grad=False)
    val = _Val(w13_rp=torch.zeros(1), w13_sfb=q.data.view(torch.int32), w2_rp=torch.zeros(1), w2_sfb=p2.data)
    layer.w13_weight_scale_inv = q
    try:
        m._code_scales(_Experts(plan=None, _impl=_Impl(w1_blockscale=q.data, w2_blockscale=p2.data,
                                                      representation=_Rep(value=val))), layer, torch.device("cpu"))
        check(False, "a layer spanning 41 values was accepted")
    except (SystemExit, RuntimeError):
        pass


def test_chunk_validation():
    """REVIEW_sol_scalecodes R2: a negative chunk used to skip every loop and return all-zero codes as success."""
    word = torch.full((1, 4), 0x78787878, dtype=torch.int32)                 # Sol's reproduction
    for bad in ("-1", "0", "x", "65", ""):
        os.environ["DSV41_MOE_SCALE_CODES_CHUNK"] = bad
        try:
            sc.convert(word, "w")
            check(False, f"CHUNK={bad!r} accepted")
        except sc.CodeError:
            pass
    os.environ.pop("DSV41_MOE_SCALE_CODES_CHUNK")
    for bad in (-1, 0, 65, "2.5"):
        try:
            sc.pack(torch.zeros(2, 16, dtype=torch.uint8), chunk=bad)
            check(False, f"pack(chunk={bad!r}) accepted")
        except sc.CodeError:
            pass
    c, span = sc.convert(word, "w")
    check(span == 1 and torch.equal(sc.unpack(c, 16), word.view(torch.uint8).reshape(1, -1)), "valid word")
    x = torch.full((5, 16), 120, dtype=torch.uint8)
    x[4, 0], x[4, 1] = 100, 117                                             # last expert, partial chunk, span 18
    try:
        sc.pack(x, chunk=2)
        check(False, "an unrepresentable final expert in a partial chunk was accepted")
    except sc.CodeError:
        pass
    try:
        sc.pack(torch.zeros(0, 16, dtype=torch.uint8))
        check(False, "empty storage accepted")
    except sc.CodeError:
        pass


def test_receipt_after_release():
    """Single-storage ownership (2026-10-30): _code_scales returns the native storage's bytes by resizing that storage,
    and _codes_receipt proves it (storages empty, allocator delta). A live-tensor account stands in for the CUDA
    allocator (REVIEW_sol_scalecodes R1). The 10-27 failures as cases: a view kept by someone else (SGLang's snapshots,
    the sanitizer runs' Triton launch records) no longer blocks the release; it names an empty storage."""
    import gc
    sys.path.insert(0, os.path.join(ROOT, "adapter"))
    import moe_b12x_next as m
    tracked = set()                                 # storage identities the account tracks (native + codes)

    def track(t):
        tracked.add(m._storage_key(t.untyped_storage()))
        return t

    def allocated(_dev):
        """Bytes of tracked storages that some live tensor still references (what the allocator would hold)."""
        gc.collect()
        seen = {}
        for o in gc.get_objects():
            if isinstance(o, torch.Tensor):
                try:
                    st = o.untyped_storage()
                except RuntimeError:
                    continue
                k = m._storage_key(st)
                if k in tracked and st.nbytes():
                    seen[k] = st.nbytes()
        return sum(seen.values())

    real_convert = sc.convert
    sc.convert = lambda t, label="": (lambda c, s_: (track(c), s_))(*real_convert(t, label))
    saved = (m._ALLOCATED, m._SYNC, m._RESET_PEAK, m._MAX_ALLOCATED, m._EMPTY_CACHE)
    m._ALLOCATED, m._SYNC, m._RESET_PEAK = allocated, (lambda d: None), (lambda d: None)
    m._MAX_ALLOCATED = allocated
    m._EMPTY_CACHE = lambda d: None
    slack, m._ROUND_SLACK = m._ROUND_SLACK, 0      # block rounding is a CUDA allocator property, absent here
    dev = type("Dev", (), {"type": "cuda"})()

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _ImplCache(_Impl):
        cache: dict

    try:
        def build(impl_cls=_Impl, **extra):
            gen = torch.Generator().manual_seed(4)
            n13 = track(repack(scales(2, 2 * N, K32_W13, gen), N))
            n2 = track(repack(scales(2, K, K32_W2, gen)))
            p13 = torch.nn.Parameter(n13, requires_grad=False)        # as SGLang creates them: not views
            p2 = torch.nn.Parameter(n2, requires_grad=False)
            del n13, n2
            val = _Val(w13_rp=torch.zeros(1), w13_sfb=p13.data.view(torch.int32), w2_rp=torch.zeros(1),
                       w2_sfb=p2.data.view(torch.int32))
            kw = {k: (v(p13, p2) if callable(v) else v) for k, v in extra.items()}
            ex = _Experts(plan=None, _impl=impl_cls(w1_blockscale=p13.data, w2_blockscale=p2.data,
                                                    representation=_Rep(value=val), **kw))
            ly = type("L", (), {})()
            ly.w13_weight_scale_inv, ly.w2_weight_scale_inv = p13, p2
            return ex, ly, p13.untyped_storage().nbytes() + p2.untyped_storage().nbytes()

        # the caller's sequence in _convert_layer: drop the pointer-check view, rebind, then the receipt
        experts, layer, native = build()
        rep = experts._impl.representation.value
        del rep
        experts, meta = m._code_scales(experts, layer, dev)
        freed = m._codes_receipt(meta)
        check(freed == native - meta["coded"], f"receipt after release: freed {freed}, want {native - meta['coded']}")
        last = m._CODES["per_layer"][-1]
        check(last["freed"] == freed and last["released"] == 2 and last["aliases"] == 0, f"per-layer receipt {last}")
        check(meta["native_storages"] is None, "the receipt keeps the emptied storage objects")
        # 10-27 (sanitizer legs; SGLang's snapshots on real boots): a view someone else keeps. The bytes are released
        # anyway, the receipt passes, the holder names an empty storage, and the receipt counts it
        experts, layer, native = build()
        stale = experts._impl.representation.value.w13_sfb
        experts, meta = m._code_scales(experts, layer, dev)
        freed = m._codes_receipt(meta)
        check(freed == native - meta["coded"], f"held view: freed {freed}, want {native - meta['coded']}")
        check(stale.untyped_storage().nbytes() == 0 and stale.numel() > 0, "the held view still names native bytes")
        check(m._CODES["per_layer"][-1]["aliases"] == 1, f"aliases not counted: {m._CODES['per_layer'][-1]}")
        # the holder diagnosis (opt-in): names the holder; a frame is named by its code name
        holders_saved, m._HOLDERS = m._HOLDERS, True
        try:
            experts, layer, native = build()
            kept = [experts._impl.representation.value.w2_sfb]
            experts, meta = m._code_scales(experts, layer, dev)
            m._codes_receipt(meta)
            h = m._CODES["per_layer"][-1].get("holders", "")
            check("torch.int32" in h and "list" in h and "_holders" not in h,
                  f"holder diagnosis: {h!r}")
            del kept
        finally:
            m._HOLDERS = holders_saved
        del stale
        # structural ownership: a coded tree that still names the native storage would read freed bytes: refused
        experts, layer, native = build(_ImplCache, cache=lambda p13, p2: {"w13": p13.data[:1]})
        try:
            m._code_scales(experts, layer, dev)
            check(False, "a coded tree still naming the native storage was accepted")
        except RuntimeError as exc:
            check("still reference the native scale storage" in str(exc) and "cache" in str(exc),
                  f"stale coded tree refused for another reason: {exc}")
        # sol review 8eed812 #5: an alias in a REGISTERED CHILD MODULE of the layer (a buffer of layer.child) must refuse
        # before any resize; the equivalent direct buffer refuses too
        for where in ("child", "direct"):
            experts, layer, native = build()
            ly = torch.nn.Module()
            ly.w13_weight_scale_inv, ly.w2_weight_scale_inv = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
            alias = ly.w2_weight_scale_inv.data[:1]
            if where == "child":
                ly.child = torch.nn.Module()
                ly.child.register_buffer("stale", alias, persistent=False)
            else:
                ly.register_buffer("stale", alias, persistent=False)
            st13, st2 = (ly.w13_weight_scale_inv.data.untyped_storage(), ly.w2_weight_scale_inv.data.untyped_storage())
            try:
                m._code_scales(experts, ly, dev)
                check(False, f"a {where} module buffer naming the native storage was accepted")
            except RuntimeError as exc:
                check("still reference the native scale storage" in str(exc) and "stale" in str(exc),
                      f"{where} alias refused for another reason: {exc}")
            check(st13.nbytes() > 0 and st2.nbytes() > 0, f"{where} alias: a storage was resized before the refusal")
        # A shared container reached first at the depth limit must be revisited through its shorter path. Otherwise
        # DFS marks the container seen before visiting its tensor, and silently misses the shallow alias.
        experts, layer, native = build()
        ly = torch.nn.Module()
        ly.w13_weight_scale_inv, ly.w2_weight_scale_inv = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
        shared = {"stale": ly.w2_weight_scale_inv.data[:1]}
        deep = shared
        for _ in range(7):
            deep = [deep]
        ly.deep_holder, ly.short_holder = deep, shared
        st13, st2 = (ly.w13_weight_scale_inv.data.untyped_storage(), ly.w2_weight_scale_inv.data.untyped_storage())
        try:
            m._code_scales(experts, ly, dev)
            check(False, "shared alias reached first through a deep path was accepted")
        except RuntimeError as exc:
            check("still reference the native scale storage" in str(exc) and "short_holder" in str(exc),
                  f"shared alias refused for another reason: {exc}")
        check(st13.nbytes() > 0 and st2.nbytes() > 0, "shared alias: a storage was resized before the refusal")
        # A stale alias held beyond the old depth-eight cutoff is still reachable and must refuse before resize.
        experts, layer, native = build()
        ly = torch.nn.Module()
        ly.w13_weight_scale_inv, ly.w2_weight_scale_inv = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
        deep = {"stale": ly.w2_weight_scale_inv.data[:1]}
        for _ in range(20):
            deep = [deep]
        ly.deep_only_holder = deep
        st13, st2 = (ly.w13_weight_scale_inv.data.untyped_storage(), ly.w2_weight_scale_inv.data.untyped_storage())
        try:
            m._code_scales(experts, ly, dev)
            check(False, "alias beyond the old depth limit was accepted")
        except RuntimeError as exc:
            check("still reference the native scale storage" in str(exc) and "deep_only_holder" in str(exc),
                  f"deep alias refused for another reason: {exc}")
        check(st13.nbytes() > 0 and st2.nbytes() > 0, "deep alias: a storage was resized before the refusal")
        # A tensor may also be held in an inherited string-valued __slots__ entry or as a dictionary key.
        class SlotBase:
            __slots__ = "stale"

        class SlotChild(SlotBase):
            __slots__ = ()

        import functools
        import weakref
        from collections import deque

        class CallableHolder:
            def __init__(self, value):
                self.value = value

            def __call__(self):
                return self.value

        class PrivateSlotHolder:
            __slots__ = "__stale"

            def __init__(self, value):
                self.__stale = value

        class SlotModule(torch.nn.Module):
            __slots__ = ("stale",)

        def closure(value):
            def read():
                return value
            return read

        for where in ("slot", "private-slot", "module-slot", "tensor-attribute", "deque", "dlpack-full", "dlpack-slice", "raw-storage", "typed-storage", "dictkey", "callable", "partial", "closure", "weakref"):
            experts, layer, native = build()
            ly = torch.nn.Module()
            ly.w13_weight_scale_inv, ly.w2_weight_scale_inv = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
            alias = ly.w2_weight_scale_inv.data[:1]
            if where == "slot":
                holder = SlotChild()
                holder.stale = alias
                ly.extra_holder = holder
            elif where == "private-slot":
                ly.extra_holder = PrivateSlotHolder(alias)
            elif where == "module-slot":
                ly.extra_holder = SlotModule()
                ly.extra_holder.stale = alias
            elif where == "tensor-attribute":
                ly.extra_holder = torch.zeros(1)
                ly.extra_holder.stale = alias
            elif where == "deque":
                ly.extra_holder = deque([alias])
            elif where == "dlpack-full":
                ly.extra_holder = torch.from_dlpack(ly.w2_weight_scale_inv.data)
                check(m._storage_key(ly.extra_holder.untyped_storage()) != m._storage_key(ly.w2_weight_scale_inv.untyped_storage()),
                      "DLPack full alias did not create a distinct StorageImpl")
            elif where == "dlpack-slice":
                ly.extra_holder = torch.from_dlpack(ly.w2_weight_scale_inv.data[1:])
                check(m._storage_key(ly.extra_holder.untyped_storage()) != m._storage_key(ly.w2_weight_scale_inv.untyped_storage()),
                      "DLPack slice alias did not create a distinct StorageImpl")
            elif where == "raw-storage":
                ly.extra_holder = ly.w2_weight_scale_inv.untyped_storage()
            elif where == "typed-storage":
                ly.extra_holder = ly.w2_weight_scale_inv._typed_storage()
            elif where == "dictkey":
                ly.extra_holder = {alias: 1}
            elif where == "callable":
                ly.extra_holder = CallableHolder(alias)
            elif where == "partial":
                ly.extra_holder = functools.partial(lambda value: value, alias)
            elif where == "weakref":
                ly.extra_holder = weakref.ref(alias)
            else:
                ly.extra_holder = closure(alias)
            st13, st2 = (ly.w13_weight_scale_inv.data.untyped_storage(), ly.w2_weight_scale_inv.data.untyped_storage())
            params_before = [(tuple(p.shape), p.data_ptr()) for p in (ly.w13_weight_scale_inv, ly.w2_weight_scale_inv)]
            try:
                m._code_scales(experts, ly, dev)
                check(False, f"{where} alias was accepted")
            except RuntimeError as exc:
                check("still reference the native scale storage" in str(exc) and "extra_holder" in str(exc),
                      f"{where} alias refused for another reason: {exc}")
            check(st13.nbytes() > 0 and st2.nbytes() > 0, f"{where}: a storage was resized before the refusal")
            check([(tuple(p.shape), p.data_ptr()) for p in (ly.w13_weight_scale_inv, ly.w2_weight_scale_inv)] == params_before,
                  f"{where}: parameter bindings changed on refusal")
        experts, layer, native = build()
        layer.uninspectable = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), size=(2,))
        st13, st2 = (layer.w13_weight_scale_inv.data.untyped_storage(), layer.w2_weight_scale_inv.data.untyped_storage())
        params_before = [(tuple(p.shape), p.data_ptr()) for p in (layer.w13_weight_scale_inv, layer.w2_weight_scale_inv)]
        try:
            m._code_scales(experts, layer, dev)
            check(False, "an uninspectable tensor storage was accepted")
        except RuntimeError as exc:
            check("cannot inspect" in str(exc), f"uninspectable storage refused for another reason: {exc}")
        check(st13.nbytes() > 0 and st2.nbytes() > 0, "uninspectable tensor: a storage was resized before refusal")
        check([(tuple(p.shape), p.data_ptr()) for p in (layer.w13_weight_scale_inv, layer.w2_weight_scale_inv)] == params_before,
              "uninspectable tensor: parameter bindings changed on refusal")
        # Cycles terminate, and exhausting the traversal budget refuses rather than silently accepting the layer.
        experts, layer, native = build()
        alias = layer.w2_weight_scale_inv.data[:1]
        cycle = []
        cycle.extend((cycle, alias))
        check(bool(m._aliases_in(cycle, {m._storage_key(alias.untyped_storage())})), "cyclic graph hid an alias")
        st13, st2 = (layer.w13_weight_scale_inv.data.untyped_storage(), layer.w2_weight_scale_inv.data.untyped_storage())
        old_walk = m._aliases_in
        m._aliases_in = lambda root, keys, **kw: old_walk(root, keys, max_nodes=1, **kw)
        try:
            m._code_scales(experts, layer, dev)
            check(False, "an exhausted ownership-walk budget was accepted")
        except RuntimeError as exc:
            check("ownership walk exceeds" in str(exc), f"walk budget refused for another reason: {exc}")
        finally:
            m._aliases_in = old_walk
        check(st13.nbytes() > 0 and st2.nbytes() > 0, "walk budget: a storage was resized before the refusal")
        # a parameter that is a view into a larger storage (another tensor's bytes would be resized away): refused
        experts, layer, native = build()
        big = torch.zeros(layer.w2_weight_scale_inv.numel() + 64, dtype=torch.uint8)
        big[:layer.w2_weight_scale_inv.numel()].view_as(layer.w2_weight_scale_inv).copy_(layer.w2_weight_scale_inv.data)
        pv = torch.nn.Parameter(big[:layer.w2_weight_scale_inv.numel()].view_as(layer.w2_weight_scale_inv),
                                requires_grad=False)
        layer.w2_weight_scale_inv = pv
        v = experts._impl.representation.value
        experts = dataclasses.replace(experts, _impl=dataclasses.replace(experts._impl, representation=_Rep(
            value=dataclasses.replace(v, w2_sfb=pv.data.view(torch.int32)))))
        try:
            m._code_scales(experts, layer, dev)
            check(False, "a parameter sharing its storage was accepted")
        except RuntimeError as exc:
            check("not exactly one parameter-owned storage" in str(exc),
                  f"shared storage refused for another reason: {exc}")
        check(big.untyped_storage().nbytes() > 0, "a shared storage was resized")
        # a storage the allocator does not own (not resizable): refused, nothing resized
        experts, layer, native = build()
        raw = bytearray(layer.w13_weight_scale_inv.data.numpy().tobytes())
        fb = torch.frombuffer(raw, dtype=torch.uint8).view_as(layer.w13_weight_scale_inv)
        pf = torch.nn.Parameter(fb, requires_grad=False)
        layer.w13_weight_scale_inv = pf
        v = experts._impl.representation.value
        experts = dataclasses.replace(experts, _impl=dataclasses.replace(experts._impl, representation=_Rep(
            value=dataclasses.replace(v, w13_sfb=pf.data.view(torch.int32)))))
        try:
            m._code_scales(experts, layer, dev)
            check(False, "a non-resizable storage was accepted")
        except RuntimeError as exc:
            check("not resizable" in str(exc), f"non-resizable storage refused for another reason: {exc}")
        check(layer.w2_weight_scale_inv.untyped_storage().nbytes() > 0, "a refused layer had a storage resized")
        # a receipt whose storages were not emptied: refused
        experts, layer, native = build()
        experts, meta = m._code_scales(experts, layer, dev)
        meta["native_storages"] = [torch.zeros(4, dtype=torch.uint8).untyped_storage()]
        try:
            m._codes_receipt(meta)
            check(False, "a non-empty native storage passed the receipt")
        except RuntimeError as exc:
            check("not released" in str(exc), f"non-empty storage refused for another reason: {exc}")
        # fleet 2026-10-26 (compute-sanitizer): the native blocks are released but the caching allocator still counts
        # them until it processes its deferred frees. The receipt must process them (empty_cache) before reading.
        pending = {"on": False, "native": 0}

        def deferred(_dev):
            return allocated(_dev) + (pending["native"] if pending["on"] else 0)

        def empty_cache(_dev):
            pending["on"] = False

        experts, layer, native = build()
        pending["native"] = native
        m._ALLOCATED, m._EMPTY_CACHE = deferred, empty_cache
        experts, meta = m._code_scales(experts, layer, dev)
        pending["on"] = True                        # the release happened; the allocator has not processed it yet
        freed = m._codes_receipt(meta)
        check(freed == native - meta["coded"], f"deferred frees: receipt {freed}, want {native - meta['coded']}")
        experts, layer, native = build()            # the same without processing the deferred frees: refused
        pending["native"] = native
        m._EMPTY_CACHE = lambda d: None
        experts, meta = m._code_scales(experts, layer, dev)
        pending["on"] = True
        try:
            m._codes_receipt(meta)
            check(False, "an unprocessed deferred free was accepted as a release")
        except RuntimeError:
            pass
        m._ALLOCATED = allocated
        # raw checkpoint bytes > 247 (would be clamped by the repack) refuse before the repack
        experts, layer, native = build()
        layer.w2_weight_scale_inv.data.view(torch.uint8)[0, 0] = 0xFF
        try:
            m._check_raw_scales(layer)
            check(False, "a 0xff checkpoint byte was accepted")
        except RuntimeError:
            pass
    finally:
        sc.convert = real_convert
        m._ALLOCATED, m._SYNC, m._RESET_PEAK, m._MAX_ALLOCATED, m._EMPTY_CACHE = saved
        m._ROUND_SLACK = slack


def test_native_branches_unchanged():
    """Every hunk of the b12x_next patch only ADDS lines or moves a native statement under an else of a
    const_expr(coded) branch: removing the patch's coded branches must give back the stock sources."""
    patch = os.path.join(ROOT, "scripts", "b12x_next-scale-codes.patch")
    if not os.path.exists(patch):
        FAILS.append("scripts/b12x_next-scale-codes.patch missing")
        return
    text = open(patch).read()
    check("coded_scales" in text and "w4a8_scale_codes.py" in text, "patch content")
    removed = [l for l in text.splitlines() if l.startswith("-") and not l.startswith("---")]
    # native statements may be re-indented under `else:`; their stripped text must reappear as an added line
    added = {l[1:].strip() for l in text.splitlines() if l.startswith("+") and not l.startswith("+++")}
    # lines the patch legitimately rewrites: signatures gaining coded_scales, a docstring, closing parens of the
    # compile keys that gain `+ coded_key`, and the compact projection constructor call
    allowed = {'"""Stage exact N64-packed E8M0 bytes into the N128/K128 SFB tile."""',
               "self.projection = W4A8CompactMicroProjectionKernel(k, n, num_topk)",
               "),", "def __init__(self, k: int, n: int, num_topk: int):"}
    lost = [l for l in removed if l[1:].strip() and l[1:].strip() not in added and l[1:].strip() not in allowed]
    check(not lost, f"patch drops native lines: {lost[:5]}")


def _load_post_load(name="sglang.srt.model_loader.post_load", edit=None):
    """SGLang's real model_loader/post_load.py (the staging context process_weights_after_loading runs in), loaded from
    DSV41_SGLANG_SOURCE_ROOT (required when set), the staged runtime/sglang-canary, or the installed sglang package.
    edit(text) -> text makes a drifted copy for the contract test."""
    import importlib.util
    roots = [os.environ.get("DSV41_SGLANG_SOURCE_ROOT"), os.path.join(ROOT, "runtime", "sglang-canary", "python")]
    try:
        spec = importlib.util.find_spec("sglang")
        roots += [os.path.dirname(loc) for loc in (spec.submodule_search_locations or [])] if spec else []
    except (ImportError, ValueError):
        pass
    for r in roots:
        path = os.path.join(r, "sglang", "srt", "model_loader", "post_load.py") if r else None
        if path and os.path.isfile(path):
            text = open(path).read()
            if edit:
                import tempfile
                text = edit(text)
                path = os.path.join(tempfile.mkdtemp(), "post_load.py")      # inspect.getsource reads the file
                with open(path, "w") as f:
                    f.write(text)
            mod = type(sys)(name)
            mod.__file__ = path
            sys.modules[name] = mod            # dataclasses resolves the module while the class is created
            exec(compile(text, path, "exec"), mod.__dict__)
            return mod
    if os.environ.get("DSV41_SGLANG_SOURCE_ROOT"):
        FAILS.append(f"post_load.py not found under DSV41_SGLANG_SOURCE_ROOT={os.environ['DSV41_SGLANG_SOURCE_ROOT']}")
    return None


def _load_loader_flow(pl):
    """DefaultModelLoader.postprocess_weights and device_loading_context, compiled verbatim from the loader.py next to
    the post_load.py under test (same root), bound to that post_load's stage_module_for_post_load."""
    import ast
    import contextlib
    import textwrap
    path = os.path.join(os.path.dirname(pl.__file__), "loader.py")
    if not os.path.isfile(path):
        return None
    text = open(path).read()
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)

    def segment(node):
        first = min([d.lineno for d in node.decorator_list] + [node.lineno])
        return "".join(lines[first - 1:node.end_lineno])
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "device_loading_context":
            found["dlc"] = segment(node)
        if isinstance(node, ast.ClassDef) and node.name == "DefaultModelLoader":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "postprocess_weights":
                    found["pw"] = textwrap.dedent(segment(sub)).replace("@staticmethod\n", "", 1)
    if set(found) != {"dlc", "pw"}:
        return None
    ns = {"torch": torch, "contextmanager": contextlib.contextmanager, "is_pin_memory_available": lambda: False,
          "stage_module_for_post_load": pl.stage_module_for_post_load}
    exec(compile(found["dlc"] + "\n\n" + found["pw"], path, "exec"), ns)
    import types
    return types.SimpleNamespace(postprocess_weights=ns["postprocess_weights"])


def test_post_load_staging():
    """Fleet 2026-10-27, sc-v26 real TP4 boots: SGLang runs process_weights_after_loading inside
    stage_module_for_post_load (model_loader/post_load.py), whose _TensorState snapshots name the scale parameters'
    native storage until the context exits (13:35: original_data AND staged_data; 18:37: staged_data). 4009423 / d1d09b8
    rewrote those snapshots; single-storage ownership (2026-10-30) does not touch SGLang's state at all: the storage is
    emptied, so the snapshots name an empty storage. Three staging regimes with SGLang's real post_load.py (CPU):
      none   process device == origin: no staging (the GPU gate's harness never enters a context at all)
      alias  the production regime: origin != process device, `.to()` returns the tensor itself (CUDA's same-device
             semantics, emulated on CPU by a Tensor.to that returns self when the device type matches)
      copy   origin != process device with a real copy (CPU offload-like): the experts are built from the STAGED data
    For each: the receipt passes inside the context with the snapshots untouched, the exit is clean (the parameters are
    the same objects, empty, at their origin), and every native storage is empty or gone after the exit. Then SGLang's
    real loader flow (DefaultModelLoader.postprocess_weights + device_loading_context, compiled from loader.py)."""
    import contextlib
    import gc
    import inspect as _inspect
    from torch.multiprocessing.reductions import StorageWeakRef
    sys.path.insert(0, os.path.join(ROOT, "adapter"))
    import moe_b12x_next as m
    name = "sglang.srt.model_loader.post_load"
    saved_mod = sys.modules.get(name)
    pl = _load_post_load()
    if pl is None:
        print("test_post_load_staging: SKIP (no SGLang post_load.py; set DSV41_SGLANG_SOURCE_ROOT)", flush=True)
        return
    origin = torch.device("cpu")
    src = _inspect.getsource(m._code_scales) + _inspect.getsource(m._codes_receipt)
    check("_TensorState" not in src and "post_load" not in src.replace("post-load", ""),
          "the codes still read or rewrite SGLang's post-load state")

    class Layer(torch.nn.Module):
        pass

    def build_layer(seed):
        """Native shapes of one TP4 rank at E=4: w13 scales (E, 1152, 160), w2 scales (E, 5120, 18)."""
        gen = torch.Generator().manual_seed(seed)
        ly = Layer()
        ly.w13_weight_scale_inv = torch.nn.Parameter(scales(4, 2 * N, K32_W13, gen), requires_grad=False)
        ly.w2_weight_scale_inv = torch.nn.Parameter(scales(4, K, K32_W2, gen), requires_grad=False)
        ly.w13_weight = torch.nn.Parameter(torch.zeros(4, dtype=torch.uint8), requires_grad=False)
        return ly

    def repack_in_place(ly):
        """b12x_next's in-place N64 repack of the scale parameters' CURRENT data (inside the context: the staged data
        when staged), then the prepared experts viewing that storage, as _convert_layer's prepare_weights does."""
        p13, p2 = ly.w13_weight_scale_inv, ly.w2_weight_scale_inv
        for p, group in ((p13, N), (p2, None)):
            d = p.data
            d.view(d.shape[0], -1).copy_(repack(d.clone(), group))
        e = p13.shape[0]
        val = _Val(w13_rp=torch.zeros(1), w13_sfb=p13.data.view(e, -1).view(torch.int32), w2_rp=torch.zeros(1),
                   w2_sfb=p2.data.view(e, -1).view(torch.int32))
        return _Experts(plan=None, _impl=_Impl(w1_blockscale=p13.data, w2_blockscale=p2.data,
                                              representation=_Rep(value=val)))

    @contextlib.contextmanager
    def regime(kind):
        if kind == "alias":
            orig_to = torch.Tensor.to

            def same_device_to(self, *a, **kw):
                dev = a[0] if a and isinstance(a[0], torch.device) else None
                if dev is not None and len(a) == 1 and not kw and dev.type == self.device.type:
                    return self                      # CUDA: .to("cuda") of a cuda:0 tensor is the tensor itself
                return orig_to(self, *a, **kw)
            torch.Tensor.to = same_device_to
            try:
                yield torch.device("cpu", 0)
            finally:
                torch.Tensor.to = orig_to
        elif kind == "copy":
            yield torch.device("cpu", 0)             # unequal to the origin; .to() copies
        else:
            yield origin

    def snapshots(ly):
        out = {}
        for nm in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
            p = getattr(ly, nm)
            for r in gc.get_referrers(p):
                if isinstance(r, pl._TensorState) and r.tensor is p:
                    out[nm] = (r, r.original_data, r.staged_data)
        return out

    def convert(ly, process, keep=None):
        """The codes step inside SGLang's staging context, receipt inside it (as _convert_layer)."""
        with pl.stage_module_for_post_load(ly, process):
            snaps = snapshots(ly)
            ex = repack_in_place(ly)
            if keep is not None:
                keep.append(ex._impl.representation.value.w2_sfb)
            box = [ex]
            del ex
            out, meta = m._code_scales(box.pop(), ly, origin)
            m._codes_receipt(meta)
            untouched = all(st.original_data is od and st.staged_data is sd for st, od, sd in snaps.values())
            snap_bytes = sum(t.untyped_storage().nbytes() for _, od, sd in snaps.values() for t in (od, sd)
                             if t is not None and _storage_in(t, meta))
        return out, meta, snaps, untouched, snap_bytes

    def _storage_in(t, meta):
        return m._storage_key(t.untyped_storage()) in meta["native_keys"]

    for kind in ("none", "alias", "copy"):
        with regime(kind) as process:
            ly = build_layer(33)
            p13, p2 = ly.w13_weight_scale_inv, ly.w2_weight_scale_inv
            origin_refs = [StorageWeakRef(p.data.untyped_storage()) for p in (p13, p2)]
            out, meta, snaps, untouched, snap_bytes = convert(ly, process)
            check(untouched, f"{kind}: SGLang's snapshots were rewritten")
            check(snap_bytes == 0, f"{kind}: a snapshot still names {snap_bytes} native bytes inside the context")
            if kind != "none":
                check(len(snaps) == 2, f"{kind}: {len(snaps)} snapshots of the scale parameters, 2 expected")
            check(ly.w13_weight_scale_inv is p13 and ly.w2_weight_scale_inv is p2 and p13.numel() == 0
                  and p2.numel() == 0 and p13.device == origin and p2.device == origin,
                  f"{kind}: after the exit the parameters are the same objects, empty, at their origin")
            check(out._impl.representation.value.w13_sfb.dtype == torch.uint8, f"{kind}: codes not bound")
            del snaps
            gc.collect()
            # alias / none: the origin storage IS the emptied one; copy: the origin copy dies with the snapshots
            check(all(r.expired() for r in origin_refs) or kind != "copy", f"{kind}: the origin copy outlived the exit")
            check(all(r.expired() for r in meta["native_refs"]), f"{kind}: native storage named after the exit")
        # a view someone else keeps: released anyway, counted
        with regime(kind) as process:
            ly = build_layer(34)
            stale = []
            out, meta, _, _, _ = convert(ly, process, keep=stale)
            check(stale[0].untyped_storage().nbytes() == 0, f"{kind}: a kept view still names native bytes")
            check(m._CODES["per_layer"][-1]["aliases"] >= 1, f"{kind}: the kept view was not counted")
            del stale
    # SGLang's real loader flow for one MoE layer: DefaultModelLoader.postprocess_weights + device_loading_context,
    # executed verbatim from loader.py (the module itself needs the full serving stack to import), target
    # torch.device("cuda") as device_config builds it, the layer's params on the CPU standing in for cuda:0
    # (.to("cuda") returns the tensor itself, as CUDA does for a tensor already on that GPU)
    flow = _load_loader_flow(pl)
    check(flow is not None, "loader.py's device_loading_context / postprocess_weights not found")
    if flow is not None:
        class Method:
            def __init__(self):
                self.metas = []

            def process_weights_after_loading(self, layer):      # as _convert_layer: repack, codes, receipt
                ex = repack_in_place(layer)
                box = [ex]
                del ex
                out, meta = m._code_scales(box.pop(), layer, origin)
                m._codes_receipt(meta)
                layer.experts_out = out
                self.metas.append(meta)

        def build_model(seed):
            model = torch.nn.Module()
            model.mlp = torch.nn.Module()
            model.mlp.experts = build_layer(seed)
            model.mlp.experts.quant_method = Method()
            return model

        orig_to = torch.Tensor.to

        def cuda_is_here(self, *a, **kw):
            dev = a[0] if a and isinstance(a[0], torch.device) else None
            if dev is not None and dev.type == "cuda" and len(a) == 1 and set(kw) <= {"non_blocking"}:
                return self
            return orig_to(self, *a, **kw)
        for target in (torch.device("cuda"), torch.device("cpu")):
            model = build_model(40)
            ly = model.mlp.experts
            p13, p2 = ly.w13_weight_scale_inv, ly.w2_weight_scale_inv
            torch.Tensor.to = cuda_is_here
            try:
                flow.postprocess_weights(model, target)
                err = None
            except RuntimeError as exc:
                err = str(exc)
            finally:
                torch.Tensor.to = orig_to
            check(err is None, f"real loader flow ({target}): refused: {err}")
            if err is None:
                meta = ly.quant_method.metas[-1]
                check(ly.w13_weight_scale_inv is p13 and p13.numel() == 0 and p2.numel() == 0
                      and ly.w13_weight.numel() == 4, f"real loader flow ({target}): parameters after the exit")
                gc.collect()
                check(all(r.expired() for r in meta["native_refs"]),
                      f"real loader flow ({target}): native storage named after the exit")
    if saved_mod is not None:
        sys.modules[name] = saved_mod
    else:
        sys.modules.pop(name, None)


if __name__ == "__main__":
    for fn in (test_pack_roundtrip, test_readers, test_adapter_hook, test_chunk_validation, test_receipt_after_release,
               test_post_load_staging, test_native_branches_unchanged):
        n = len(FAILS)
        fn()
        print(f"{fn.__name__}: {'ok' if len(FAILS) == n else 'FAIL'}", flush=True)
    for f in FAILS:
        print("  " + f)
    assert not FAILS, f"{len(FAILS)} check(s) failed"
    print("test_scale_codes: ok")
