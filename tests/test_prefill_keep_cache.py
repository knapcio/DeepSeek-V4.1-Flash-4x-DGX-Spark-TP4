"""DSV41_PREFILL_KEEP_CACHE: the policy, the plumbing helpers, and an offline memory accounting of a whole prefill.

Pure Python (no torch, no GPU): python3 tests/test_prefill_keep_cache.py   or   pytest tests/test_prefill_keep_cache.py

The accounting replays a model of one 8192-row chunk's allocations (stable 8192-row transients of 20 full-row layers +
the dense indexer's prefix-dependent tensors on 4 index layers, sizes from adapter/indexer_chunked_v3.py and
spark_prefill_dense.py) through a re-implementation of PyTorch's CUDACachingAllocator block rules (512 B rounding,
small/large pools, 2 MiB / 20 MiB / 2 MiB-rounded segments, best fit by (size, address), split thresholds, coalescing,
empty_cache = release fully free segments). Three regimes over a whole prompt:
  A  today: empty_cache after every chunk
  B  keep the cache, no padding (why the per-chunk hook exists: reserved grows with the square of the prompt)
  C  keep + final-width padding + the guard (the change)
and asserts that C's reserved high-water is within the documented bound of A's and that C's kept chunks need no new
segment. It is a model: the real allocation list must be read off the fleet (DESIGN.md receipts).
"""
import bisect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))
import prefill_keep_cache as keep  # noqa: E402

MIB = 1 << 20
GIB = 1 << 30


# ------------------------------------------------------------------ CUDACachingAllocator model
class Block:
    __slots__ = ("seg", "addr", "size", "free", "prev", "next", "pool")

    def __init__(self, seg, addr, size, pool):
        self.seg, self.addr, self.size, self.free, self.prev, self.next, self.pool = seg, addr, size, True, None, None, pool


class Allocator:
    MIN_BLOCK, SMALL, SMALL_BUF, MIN_LARGE, ROUND_LARGE, LARGE_BUF = 512, MIB, 2 * MIB, 10 * MIB, 2 * MIB, 20 * MIB

    def __init__(self):
        self.free = {"small": [], "large": []}   # sorted [(size, addr, block)]
        self.reserved = 0
        self.allocated = 0
        self.peak = 0
        self.num_device_alloc = 0
        self.num_device_free = 0
        self._next_addr = 1 << 40
        self.live = {}

    @staticmethod
    def round_size(n):
        return 512 if n < 512 else (n + 511) // 512 * 512

    def _pool(self, size):
        return "small" if size <= self.SMALL else "large"

    def _seg_size(self, size):
        if size <= self.SMALL:
            return self.SMALL_BUF
        if size < self.MIN_LARGE:
            return self.LARGE_BUF
        return (size + self.ROUND_LARGE - 1) // self.ROUND_LARGE * self.ROUND_LARGE

    def _insert(self, b):
        bisect.insort(self.free[b.pool], (b.size, b.addr, id(b), b))

    def _remove(self, b):
        lst = self.free[b.pool]
        i = bisect.bisect_left(lst, (b.size, b.addr, id(b)))
        assert lst[i][3] is b
        lst.pop(i)

    def malloc(self, n):
        size = self.round_size(n)
        pool = self._pool(size)
        lst = self.free[pool]
        i = bisect.bisect_left(lst, (size, -1, -1))
        if i < len(lst):
            b = lst[i][3]
            self._remove(b)
        else:
            seg = self._seg_size(size)
            self.num_device_alloc += 1
            self.reserved += seg
            b = Block(object(), self._next_addr, seg, pool)
            self._next_addr += seg + (1 << 30)
        rem = b.size - size
        if (pool == "small" and rem >= self.MIN_BLOCK) or (pool == "large" and rem > self.SMALL):
            r = Block(b.seg, b.addr + size, rem, pool)
            r.prev, r.next = b, b.next
            if b.next is not None:
                b.next.prev = r
            b.next = r
            b.size = size
            self._insert(r)
        b.free = False
        self.allocated += b.size
        self.peak = max(self.peak, self.allocated)
        h = object()
        self.live[h] = b
        return h

    def release(self, h):
        b = self.live.pop(h)
        self.allocated -= b.size
        b.free = True
        for nb in (b.prev, b.next):
            if nb is not None and nb.free:
                self._remove(nb)
                if nb is b.prev:
                    nb.size += b.size
                    nb.next = b.next
                    if b.next is not None:
                        b.next.prev = nb
                    b = nb
                else:
                    b.size += nb.size
                    b.next = nb.next
                    if nb.next is not None:
                        nb.next.prev = b
        self._insert(b)

    def empty_cache(self):
        for pool in ("small", "large"):
            keep_list = []
            for item in self.free[pool]:
                b = item[3]
                if b.prev is None and b.next is None:
                    self.reserved -= b.size
                    self.num_device_free += 1
                else:
                    keep_list.append(item)
            self.free[pool] = keep_list

    def reset_peak(self):
        self.peak = self.allocated

    def stats(self):
        return {"reserved": self.reserved, "peak": self.peak, "allocs": self.num_device_alloc,
                "frees": self.num_device_free}


# ------------------------------------------------------------------ one chunk's allocations (model)
CHUNK, WORLD, RATIO, BLOCK, TAIL = 8192, 4, 4, 64, 128
SPLIT_MIN_CTX = 32768            # SPARK_PREFILL_TP_MIN_CONTEXT, compressed units
BUDGET = 2 * GIB                 # DSV41_INDEXER_LOGITS_BUDGET_BYTES
INDEX_LAYERS = (2, 8, 14, 20)
FULL_LAYERS = 21


def ceil_to(x, a):
    return (x + a - 1) // a * a


def indexer_events(seq, final, pad):
    """(op, name, bytes) of one index-layer call, after adapter/indexer_chunked_v3.py."""
    lc = seq // RATIO
    ev = [("a", "k_slots", lc * 8), ("a", "k_fp4", lc * 64), ("a", "k_sf", lc * 4),
          ("a", "q_fp4", CHUNK * 8 * 64), ("a", "weights", CHUNK * 8 * 4), ("a", "selected", CHUNK * 512 * 4)]
    split = lc >= SPLIT_MIN_CTX
    rows = CHUNK // WORLD if split else CHUNK
    if split:
        w = ceil_to(lc, 4)
        if pad:
            w = keep.padded_width(lc, final // RATIO, keep.lcm(4, BLOCK))
    else:
        w = ceil_to(lc, 4)
        if pad:  # adapter/indexer_chunked_v3.unsplit_pad_width with the SG18 split enabled
            w = keep.padded_width(lc, keep.last_unsplit_lc(seq, CHUNK, final, RATIO, SPLIT_MIN_CTX), 4)
    rows_piece = min(rows, max(1, BUDGET // (w * 4)))
    pieces = -(-rows // rows_piece)
    if split:
        ev.append(("a", "packed", rows * (512 + 64) * 4))
    for p in range(pieces):
        r = min(rows_piece, rows - p * rows_piece)
        ev.append(("a", f"logits{p}", r * w * 4))
        # causal mask temp (padded: full padded row; stock: current width)
        ev.append(("a", f"mask{p}", r * (w if pad else lc)))
        ev.append(("f", f"mask{p}", 0))
        if split and not pad:  # candidate_block_ids F.pad copy (padded mode takes maxima on the padded view)
            ev.append(("a", f"fpad{p}", r * ceil_to(lc, BLOCK) * 4))
        ev.append(("a", f"blkmax{p}", r * (ceil_to(lc, BLOCK) // BLOCK) * 4))
        if split and not pad:
            ev.append(("f", f"fpad{p}", 0))
        ev.append(("f", f"blkmax{p}", 0))
        ev.append(("f", f"logits{p}", 0))
    if split:
        ev.append(("a", "gathered", CHUNK * (512 + 64) * 4))
        ev.append(("f", "packed", 0))
    ev.append(("a", "tail_mask", TAIL * lc))     # published, tail-only; lives to the end of the forward
    for name in ("k_slots", "k_fp4", "k_sf", "q_fp4", "weights", "selected") + (("gathered",) if split else ()):
        ev.append(("f", name, 0))
    return ev


def layer_events(i, seq, final, pad):
    H = CHUNK * 5120 * 2
    ev = [("a", f"x{i}", H), ("a", f"xq{i}", H // 2), ("a", f"xs{i}", CHUNK * 160)]
    ev += [("a", f"qkva{i}", CHUNK * 1792 * 2), ("f", f"xq{i}", 0), ("f", f"xs{i}", 0)]
    if i in INDEX_LAYERS:
        ev += [(op, f"{n}@{i}", b) for op, n, b in indexer_events(seq, final, pad)]
    ev += [("a", f"qb{i}", CHUNK * 4096 * 2), ("a", f"mla{i}", CHUNK * 4096 * 2), ("f", f"qkva{i}", 0),
           ("f", f"qb{i}", 0), ("a", f"woa{i}", CHUNK * 2048 * 2), ("f", f"mla{i}", 0), ("a", f"wob{i}", H),
           ("f", f"woa{i}", 0), ("f", f"x{i}", 0), ("a", f"shard{i}", H // WORLD), ("f", f"wob{i}", 0),
           ("a", f"ffn{i}", H), ("a", f"fq{i}", H // 2), ("a", f"fc1{i}", CHUNK * 1152 * 2),
           ("a", f"fc2{i}", H), ("f", f"fc1{i}", 0), ("a", f"p1{i}", CHUNK * 6 * 5120 * 2), ("a", f"p2{i}", CHUNK * 6 * 5120 * 2),
           ("f", f"p1{i}", 0), ("a", f"fin{i}", H), ("f", f"p2{i}", 0), ("a", f"add{i}", H), ("f", f"fin{i}", 0),
           ("f", f"fc2{i}", 0), ("f", f"fq{i}", 0), ("f", f"ffn{i}", 0), ("a", f"out{i}", H // WORLD), ("f", f"add{i}", 0),
           ("f", f"shard{i}", 0)]
    if i > 0:
        ev.append(("f", f"out{i - 1}", 0))
    return ev


def chunk_events(seq, final, pad):
    ev = [("a", "meta", CHUNK * 512 * 4), ("a", "emb", CHUNK * 5120 * 2)]
    for i in range(FULL_LAYERS):
        ev += layer_events(i, seq, final, pad)
    ev += [("f", f"out{FULL_LAYERS - 1}", 0), ("f", "emb", 0), ("f", "meta", 0)]
    ev += [("f", f"tail_mask@{i}", 0) for i in INDEX_LAYERS]
    return ev


def run_chunk(alloc, events):
    live = {}
    for op, name, size in events:
        if op == "a":
            assert name not in live, name
            live[name] = alloc.malloc(size)
        else:
            alloc.release(live.pop(name))
    assert not live, sorted(live)


def simulate(prompt, regime, slack=256 * MIB):
    """regime A (empty every chunk), B (keep, no pad), C (keep + pad + Policy). -> receipts dict"""
    alloc = Allocator()
    persistent = alloc.malloc(60 * GIB)       # weights + KV pool: allocated, never freed
    pol = keep.Policy(slack=slack)
    pol.emptied(alloc.stats())                # boot ends with an empty cache
    seqs = list(range(CHUNK, prompt + 1, CHUNK))
    if seqs[-1] != prompt:
        seqs.append(prompt)
    out = {"reserved_max": 0, "kept": 0, "new_segments_kept": 0, "empties": 0, "trips": 0}
    prev = 0
    for seq in seqs:
        batch = keep.Batch(("A",), (prompt,), (seq,), (seq - prev,))
        prev = seq
        if regime == "C":
            first_empty, _ = pol.begin(batch, alloc.stats())
            if first_empty:
                alloc.empty_cache()
                pol.emptied(alloc.stats())
        alloc.reset_peak()
        before = alloc.stats()
        pol.before(before)
        run_chunk(alloc, chunk_events(seq, prompt, pad=(regime == "C")))
        out["reserved_max"] = max(out["reserved_max"], alloc.reserved)
        if regime == "A":
            alloc.empty_cache()
            out["empties"] += 1
        elif regime == "B":
            pass
        else:
            remeasure = pol.session.get("remeasure", False) or pol.session["chunks"] == 0
            empty, reason = pol.after(batch, alloc.stats())
            if empty:
                alloc.empty_cache()
                pol.emptied(alloc.stats())
                out["empties"] += 1
                out["trips"] += reason.startswith("trip")
            else:
                out["kept"] += 1
                if not remeasure:
                    out["new_segments_kept"] += alloc.stats()["allocs"] - before["allocs"]
    alloc.release(persistent)
    out["policy_sessions"] = pol.sessions
    return out


# ------------------------------------------------------------------ tests
def _st(reserved=0, peak=0, allocs=0):
    return {"reserved": reserved, "peak": peak, "allocs": allocs, "frees": 0}


def _chunk(p, batch, stats_before, stats_after):
    """begin + before + after; returns (empty_before, empty_after, reason_after)."""
    eb, _ = p.begin(batch, stats_before)
    if eb:
        p.emptied(stats_before)
    p.before(stats_before)
    ea, why = p.after(batch, stats_after)
    if ea:
        p.emptied(stats_after)
    return eb, ea, why


B = keep.Batch


def test_policy_keeps_middle_chunks_and_empties_at_end():
    p = keep.Policy(slack=100)
    p.emptied(_st(allocs=5))
    assert _chunk(p, B(["a"], [20000], [8192], [8192]), _st(allocs=5), _st(1000, 990, 9)) == (False, False, "keep")
    assert p.frag_ref == 10 and p.holding
    assert _chunk(p, B(["a"], [20000], [16384], [8192]), _st(1000, 0, 9), _st(1000, 995, 9)) == (False, False, "keep")
    assert _chunk(p, B(["a"], [20000], [20000], [3616]), _st(1000, 0, 9), _st(1000, 999, 9)) == (False, True, "end")
    s = p.sessions[-1]
    assert s["chunks"] == 3 and s["kept"] == 2 and s["new_segments_kept"] == 0 and s["empty"] == {"end": 1}
    assert not p.holding


def test_policy_new_prompt_starts_from_an_empty_cache_and_its_own_reference():
    p = keep.Policy(slack=100)
    # first prefill after boot: cache state unknown -> empty BEFORE chunk 1, which becomes the reference
    eb, ea, why = _chunk(p, B(["a"], [20000], [8192], [8192]), _st(50, 0, 3), _st(1000, 990, 9))
    assert eb and not ea and why == "keep" and p.frag_ref == 10
    _chunk(p, B(["a"], [20000], [20000], [11808]), _st(1000, 0, 9), _st(1000, 990, 9))
    p.frag_ref = 777                                  # stale value from prompt a
    # decode mallocs between prompts -> prompt b empties first and measures its own reference
    eb, _, _ = _chunk(p, B(["b"], [30000], [8192], [8192]), _st(200, 0, 12), _st(1000, 950, 20))
    assert eb and p.frag_ref == 50


def test_policy_unknown_final_length_is_todays_behaviour():
    p = keep.Policy()
    p.emptied(_st())
    assert _chunk(p, B(None, None, [9000], None), _st(), _st(10, 9, 1))[1:] == (True, "unknown-final")


def test_policy_switch_to_another_prompt_empties_before():
    """P0-1 / P1-1: a cancelled prompt's held cache is emptied before any long chunk that does not continue it."""
    p = keep.Policy(slack=100)
    p.emptied(_st())
    _chunk(p, B(["a"], [1 << 20], [8192], [8192]), _st(), _st(1000, 990, 9))
    _chunk(p, B(["a"], [1 << 20], [16384], [8192]), _st(1000, 0, 9), _st(1000, 990, 9))
    assert p.holding
    # a cancelled, then prompt b
    assert p.begin(B(["b"], [40000], [8192], [8192]), _st(1000, 0, 9)) == (True, "switch")
    assert p.sessions[-1]["empty"] == {"switch": 1}
    # without request ids the final lengths stand in for them
    p = keep.Policy(slack=100)
    p.emptied(_st())
    _chunk(p, B(None, [1 << 20], [8192], None), _st(), _st(1000, 990, 9))
    assert p.begin(B(None, [40000], [8192], None), _st(1000, 0, 9))[0]


def test_policy_mixed_batch_does_not_carry_a_into_b():
    """P1-1: A's last chunk shares a batch with B's first chunk -> empty after it; B starts on an empty cache."""
    p = keep.Policy(slack=100)
    p.emptied(_st())
    _chunk(p, B(["a"], [20000], [8192], [8192]), _st(), _st(1000, 990, 9))
    _chunk(p, B(["a"], [20000], [16384], [8192]), _st(1000, 0, 9), _st(1000, 990, 9))
    eb, ea, why = _chunk(p, B(["a", "b"], [20000, 50000], [20000, 4576], [3616, 4576]), _st(1000, 0, 9),
                         _st(1000, 990, 9))
    assert eb and ea and why == "mixed"            # not a continuation (empty before) and A ends (empty after)
    assert not p.holding
    eb, ea, why = _chunk(p, B(["b"], [50000], [12768], [8192]), _st(0, 0, 9), _st(1000, 990, 15))
    assert not eb and why == "keep"                # B's own first kept chunk, from the emptied cache


def test_policy_row_change_gap_and_rewind_empty_before():
    """P1-1: more rows than the session's first chunk (bypass without solo), a gap, a rewind: empty first."""
    for nxt in (B(["a"], [100000], [16384 + 256], [8192 + 256]),     # +256 rows
                B(["a"], [100000], [24576], [8192]),                 # gap: prefix 16384 != 8192
                B(["a"], [100000], [8192], [8192]),                  # rewind / re-prefill
                B(["a"], [90000], [16384], [8192])):                 # same rid, other final length
        p = keep.Policy(slack=100)
        p.emptied(_st())
        _chunk(p, B(["a"], [100000], [8192], [8192]), _st(), _st(1000, 990, 9))
        assert p.begin(nxt, _st(1000, 0, 9)) == (True, "switch"), (nxt.seqs, nxt.extends, nxt.finals)
    p = keep.Policy(slack=100)
    p.emptied(_st())
    _chunk(p, B(["a"], [100000], [8192], [8192]), _st(), _st(1000, 990, 9))
    assert p.begin(B(["a"], [100000], [12000], [3808]), _st(1000, 0, 9)) == (False, "continue")   # short last chunk


def test_policy_release():
    p = keep.Policy(slack=100)
    assert not p.release()
    p.emptied(_st())
    _chunk(p, B(["a"], [100000], [8192], [8192]), _st(), _st(1000, 990, 9))
    assert p.release() and not p.holding and p.sessions[-1]["empty"] == {"released": 1}
    assert not p.release()


def test_policy_trip_and_big_trip():
    p = keep.Policy(slack=100)
    p.emptied(_st())
    a = lambda seq: B(["a"], [100000], [seq], [8192])  # noqa: E731
    _chunk(p, a(8192), _st(), _st(1000, 990, 4))                                        # frag_ref 10
    assert _chunk(p, a(16384), _st(1000, 0, 4), _st(1200, 1000, 5))[1:] == (True, "trip")  # slack 200
    # after a trip the next chunk starts from the empty cache: no empty before, a new reference
    assert _chunk(p, a(24576), _st(0, 0, 5), _st(1000, 990, 9)) == (False, False, "keep")
    assert _chunk(p, a(32768), _st(1000, 0, 9), _st(2000, 1000, 10))[1:] == (True, "trip-big")
    assert _chunk(p, a(40960), _st(0, 0, 10), _st(1000, 990, 14))[1:] == (True, "disabled")
    assert _chunk(p, B(["a"], [100000], [100000], [8192]), _st(0, 0, 14), _st(1000, 990, 14))[1:] == (True, "end")
    eb, _, _ = _chunk(p, B(["b"], [50000], [8192], [8192]), _st(0, 0, 14), _st(1000, 990, 18))
    assert not p.session["disabled"]                                                    # next prompt starts clean


def test_final_lens_and_continuation():
    class R:
        def __init__(self, end, n):
            self.extend_range = type("X", (), {"end": end})()
            self.origin_input_ids = list(range(n))
    assert keep.final_lens_of([R(8192, 20024), R(300, 300)]) == [20024, 300]
    assert keep.final_lens_of([object()]) is None
    assert keep.prompt_continues([8192, 300], [20024, 300]) is True
    assert keep.prompt_continues([20024, 300], [20024, 300]) is False
    assert keep.prompt_continues([1], None) is None


def test_padded_width():
    assert keep.padded_width(2048, None, 4) == 2048
    assert keep.padded_width(2049, None, 4) == 2052
    assert keep.padded_width(2048, 5006, 4) == 5008
    assert keep.padded_width(2048, 262144, 4, cap=32767) == 32768
    assert keep.padded_width(40000, 262144, 4, cap=32767) == 40000    # a short chunk past the cap: unchanged
    assert keep.padded_width(33000, 60000, 64) == 60032
    assert keep.lcm(4, 64) == 64 and keep.lcm(4, 6) == 12


def test_last_unsplit_lc():
    # chunks end at 8192, 16384, ...; split from end // 4 >= 32768, i.e. end >= 131072
    assert keep.last_unsplit_lc(8192, 8192, 20024, 4, 32768) == 20024 // 4
    assert keep.last_unsplit_lc(8192, 8192, 1048576, 4, 32768) == 122880 // 4
    assert keep.last_unsplit_lc(5000, 8192, 1048576, 4, 32768) == (5000 + 15 * 8192) // 4   # prefix-cache start
    assert keep.last_unsplit_lc(131072, 8192, 1048576, 4, 32768) == 131072 // 4             # already split


def _accounting(prompt):
    a = simulate(prompt, "A")
    c = simulate(prompt, "C")
    return a, c


def test_accounting_32k():
    a, c = _accounting(32768)
    assert c["reserved_max"] <= a["reserved_max"] + 16 * MIB, (a["reserved_max"] / MIB, c["reserved_max"] / MIB)
    assert c["kept"] >= 2 and c["trips"] == 0 and c["new_segments_kept"] <= 2, c  # small growing tensors (k gather, tail masks)


def test_accounting_131k_and_unpadded_growth():
    a, c = _accounting(131072 + 5000)
    b = simulate(131072 + 5000, "B")
    assert c["reserved_max"] <= a["reserved_max"] + 16 * MIB, (a["reserved_max"] / MIB, c["reserved_max"] / MIB)
    assert c["trips"] == 0 and c["new_segments_kept"] <= 2, c  # the unsplit -> split switch may need one
    assert b["reserved_max"] > a["reserved_max"] + 1 * GIB, (a["reserved_max"] / MIB, b["reserved_max"] / MIB)


def test_accounting_524k():
    a, c = _accounting(524288)
    assert c["reserved_max"] <= a["reserved_max"] + 16 * MIB, (a["reserved_max"] / MIB, c["reserved_max"] / MIB)
    assert c["new_segments_kept"] <= 2 and c["trips"] <= 1, c  # the unsplit -> split switch frees the 1 GiB unsplit logits
    assert c["kept"] >= 60, c


def test_accounting_1m():
    a, c = _accounting(1048576)
    # padding also drops candidate_block_ids' F.pad copy of the 2 GiB split logits, so C sits below A here
    assert c["reserved_max"] <= a["reserved_max"] + 16 * MIB, (a["reserved_max"] / MIB, c["reserved_max"] / MIB)
    assert c["new_segments_kept"] <= 2 and c["trips"] <= 1 and c["kept"] >= 120, c


def report():
    for prompt in (32768, 131072 + 5000, 262144, 524288, 1048576):
        a, c = _accounting(prompt)
        chunks = -(-prompt // CHUNK)
        line = (f"prompt {prompt:>8}: chunks {chunks:>3}  A reserved_max {a['reserved_max'] / GIB - 60:6.2f} GiB transient,"
                f" empties {a['empties']:>3} | C {c['reserved_max'] / GIB - 60:6.2f} GiB, kept {c['kept']:>3},"
                f" empties {c['empties']:>2}, trips {c['trips']}, new segments in kept chunks {c['new_segments_kept']}")
        if prompt <= 262144:
            b = simulate(prompt, "B")
            line += f" | B (keep, no pad) {b['reserved_max'] / GIB - 60:7.2f} GiB"
        print(line, flush=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[OK] {name}", flush=True)
    report()
    print("all passed")
