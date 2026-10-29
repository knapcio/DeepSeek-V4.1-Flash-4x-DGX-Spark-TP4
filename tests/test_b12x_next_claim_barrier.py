#!/usr/bin/env python3
"""The b12x_next dynamic-kernel producer loop's batch-claim race, as an exhaustive interleaving model (stdlib; CPU).

dynamic.py's producer loop (runtime/b12x_next/b12x_next/moe/_shared/kernels/dynamic.py) is, per CTA:

    while produce_active:
        if leader: batch_base = atomicAdd(pair_head, claim); st.shared ctrl[7] = batch_base
        bar.sync 0                                   # publication: the store before every read
        batch_base = ld.volatile.shared ctrl[7]
        [bar.sync 0, b12x d7937d7f backport]        # reuse: every read before the next store
        if batch_base >= limit: produce_active = 0
        else: per-warp work on tokens batch_base + warp

The model runs W warps (warp 0 holds the leader) as independent processes over one shared word, a global claim
counter and a generation-counting CTA barrier, and explores EVERY interleaving (depth-first with a visited set).
Invariant: each warp reads, in its iteration i, the value the leader stored in iteration i, so every claimed batch
is processed exactly once by every warp group (no token skipped, none duplicated, no early stop).

  stock (no reuse barrier)  -> a counterexample exists (a lagging warp reads the NEXT claim): the race is real
  fixed (reuse barrier)     -> no interleaving violates the invariant
  parity (two claim slots, alternating by iteration; zero extra barriers) -> also safe: the alternative noted in
                               the upstream issue draft

It also pins the patch in the vendored source: the barrier follows the claim read at the loop-body level (outside
every token / leader predicate), exactly once, and the stock barrier before the read is still there.
usage: python3 tests/test_b12x_next_claim_barrier.py
"""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _package_dir():
    """runtime/b12x_next/b12x_next in a checkout; the installed package (/opt/b12x_next) in the image."""
    here = os.path.join(ROOT, "runtime", "b12x_next", "b12x_next")
    if os.path.isdir(here):
        return here
    import importlib.util
    spec = importlib.util.find_spec("b12x_next")     # locates the package without importing it (no torch / cutlass)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("b12x_next not found (checkout runtime/b12x_next or PYTHONPATH)")
    return list(spec.submodule_search_locations)[0]


PKG = _package_dir()
DYNAMIC = os.path.join(PKG, "moe", "_shared", "kernels", "dynamic.py")


def explore(warps, batches, variant, limit_states=2_000_000):
    """Exhaustive search. Returns (counterexample trace or None, states visited).

    Per warp program counters: 0 claim (leader only; others skip), 1 arrive publication barrier, 2 wait publication,
    3 read claim, 4 arrive reuse barrier (fixed only), 5 wait reuse, 6 work/decide, then back to 0 or done.
    The claim counter hands out 0, 1, ..., batches-1 and then the terminal value `batches` (>= limit)."""
    slots = 2 if variant == "parity" else 1
    # state: (pcs, iters, vals, shared slots, head, bar_count, bar_gen, waits(gen per warp or -1), done)
    init = (tuple([0] * warps), tuple([0] * warps), tuple([-1] * warps), tuple([-1] * slots), 0, 0, 0,
            tuple([-1] * warps))
    stack = [(init, ())]
    seen = set()
    while stack:
        state, trace = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        if len(seen) > limit_states:
            raise RuntimeError("state space too large")
        pcs, iters, vals, shared, head, bcount, bgen, waits = state
        for w in range(warps):
            pc, it = pcs[w], iters[w]
            if pc == 99:
                continue
            npcs, niters, nvals, nshared, nwaits = list(pcs), list(iters), list(vals), list(shared), list(waits)
            nhead, nbcount, nbgen = head, bcount, bgen
            step = None
            if pc == 0:                                   # leader claims + stores; others go straight to the barrier
                if w == 0:
                    claim = min(head, batches)
                    nhead = head + 1
                    nshared[it % slots] = claim
                    step = f"w0 claim it{it} -> slot{it % slots}={claim}"
                else:
                    step = f"w{w} enter it{it}"
                npcs[w] = 1
            elif pc in (1, 4):                            # arrive at a CTA barrier
                nwaits[w] = bgen
                nbcount = bcount + 1
                if nbcount == warps:                      # last arrival releases the generation
                    nbcount, nbgen = 0, bgen + 1
                npcs[w] = pc + 1
                step = f"w{w} arrive bar{'pub' if pc == 1 else 'reuse'} gen{bgen}"
            elif pc in (2, 5):                            # wait until the generation advanced
                if bgen == waits[w]:
                    continue
                nwaits[w] = -1
                npcs[w] = 3 if pc == 2 else 6
                step = f"w{w} pass bar{'pub' if pc == 2 else 'reuse'}"
            elif pc == 3:                                 # read the claim
                nvals[w] = shared[it % slots]
                npcs[w] = 4 if variant == "fixed" else 6
                step = f"w{w} read it{it} slot{it % slots}={nvals[w]}"
            elif pc == 6:                                 # check: iteration i must see claim i (or the terminal)
                expect = min(it, batches)
                if vals[w] != expect:
                    return trace + (f"w{w} it{it} used claim {vals[w]}, the leader stored {expect} for it{it}",), \
                        len(seen)
                if vals[w] >= batches:
                    npcs[w] = 99
                    step = f"w{w} done"
                else:
                    npcs[w] = 0
                    niters[w] = it + 1
                    step = f"w{w} work batch {vals[w]}"
            nstate = (tuple(npcs), tuple(niters), tuple(nvals), tuple(nshared), nhead, nbcount, nbgen, tuple(nwaits))
            stack.append((nstate, trace + (step,)))
    return None, len(seen)


class ClaimRaceModelTests(unittest.TestCase):
    def test_stock_loop_has_a_counterexample(self):
        for warps in (2, 3):
            cex, n = explore(warps, batches=2, variant="stock")
            self.assertIsNotNone(cex, warps)
            self.assertIn("used claim", cex[-1])
        cex, _ = explore(2, batches=2, variant="stock")
        print("stock counterexample:", " | ".join(cex))

    def test_reuse_barrier_closes_it_exhaustively(self):
        for warps, batches in ((2, 2), (2, 3), (3, 2), (3, 3), (4, 2)):
            cex, n = explore(warps, batches, variant="fixed")
            self.assertIsNone(cex, (warps, batches, cex))
            self.assertGreater(n, 50)

    def test_parity_slots_are_an_alternative(self):
        for warps, batches in ((2, 3), (3, 3), (4, 2)):
            cex, _ = explore(warps, batches, variant="parity")
            self.assertIsNone(cex, (warps, batches, cex))


class ClaimBarrierSourceTests(unittest.TestCase):
    def setUp(self):
        with open(DYNAMIC) as f:
            self.src = f.read()

    def test_barrier_follows_the_claim_read_at_loop_body_level(self):
        m = re.search(r"\n( +)cute\.arch\.sync_threads\(\)\n\1batch_base = _ld_shared_i32\(ctrl_base_addr \+ Int32\(28\)\)\n"
                      r"(?:\1#[^\n]*\n)+\1cute\.arch\.sync_threads\(\)\n\1producer_limit = total_pairs\n", self.src)
        self.assertIsNotNone(m, "claim read not followed by the reuse barrier at the same (loop-body) indentation")
        body = m.group(1)
        loop = self.src.rfind("while produce_active > Int32(0):", 0, m.start())
        self.assertGreater(loop, 0)
        loop_indent = len(self.src[self.src.rfind("\n", 0, loop) + 1:loop])
        self.assertEqual(len(body), loop_indent + 4, "the barrier must sit directly in the while body")

    def test_exactly_one_claim_store_and_read(self):
        self.assertEqual(self.src.count("_st_shared_i32(ctrl_base_addr + Int32(28), batch_base)"), 1)
        self.assertEqual(self.src.count("_ld_shared_i32(ctrl_base_addr + Int32(28))"), 1)
        # the backport of b12x d7937d7f keeps upstream's comment verbatim
        self.assertEqual(self.src.count("All warps retain the claim before the leader can reuse its slot."), 1)

    def test_patch_recorded(self):
        def read(*parts, mode="r"):
            with open(os.path.join(ROOT, *parts), mode) as f:
                return f.read()
        patch = read("scripts", "b12x_next-claim-barrier.patch")
        self.assertIn("+            cute.arch.sync_threads()", patch)
        self.assertIn("d7937d7fc2e5511a0b61eb3572cd149f563f9b4f", patch)
        self.assertIn("Martin Vit", patch)
        self.assertIn("b12x_next-claim-barrier.patch", read("scripts", "build_b12x_next.sh"))
        import hashlib
        names = ["compact-n64-m64", "prequant-input", "barrier-zero", "det-triton-planner", "claim-barrier"]
        blob = b"".join(read("scripts", f"b12x_next-{n}.patch", mode="rb") for n in names)
        with open(os.path.join(PKG, "SOURCE_PATCH")) as f:
            source_patch = f.read().strip()
        self.assertEqual(hashlib.sha256(blob).hexdigest()[:16], source_patch)


if __name__ == "__main__":
    unittest.main()
