# Issue #16: decoder SWA / prefix-cache CUDA faults

Status: **open; v2.4 quarantined; source-level candidate ready; GPU soak pending**.

## Evidence

The external report contains four Xid 13/43 failures under large prefix-cache hits (roughly
200k-966k cached tokens with a small fresh suffix). The first two used
`DSV41_PREFILL_BYPASS=256`; the next two reproduced with bypass disabled and 4096-token
prefill chunks. That rules out bypass as the sole cause. The reporter saw the v2.1-era stack
run for more than 20 hours before the upgrade and again after rollback, so the current fault
domain is the v2.2-v2.4 delta or an older latent SGLang bug made reachable by that delta.

The recipe pins SGLang at `f80c91a4b97e52e17fbcff088aa6b47b6270f6b9`. Upstream later
merged [#41303](https://github.com/sgl-project/sglang/pull/41303), which fixes two relevant
decoder bounded-replay cases:

1. an ordinary radix-cache match now holds back one complete SWA window, ensuring late-layer
   window KV is written for this request; and
2. a short final extend uses `seq_len - tail_len` as its attention floor rather than
   `seq_len - min(tail_len, extend_len)`, retaining rows written by the prior extend.

`patches/sglang-41303-bounded-replay-window.patch` is a narrow adaptation of that upstream
change to the pinned request-matching layout. It does not include unrelated post-pin SGLang
changes. `scripts/fetch-sglang-canary.sh` applies it with `--fuzz=0`, and
`scripts/verify_sglang_bounded_replay.py` makes staging and both canary image builds fail if
any required hunk is absent.

This path match is strong enough to test first, but it does **not** prove that #41303 caused
the reported illegal address. Do not close the issue from source inspection alone.

## Safe operating profile

Until the controlled soak passes, use the v2.1-compatible block in `.env.tp4.example`:

- 4096-token prefill chunks and memory fraction 0.80;
- no `DSV41_PREFILL_BYPASS`, `DSV41_SPEC_SYNC_FREE`, `DSV41_EAGER_GLUE`,
  `DSV41_SPLIT_COMPACT_GATHER`, `DSV41_CERT_HEAD`, or `DSV41_L2_PREFETCH_WOA`;
- no `--prefill-decode-interval 2` or `--swa-prefix-tails 128`.

The historical v2.4 numbers remain documented, but that block is explicitly quarantined.

## Controlled qualification plan

Run only in an announced maintenance window; preserve kernel logs after every stage.

1. Build the canary image from a clean checkout. Confirm the build prints
   `SGLang #41303 bounded-replay safety backport: verified`.
2. Boot the v2.1-compatible profile with the #41303 backport and run the standard smoke,
   qeval, prefix scan, 1M needle, then the issue reproducer with alternating 200k-1M cached
   prefixes and short fresh suffixes.
3. Soak for at least 24 hours or the reporter's previous failure volume, whichever takes
   longer. Record request mix, cache-hit sizes, restarts, allocator errors and kernel Xids.
4. If clean, add v2.2-v2.4 deltas one at a time in this order: sync-free, eager glue, compact
   gather, L2 `wo_a` prefetch, certified head, prefix-tail reserve/cadence, 8192-token chunks,
   and bypass last. Return to the clean control between candidates.
5. If a stage fails, reproduce once with that single delta removed before assigning blame.

Acceptance requires zero CUDA errors, Xids, worker exits, empty replies or quality regressions;
stable decode/prefill performance is secondary. Keep issue #16 open until the clean soak and
one re-enable pass are attached as raw results.
