# Running CHUNKED_PREFILL_SIZE above 1024 without the long-prompt memory problem

Analysis only (2026-09-11). Nothing here is implemented or tested unless it says so; the
measurements come from the boots recorded in `logs/profile-2026-09-10/REPORT.md` §17-17b.

> Superseded for deployment: the shipped profile runs `CHUNKED_PREFILL_SIZE=4096` with the
> sglang#39187 indexer backport at `MEM_FRACTION_STATIC=0.80` (see `.env.tp4.example` and the
> [README](../README.md)). The sizes below are the pre-backport analysis that led there; do not
> copy them into `.env.tp4`.

## 1. Why the chunk size is tied to memory at all

On the head (rank 0, ~4.5-5.3 GB free after boot) a prefill's memory peak is set by the
DeepSeek-V4 low-ratio indexer, not by KV. For every prefill chunk, each of the ratio-1/2
source layers scores the chunk's `T` query rows against the whole compressed prefix `L`
(`deepseek_v4_backend._low_ratio_index_topk_dense` → `_dense_fp4_mqa_logits`, output
`[T, max_seqlen_k]` fp32, `max_seqlen_k` = the longest compressed length in the batch). Around
that one buffer the same function keeps, at the same moment:

| buffer | bytes per (row, key) | note |
|---|---|---|
| `logits` fp32 | 4 | the kernel output |
| candidate-block copy (`select_candidate_blocks`, layer 20 only) | up to 4 | "pads and pools a copy of its rows", capped at `_TORCH_INDEXER_SCORE_BUDGET_BYTES` = 1 GiB per call |
| `candidate_masks` bool | 1 | published by layer 20, consumed by every later low-ratio layer, lives for the whole chunk |
| `j` arange, `compress_lens` broadcast, top-k scratch | small | |

So the live peak of one chunk is roughly `c × T × L` with `c ≈ 14 B` measured (REPORT §17b:
2.97 GB at T=2048, L=100k; 2.6 GB at T=1024, L=208k; both ≈ 3.5 × 4 B × T × L, plus
~0.2-0.35 GB of touched KV pages), and the allocator's cache is emptied after each chunk by
`adapter/prefill_empty_cache.py`, so the sum over chunks no longer accumulates.

With a 1.5 GB guard and ~0.5 GB of margin the usable budget is about 2.5-3 GB, which gives
the rule that every measurement so far obeys:

```
T × L  ≲  2.0e8 token²            (T = chunk, L = prefix at that chunk)
2048 × 100k = 2.05e8  ok          2048 × 133k = 2.7e8  guard fired (boot 11, 128k)
1024 × 208k = 2.13e8  ok          1024 × 256k = 2.6e8  guard fired by 20 MB (boot 12)
```

Everything below is a way to raise `T` without letting `T × L` cross that line, or to move
the line.

## 2. Options, ranked

### 2.1 Adaptive chunk: 2048 while the prefix is short, smaller as it grows (recommended)

The scheduler already asks a `dynamic_chunk_sizer` for the size of every continuation chunk
(`scheduler.py get_new_batch_prefill`: `if self.chunked_req is not None and
self.dynamic_chunk_sizer is not None: chunked_prefill_size = sizer.predict(history_len)`,
where `history_len` is the prefix in tokens). SGLang only installs that sizer for pipeline
parallelism (`maybe_init_dynamic_chunk_sizer`, `pp_size > 1`), but the attribute is plain and
the contract is one method: `predict(history_len) -> Optional[int]`.

An adapter hook (sitecustomize on `sglang.srt.managers.scheduler`, ~30 lines) can install an
object whose `predict` returns

```
chunk(L) = clamp( round_down_to_page( BUDGET / L ), CHUNK_MIN, CHUNK_MAX )
BUDGET = 2.0e8 token²  (env knob), CHUNK_MAX = 2048, CHUNK_MIN = 256, page = 256
```

which yields a schedule of

| prefix so far | chunk | prefill rate (measured or extrapolated) |
|---|---|---|
| 0 - 98k | 2048 | 2.0-2.2k tok/s (measured) |
| 98k - 195k | 1024 | 1.5-1.7k tok/s (measured) |
| 195k - 390k | 512 | ~1.1-1.3k tok/s (extrapolated; boot 13 was stopped before measuring) |
| 390k - 780k | 256 | ~0.8k tok/s (extrapolated) |

Consequences: every prompt under ~100k tokens prefills at the full 2048 speed (today all of
them run at 1024, 15% slower); a 200k prompt takes about the same as today; 256k-500k
prompts become possible at all (est. 500k ≈ 7 min, vs. impossible), and `CONTEXT_LENGTH`
could go back to 512000 with the same memory guarantee. Decode is untouched (the chunk
size never affects decode steps).

Edge to handle: the first chunk of a request is sized statically (the sizer is consulted
only for `chunked_req`, i.e. continuation chunks). A brand-new request has `L ≈ chunk`,
fine; a request that lands on a long *cached* prefix (radix hit of, say, 150k tokens) gets
its first chunk at 2048 with `L = 150k`, over budget. Two fixes: clamp inside
`PrefillAdder.add_one_req` (it knows `req.extend_input_len` and the prefix length), or set
the static size to 1024 and let the sizer raise it to 2048 when `L` is small (the sizer's
return value overrides the static size in both directions). The second needs no adder hook.

Cost: none on decode; on prefill only the chunks beyond 98k get smaller than today's 1024
beyond 195k. Validation: one boot, ramp 64k / 128k / 200k / 256k / 400k / 500k with
`scripts/verify/memguard.py`, check prefill tok/s per band and that `lowest MemAvailable`
stays above 2 GB at every size.

### 2.2 Lower the candidate-selection copy budget (cheap, moves the line)

`_TORCH_INDEXER_SCORE_BUDGET_BYTES = 1 << 30` bounds the padded copy that layer 20 makes for
block selection; at large `T × L` that copy is a full extra `[T, L]` fp32 (0.8 GB at
2048 × 100k). It is a module constant, so the adapter can set it to 256 MiB at import
(`module._TORCH_INDEXER_SCORE_BUDGET_BYTES = 256 << 20`); the loop then processes fewer
rows per step with identical results. Expected: `c` drops from ~14 B to ~11 B per (row,
key), i.e. the budget rises from ~2.0e8 to ~2.6e8 token² and 2048-token chunks reach
~130k prefix instead of ~100k. Free to try; combine with 2.1 by raising `BUDGET`.

### 2.3 Block the indexer along the key axis and merge top-k (engine change, memory-flat)

The exact answer: score the `T` rows against key blocks of `B_k` (e.g. 32k), keep the
per-block top-512 (values + indices), then take the top-512 of the concatenated candidates.
`topk(union) == topk(∪ per-block topk)` holds exactly, so outputs do not change. Memory
becomes `c × T × B_k`, independent of the prefix: 2048-token chunks (or 4096) at any
context. The candidate-source layer (block pooling over 8-key blocks, top-2048 blocks) is
decomposable the same way, and the consumer layers' `masked_fill_` of non-candidates can
be applied per block. Touch points: `_low_ratio_index_topk_dense`,
`_publish_or_consume_candidates`, `topk_transform_ragged_v2` (needs per-block offsets or a
second pass), `_mask_topk_scores`. Effort: a day or two in a 180 KB file plus a correctness
diff against the unblocked path on the 32k sweep. This is the upstream-quality fix; 2.1 is
what to run in the meantime.

### 2.4 Context-parallel prefill over the three ranks (exists in the engine, untested here)

The engine carries a DSA prefill context-parallel mode (`--enable-dsa-prefill-context-parallel`,
`dsa_prefill_cp_mode round-robin-split`, `attn_cp_size`; backend paths
`_forward_low_ratio_sources_cp`, `_late_layer_tail_cp_layout`). It splits the query rows of a
prefill across ranks, so each rank scores `T / cp_size` rows: at `cp_size = 3` the per-rank
buffer is a third, i.e. the same `T × L` line moves to 6e8 and 2048-token chunks reach
~300k. Unknowns on this stack: whether CP composes with TP3 padding (rank 2's all-padding
attention shard), the late-layer tail, DSpark's per-chunk hidden-state injection, and
whether the gathers cost more than the GEMMs save. Worth one boot with the flags and the
32k sweep for correctness before any ramp; not something to enable blind.

### 2.5 More headroom on the head (small, additive)

- The KV pool does not help: its pages are touched only when used, so shrinking
  `MAX_TOTAL_TOKENS` frees nothing at rest and costs concurrency.
- NCCL buffers are already trimmed (139 MB pinned).
- The tokenizer, detokenizer and HTTP server (~1.3 GB of rank-0-only RSS) cannot move to a
  worker in this SGLang build without a proxy layer; that is the one real reserve.
- The docker healthcheck process (~0.6 GB RSS each time it ran the engine's imports) is
  gone since the `-S` fix; it used to bite exactly during long prefills.
- Loosening the 1.5 GB guard is not a lever: the head survived at 1.0-1.3 GB in tests, but
  the failure past zero is a host wedge, and 0.5 GB of margin is thin at 1-second sampling.

### 2.6 What does not work

- `expandable_segments:True` (would coalesce the fragmentation) returns NaN logits on every
  >64-token prefill on this stack (REPORT §17). Banned.
- `garbage_collection_threshold` is a no-op: SGLang never calls
  `set_per_process_memory_fraction`, which that option requires.
- A static chunk of 2048 with the empty-cache hook alone: 100k prompts fit, 128k do not
  (boot 11).
- Emptying the cache more often than once per chunk: the peak is inside one layer's
  forward, not between layers.

## 3. Recommendation

Implement 2.1 with 2.2 folded in (two adapter hooks, no engine edit, no rebuild of anything
but the image layer that copies `adapter/`), set `CHUNKED_PREFILL_SIZE=2048` as the maximum
and let the sizer descend, restore `CONTEXT_LENGTH=512000`, and verify with the ramp above.
Expected result: 2048-speed prefill for all prompts up to ~100-130k, long prompts to 500k
without touching the guard, decode unchanged. Keep 2.3 as the follow-up that removes the
prefix dependence for good.
