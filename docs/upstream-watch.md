# Upstream watch (updated 2026-10-22)

What the pieces we run on are pinned to, why, and what would let us move. Checked against
sgl-project/sglang, DeepGEMM, b12x, HuggingFace and the other public Spark recipes on 2026-09-18.

## 2026-10-22 delta

- [SGLang #42901](https://github.com/sgl-project/sglang/pull/42901) is the strongest small backport
  found for the active decoder-bounded-replay profile. It writes the last KV-source layer's compressed
  KV for the full prompt, then runs layer 20's indexer, attention and MoE only on the final 128 rows.
  Upstream reports 5-10% lower cold-prompt TTFT at 16K-200K on GB300 with no short-prompt change.
  `patches/sglang-42901-layer20-tail.patch` adapts the change to the pinned pre-`HcState` code and is
  zero-fuzz/source-order verified. It has not yet received a GB10 boot, speed or model-quality gate.
- [SGLang #44001](https://github.com/sgl-project/sglang/pull/44001) is a small scheduler-only addition:
  the prefill/decode interval can be delayed until the running batch reaches a chosen size. Its runtime
  change is backported with a zero default and offline behavior tests; `min-bs=4` is a candidate for a
  focused multi-user A/B, not a production default and not a C1 decode-throughput claim.
- The 2026-10-22 SM12x review found that #45745's in-place torch-indexer scoring does not touch this
  recipe: our #39187/SG18 path calls the fused FP4 scorer and already bounds its output. #45264 explicitly
  excludes SM121 because it was measured only on SM120, while #45391 and #43008 duplicate the b12x MXFP8
  routing and exact FP8 `wo_a` copy already shipped here. They stay on the watch list instead of being
  stacked blindly.
- SGLang #45234/#45247/#45275 remove host synchronizations from encoder-SWA replay. The production
  command enables decoder replay, not encoder replay, so they do not affect the served path and were not
  copied merely to increase patch count.
- SGLang #44632 and #45633 primarily improve larger verify/query row counts. The current C1 path already
  falls inside the old fused limits; they are concurrency candidates, not a route from ~90 to 100 C1 tok/s.
- MiaAI-Lab's latest public recipe head (`64b02de`, 2026-10-17) adds switchless TP4 packaging and safety
  work but no newer decode lever missing from this repository. J-M's Round-2 SGLang sweep found a 1 GiB
  indexer-logits budget useful for memory robustness, while its scheduling changes did not improve decode.

## Pinned, and why the pin stays

| Piece | We run | Upstream now | Why we stay |
|---|---|---|---|
| SGLang `dsv4.1` branch | `f80c91a4b` (2026-09-16) | head `fc3107f0d`, 176 commits later, not merged to main, no tag; umbrella PR #38798 says the branch is "being refactored and is unstable" | every head after #39671 (2026-09-16 22:15) raises `the candidate indexer needs DeepGEMM's paged sparse MQA logits` on SM121: the torch fallback was removed and the sparse kernel exists only for SM100 (DeepGEMM #432). The fix is #39914 (restores `candidate_torch.py`, base main, CI failing, untested on SM121) |
| sglang#39187 (bounded indexer prefill) | backported in `adapter/indexer_chunked_v3.py` | still OPEN and conflicting; successors #40015 / #39095 also open | nothing merged supersedes it |
| sglang#39704 (mHC medium batches) | not used | OPEN, conflicting | ±2 % here (c ≤ 16) |
| `sglang-kernel` / `sgl-deep-gemm` | 0.4.7 / 0.2.0 | latest (2026-09-13 / 09-14) | nothing newer |
| `lmsysorg/sglang:dev-dsv41` image | base for `Dockerfile` | last push 2026-09-11 (commit da64c5cbb, 684 commits behind our pin) | older than what we build |
| RoCEnante (b12x, rhys101 SG17) | pinned overlay | b12x #313 (graph-replay wedge) still open and unanswered; #383 fixed startup prep on 4x GB10 | `SGLANG_ROCE_ALLREDUCE=0` stays the escape hatch |
| Prefill TP split (rhys101 SG18) | `adapter/spark_prefill_dense.py` | rhys101 has no commits since 2026-09-14, no SG19 | nothing to port |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40` | unchanged since 2026-09-10; no V4.2, no new draft head anywhere on HF | nothing to re-download |
| DSpark block size | 5 | Mia's default is 3; two independent fleets (Mia #20, #21) measured k=3 −16 to −26 % on code/structured | keep 5 |

## Seen upstream, same idea as ours

- sglang#39365 (open draft): read only the shards holding the DSpark head. Same as the draft
  half of `adapter/fast_load.py`.
- sglang#39666 (merged to main): Engram host table with `posix_fadvise(DONTNEED)` of the checkpoint
  page cache after load, the same reason `fast_load` drops it (the KV pool is sized from
  `MemAvailable`).
- sglang#39482 (merged to main 2026-09-18): SM121 added to DeepGEMM's UE8M0 scale selection.
  Only matters on a DeepGEMM MoE path; ours is FlashInfer CUTLASS / b12x.
- vLLM #57432 (merged): DSpark's non-causal draft window let padded keys into the softmax
  (+20 % acceptance on GSM8K after the fix). Not our bug: SGLang's SM120 sparse kernel pads
  indices with −1 and masks them (`flash_mla_sm120.py`, `invalid_mask` → `-inf`).

## Leads measured here on 2026-09-18

- **`--sleep-on-idle`: adopted.** Head scheduler idle CPU 47 % → 14 %, workers 5 %; first
  response after 20 s idle 0.18–0.21 s either way; decode unchanged.
- **Engram gap (the "prestage" lead): closed, shipped as `adapter/engram_prefetch.py`.** Live
  profile of 40 decode steps at c1: 2.2–2.4 ms of the 3 ms idle per step sat before the two
  `_engram_gather_kernel` launches on real text (1.0 ms on sparkDash prose). Not the callback
  mechanism (a host node in a replayed graph costs ~2 µs, the pinned H2D copy 26 µs): it is NVMe
  miss latency, ~200 µs per O_DIRECT 8 KB `pread` on the packed shard, served inside the host
  callback while the graph waits. Parallelising the misses in `row_store.cpp` changed nothing.
  What worked: the hash ids for both Engram layers exist at the start of the forward, so the
  adapter forks a side stream right after `EngramHasher.forward`, runs the ids copy, the host
  lookup and the row copies there for every layer, and joins only at the layer's gather. Inside
  the CUDA graph that is a fork/join branch with a host node. Same-image A/B, identical prompts:
  step 51.69 → 49.49 ms, idle 2.97 → 0.85 ms/step, gap 2.35 → 0.13 ms, English essay c1
  42.6/43.2/42.0 → 45.1/46.4/43.7 tok/s, sparkDash prose c1 57.1/59.0 → 60.3/60.7, prose c4
  119.9 → 123.8, structured 116.1 → 121.3. Check mode (old path re-run after every prefetched
  gather) counted 0 differing gathers across benches and the 131k needle.
- **Markov W2 top-k pruning (vLLM #56694).** The bf16 vocab-sized GEMMs (LM head + Markov W2)
  are 2.8 ms/step in the same profile, so the pruning can recover at most ~1–1.5 ms/step
  (2–3 %), and it changes the draft distribution on prose, our weakest column. Not built.
- **vLLM #57432** (padded keys in the DSpark softmax): not applicable, see above.
- Real-text reference for the prose column: a 700-token English essay ("history of Krakow",
  temperature 0, warm row cache, engine otherwise idle) decodes at 44–46 tok/s at c1, against
  57 tok/s on sparkDash's prose filler; the same essay in Polish 36–40 tok/s. Acceptance, not
  Engram, is the difference. Any run with another request in flight (`running-req: 2` in the
  log) reads 20–40 % lower, so check the log before quoting a number.

## Measured here on 2026-09-19 (fresh boot, four ranks, production image)

- **RoCE all-gather: closed.** Live-profiler trace of 42 decode steps (step wall 51.8 ms mean):
  NCCL is 0.6 ms/step in total, split `all_gather` 0.288 ms (265 calls), `all_reduce` 0.278 ms
  (83 calls), `broadcast` 0.027 ms. So the whole all-gather family is 0.56 % of a step, and a RoCE
  port of it (the all-reduce port bought 2.8x on the collective) has a ceiling of ~0.35 %. Not worth
  the port; tonyd2wild's aggregate gain must come from the other items in his speed run.
- **Markov head rank, priced.** `markov_w2` is read once per block position (5 per step).
  Measured on one GB10 with the real TP4 shard (`vocab/4 = 32320` rows, bf16), median of 200 calls:

  | rank | w2 per rank | effective | ms/step | vs rank 256 | share of a 51.8 ms step |
  |---:|---:|---:|---:|---:|---:|
  | 256 (today) | 15.8 MiB | 712 GB/s | 0.108 | – | 0.21 % |
  | 320 | 19.7 MiB | 737 GB/s | 0.131 | +0.023 | +0.04 % |
  | 384 | 23.7 MiB | 551 GB/s | 0.210 | +0.102 | +0.20 % |
  | 448 | 27.6 MiB | 262 GB/s | 0.515 | +0.407 | +0.79 % |
  | 512 | 31.6 MiB | 240 GB/s | 0.641 | +0.533 | +1.03 % |
  | 1024 | 63.1 MiB | 236 GB/s | 1.306 | +1.198 | +2.31 % |
  | 2048 | 126.2 MiB | 220 GB/s | 2.797 | +2.689 | +5.19 % |

  The head is free only while it fits the 25.2 MB L2: up to rank 320 the read runs at 712–737 GB/s,
  at 448 it has fallen to DRAM speed. So a retrained drafter should take **rank 320–384 first**
  (1.25–1.5x capacity for 0.04–0.2 % of a step); rank 1024 must buy back 2.3 % in accepted length
  before it pays for itself.
- **Clock cap is not in force.** Under load all four nodes sit at 2509–2554 MHz (util 96 %, 57–59 °C),
  idle at 2411 MHz, with `clocks.applications.graphics` 2418 MHz — not the 2200 MHz the cap script
  applies. `spark-clock-cap.timer` is active but `systemctl list-timers` shows LAST 04:40 and no NEXT,
  and the cap log's last line is 2026-09-14. Nothing is latched low (tonyd2wild's failure mode), so the
  fleet is running fast; the comparability the cap was meant to give is what is missing. All recent
  A/B pairs were taken in this same uncapped state, so they remain internally valid.
- **Fresh-boot prefill baseline** for the long-uptime check (unique prefixes, thinking off, c=1):

  | prompt tokens | 7,018 | 30,020 | 57,019 | 119,019 | 231,018 |
  |---|---:|---:|---:|---:|---:|
  | TTFT s | 2.08 | 7.77 | 13.01 | 29.11 | 59.10 |
  | prefill tok/s | 3,371 | 3,862 | 4,383 | 4,089 | 3,909 |

  Re-run `diagnostics/dsv41-backlog-20260919/prefill_probe.py` after a multi-day uptime and compare
  before trusting any tuning on a long-running fleet.

### Also settled on 2026-09-19 (full session in [session-20260919.md](session-20260919.md))

- Draft block size: 3 gives prose 33.8 / code 80.4 tok/s, 5 gives 44.6 / 102.7, 7 gives 28.9 / 106.8.
  Block 5 stays; perfect per-request adaptation (the lever that pays on GLM/vLLM) would buy +4 % on
  code and nothing on prose, so it is not worth the engine work here.
- Chunked prefill 1024: prefill -34 to -38 %, and the decoders behind a long prefill got *worse*
  (5.7 vs 7.8 tok/s). Chunk 4096 stays.
- The clock cap is nearly free: 2200 MHz vs uncapped costs 1.5 % prose, 0.7 % code, 1.5-5.0 % prefill.
- Prefix cache and Engram row cache are both healthy (35k prompt repeat 8.36 s -> 0.33 s, 99.4 %
  cached; Engram 99.8 % hit). Decode is flat to 247k resident context.
- Draft acceptance is 2.30 tokens/step on prose against 5.31 on code and does not move with
  concurrency, which is the same drafter gap the Markov work targets.

## Leads not yet tried here

- LuZ-0.1.7 (luxingcom, 4x Spark TP4 ring): fused ratio-1 decode (RMSNorm + RoPE + FP4 quant +
  FlashMLA write), fused ratio-2 pair pooling, FP4 storage for ratio-1 latents. Their own
  OFF/ON table moves code 100.3 → 99.8, aggregate c1–c12 by +1 %, prose 33.3 → 36.6 tok/s
  (from a base below our real-text 44–46); the kernels are not bit-exact (fp32-ulp deltas by
  their own note) and are grafted onto a different SGLang commit + FlashInfer 0.6.18. On our
  budget the fusable tiny kernels are ~3.8 ms/step and the hc kernels 3.6 ms, so the ceiling
  here is a few percent for a large, version-bound port. Not planned.


## Watch-outs from other fleets

- tonyd2wild speed run 2: two of his four nodes sat on a latched low GPU clock after a long run
  (code 32.9 → 57.1 tok/s only after a 30–60 s power cut). Our 10-minute timer re-applies the
  2200 MHz cap but does not detect a latch; after any multi-day uptime read
  `nvidia-smi -q -d CLOCK` on all four before trusting a benchmark.
- tonyd2wild speed run 2: a lane that had served 15 h ran ~30 % slower on prefill than the same
  config fresh (cause open, a restart recovers it). Not measured here; before tuning anything on a
  long-running fleet, run the prefill sweep and compare with the fresh-boot rows in [history.md](history.md).
- Mia #23: host buddy-allocator fragmentation after the weight load on GB10 (`NV_ERR_NO_MEMORY`
  with free memory, SSH dead, ICMP alive, power cycle). Reproduced on a 4x TP4/EP2 SGLang fleet at
  512k context; same signature as our 2026-09-18 Spark_02 wedge.
- Mia #14: the September OTA's `nvidia-spark-grub-kho` (`kho=off`) caps ConnectX-7 transmit at
  12.7 Gb/s on 6.17.0-1032. Not present on our four nodes as of 2026-09-18; check
  `/proc/cmdline` after any OTA.
- Mia #21: sparkDash's prefill filler is one repeated token, so the Engram row cache inflates
  that column; random-text rows are the ones to quote.
