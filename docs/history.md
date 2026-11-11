# Measurement history

Every dated table, note and rejected experiment that used to live in the README, moved here unchanged when the README was reduced to the current state (2026-09-24). What changed in which commit is in [CHANGELOG.md](../CHANGELOG.md); the current numbers are in the [README](../README.md#current-results).

The rows from before the sparkDash 1.8.8 sections were taken with sparkDash up to 1.8.6. sparkDash 1.8.7 replaced the code prompts (c1 and the concurrent waves), so those code columns are not comparable with the 1.8.8 tables; the prose, structured and prefill prompts are unchanged. From v2.6 (2026-10-27) the README reports speedbench only.

## v2.7.3 release, 2026-11-10, speedbench (the README's current results until v2.7.4)

The README's v2.7.3 results and release gate, moved here (headings demoted, links adjusted, text unchanged) when v2.7.4 was released on 2026-11-11. The v2.7.4 numbers are in the [README](../README.md#current-results).

**v2.7.3** (2026-11-10, in production on the four Sparks since 2026-11-10 18:00) is v2.7.2 plus two lossless prefill changes and one lossless decode change in the production profile. The two prefill changes are checked bit for bit on every rank before use: the prefill MoE input is all-gathered as the MXFP8 bytes and scales b12x would compute itself instead of bf16 rows (`DSV41_PREFILL_SP_FP8_MOE=1`, checked per chunk row count), and eight measured prefill dense GEMM shapes run FlashInfer's CUTLASS SM120 MXFP8 GEMM (tactic 5) instead of the b12x call (`DSV41_PREFILL_GEMM_TACTIC=1`, checked per shape at boot). The decode change moves the attention-core L2 windows: the window after `wq_b` warms 6 MB of `wo_a` again and a second window forked at the `wo_a` call warms 10 MB of `wo_b` while `wo_a` runs from L2, with evict-first loads on `wo_a`'s single-use twin (`DSV41_L2_PREFETCH_WOA_MB=6 DSV41_L2_PREFETCH_WOB_MB=0 DSV41_L2_PREFETCH_WOB2_MB=10 DSV41_WO_A_W8_EVICT_FIRST=1`; cache hints only, outputs bit-identical). In its release gate against two v2.7.2 control boots on the official checkpoint: cold prefill **+3.9 %** at 64K, **+4.3 %** at 131K and **+3.8 %** at 1M (against the mean of the controls), and a cold 388K prompt's first token in 57.3 s instead of 59.8-60.0 s (**+4.4 to +4.8 %**); decode greedy c1 x1.007, c4 x1.006, c8 x1.003, c16 x0.990 (the decode step time at c8 and c16 at parity), sampled c1 x0.990 and c4 x1.009, speedbench load at c16 x0.995. Outputs: qeval 71 / 71 / 71, all 75 T = 0 qeval prompts inside the controls' own spread, KLD 0.005 short / 0.004 long (limit 0.035), draft acceptance on qeval identical on the three boots, reasoning length identical. The gate's new exactness check fired on the 64-token greedy continuation of one 131K document; on review that is the engine's own T = 0 run-to-run variation at 131K, which the two control boots happened not to show in this gate, not a change in the numerics (see [the release gate](#v273-release-gate)). The image stack, the KV pool (8,541,952 tokens at fraction 0.85), the launch line and every other v2.7.2 key are unchanged.

#### speedbench

[speedbench](https://github.com/knapcio/speedbench) 0.3 (`6733653`), full mode: 73 frozen prompts in six categories (prompt set speedbench-v1), 256 output tokens, thinking off, run on the head. v2.7.3 release-gate boot (2026-11-10, official 1017 checkpoint, KV pool pinned at 8,541,952 tokens as in the production profile), mean of two full runs per greedy cell and one run per sampled cell; c1 is the decode tok/s geometric mean over the prompts, c4 to c16 the aggregate tok/s:

| tok/s | all 73 prompts, c1 | prose, c1 | c4 aggregate | c8 aggregate | c16 aggregate |
|---|---:|---:|---:|---:|---:|
| greedy (temperature 0) | **90.4** | 67.2 | **136.7** | 172.2 | **213.6** |
| sampled (temperature 0.7, top-p 0.95, seed 1) | 87.0 | 64.7 | 134.7 | – | – |

The two v2.7.2 control boots of the same gate gave greedy c1 89.6 / 90.0, c4 136.0 / 135.7, c8 171.4 / 172.0, c16 215.1 / 216.3 and sampled c1 87.2 / 88.4, c4 133.2 / 133.9 tok/s. Speedbench load at c16 gave 226.8 aggregate tok/s, median TTFT 466 ms, per-stream median 17.2 tok/s (controls 228.3 / 227.6 tok/s, 473 / 464 ms). The two prefill changes do not touch decode; the decode L2 layout shortens the c1 step (paired per prompt -0.27 ms, CI95 -0.29 to -0.25) and leaves the c8 / c16 step time at parity (127.2 ms against 127.0 / 127.4 at c8, 195.1 against 194.3 / 194.7 at c16); c16 tok/s moves with the draft acceptance of the T = 0 text at concurrency, which differs by ~1.3 % between identical controls.

Per category, greedy c1 (same boots, mean of two full runs):

| tok/s | prose | code | structured | agentic | multilingual | long context | all 73 prompts |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2.7.3 | **67.2** | 113.8 | 112.7 | 99.1 | 72.4 | 92.0 | **90.4** |
| v2.7.2 (two control boots) | 66.7 / 66.7 | 113.1 / 113.0 | 111.7 / 111.9 | 99.5 / 99.7 | 71.5 / 71.8 | 89.1 / 91.6 | 89.6 / 90.0 |

v2.7.3 runs: all 89.84 / 90.98, prose 66.72 / 67.71, code 113.22 / 114.42, structured 112.09 / 113.30, agentic 98.79 / 99.36, multilingual 71.83 / 72.94, long context 90.99 / 92.91. Median time to first token 166.3 / 164.9 ms (controls 166.2-167.7 ms); 73 of 73 prompts valid in both runs, none failed. Agentic and long context vary most from run to run.

**Under load** (v2.6, one fresh boot of the v2.6 image and profile, the last full load sweep; speedbench load: closed loop, 90 s per level, 256 output tokens, thinking off)

| concurrency | block tok/s | per stream, median tok/s | TTFT, median |
|---|---:|---:|---:|
| c1 | 77.3 | 83.3 | 207 ms |
| c4 | 131.7 | 38.7 | 310 ms |
| c16 | **211.0** | 16.0 | 539 ms |

Block tok/s is all completion tokens over the wall time of the level, including request start and prefill, so the c1 value sits below the decode geomean above. The v2.7.3 release gate ran load at c16 only: 226.8 aggregate tok/s, median TTFT 466 ms, per-stream median 17.2 tok/s.

**Prefill, cold** (v2.7.3 release gate, the gate's own probes, not speedbench: 64K, 131K and 1M are the median of three flushed prompts of 62K, 127K and 969K tokens; 388K is the first token of a cold 388,000-token prompt, median of three)

| tok/s | 64K | 131K | 388K | 1M (969K tokens) |
|---|---:|---:|---:|---:|
| v2.7.3 | **6,726** | **6,851** | **6,774** (57.3 s) | **5,697** |
| v2.7.2 (two control boots) | 6,469 / 6,481 | 6,554 / 6,584 | 6,467 / 6,487 (60.0 / 59.8 s) | 5,431 / 5,545 |

The needle tests at 248K and 969K tokens passed (36.2 s and 168.6 s). The four Sparks serve the abliterated variant in production ([`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) `517efab`); its edit touches only `attn.wo_b` and `shared_experts.w2` of the main layers (the DSpark draft layers are identical), so the two run within a few percent of each other. The v2.7.2, v2.7.1 and v2.7 tables are in [docs/history.md](history.md).

#### v2.7.3 release gate

v2.7.2 / v2.7.3 / v2.7.2, three fresh boots in that order on the same four Sparks, official 1017 `1bae0e1` on all three, GPU clock capped at 2200 MHz, 2026-11-10 (run `run-20261110-114511-236731`, 362 min including the restore of production). The controls are production v2.7.2's image and profile; the v2.7.3 boot ran a gate image of `9413b93` (production v2.7.2's image with this release's changed files added) and the production profile, whose only differences are the six keys above; KV pool 8,541,952 tokens on all three. Every speed, prefill, memory, qeval, idle and latency rule passed (decode, sampled, load, decode step time, c1 step, cold prefill, memory, idle CPU, idle first token, inter-token gaps, long-prompt decode, 32K exactness), 0 safety, 0 invalid. qeval 71 / 71 / 71 with the same four failed tasks on every boot, all 75 T = 0 prompts inside the controls' bracket, KLD 0.0054 short / 0.0039 long against the controls (limit 0.035), the sampled-token distribution test p 0.949, the 16 thinking-on reasoning prompts the same length on all three boots. Draft acceptance pooled over the speedbench prompts x1.001 / x1.004 of the lower control at c1 / c4 (c8 x1.004 and c16 x0.998, reported only from this gate on: at concurrency the T = 0 text, and with it the acceptance, varies ~1.3 % between identical controls), and on qeval identical on the three boots (3.979). Long-prompt decode (flushed 32K and 131K documents, each followed by a 512-token greedy continuation) 101.8 tok/s (controls 98.9 / 100.8) at 4.00 tokens per step (3.92 / 4.00). Head memory low-water 8.43 GiB (controls 8.50 / 8.46), during the cold 1M prompts 8.68 GiB (8.71 / 8.65); idle first token 191 ms median (193 / 194); load c16 p99 inter-token gap 286 ms (286 / 290); a short prompt beside a cold 131K prefill got its first token in 0.29 s (0.33 / 0.33); the issue-16 soak 308 / 308 requests ok and its companion load 230 requests, 0 failed; no Xid on any boot. Every rank logged the GEMM tactic's boot check 8 / 8 shapes bit-identical and agreed, the MoE-input gather switched on for every chunk size with every self-check passing, and the same W2 windows on the four nodes. A deliberately cold boot of v2.7.3 (tuning caches off) was ready in 182 s with no Xid, core dump or fault line.

The verdict text was **FAIL on one rule, exactness at 131K** (new in this gate: flushed 32K, 131K and 1M documents, the first token's top-20 logprobs and a 64-token greedy continuation compared with the controls). The six control samples of the 131K document all gave the same continuation and the three v2.7.3 samples gave two other texts, so the rule read the controls as stable. Across this gate and three earlier test windows on the same code base, checkpoint and document, the v2.7.2 code alone gave five different 131K continuations in 16 samples (the DSA indexer's top-k returns its indices in a different order from call to call, which moves near-tied logits); six equal control samples happen by chance about 6 % of the time. One v2.7.3 text is one of those stock texts, leaving the commonest one at token 62 of 64 on a 0.13-nat near tie that the stock code also flips; the other is new, as about 3 in 16 stock samples are. Two of the three v2.7.3 first-token top-20 vectors are bit-identical to vectors the v2.7.2 code produced; the 32K first token was identical in every run on all three boots, and 1M stayed inside the controls' spread. Accepted on review as release-equivalent; the next gate judges the 131K continuation against the accumulated set of stock states and stores the full tokens.
## v2.7.2 release, 2026-11-10, speedbench (the README's current results until v2.7.3)

The README's v2.7.2 results and release gate, moved here (headings demoted, links adjusted, text unchanged) when v2.7.3 was released on 2026-11-10. The v2.7.3 numbers are in the [README](../README.md#current-results).

**v2.7.2** (2026-11-10, in production on the four Sparks since 2026-11-10 01:02) is v2.7.1 plus one lossless prefill change in the production profile: a prefill chunk whose prompt still goes on for at least one sliding window (128 rows) runs the 20 late layers on its last row instead of its last 128 rows, since nothing reads those rows later (`DSV41_PREFILL_CED_SKIP=1`, a port of Christopher Owen's spark-ds41f CED skip). In its release gate against two v2.7.1 control boots on the official checkpoint: cold prefill **+2.5 %** at 64K, **+3.5 %** at 131K and **+4.7 %** at 1M (against the mean of the controls), and a cold 388K prompt's first token in 59.7 s instead of 62.5-62.8 s (**+4.6 to +5.1 %**); decode unchanged (greedy c1 x0.999, c4 x0.997, c8 x1.016, c16 x0.997, sampled c1 x0.997 and c4 x0.985, speedbench load at c16 x1.005). Outputs: qeval 71 / 71 / 71, all 75 T = 0 qeval prompts inside the controls' own spread, KLD 0.006 short / 0.004 long (limit 0.035), draft acceptance on qeval identical on the three boots, reasoning length identical. Prompts that fit in one chunk are not affected. The image stack, the KV pool (8,541,952 tokens at fraction 0.85), the launch line and every v2.7.1 key are unchanged.

#### speedbench

[speedbench](https://github.com/knapcio/speedbench) 0.3 (`6733653`), full mode: 73 frozen prompts in six categories (prompt set speedbench-v1), 256 output tokens, thinking off, run on the head. v2.7.2 release-gate boot (2026-11-09, official 1017 checkpoint, KV pool pinned at 8,541,952 tokens as in the production profile), mean of two full runs per greedy cell and one run per sampled cell; c1 is the decode tok/s geometric mean over the prompts, c4 to c16 the aggregate tok/s:

| tok/s | all 73 prompts, c1 | prose, c1 | c4 aggregate | c8 aggregate | c16 aggregate |
|---|---:|---:|---:|---:|---:|
| greedy (temperature 0) | **89.9** | 66.6 | **135.3** | 172.4 | **215.6** |
| sampled (temperature 0.7, top-p 0.95, seed 1) | 87.4 | 64.6 | 131.9 | – | – |

The two v2.7.1 control boots of the same gate gave greedy c1 90.1 / 89.9, c4 135.7 / 135.8, c8 168.0 / 171.4, c16 216.6 / 216.0 and sampled c1 87.8 / 87.5, c4 133.4 / 134.6 tok/s. Speedbench load at c16 gave 228.6 aggregate tok/s, median TTFT 468 ms, per-stream median 16.7 tok/s (controls 226.3 / 228.5 tok/s, 468 / 466 ms). The CED skip only changes prefill chunks that are followed by more of the same prompt, so decode is at parity by design; the differences above are run-to-run spread.

Per category, greedy c1 (same boots, mean of two full runs):

| tok/s | prose | code | structured | agentic | multilingual | long context | all 73 prompts |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2.7.2 | **66.6** | 113.0 | 111.8 | 100.5 | 71.7 | 90.6 | **89.9** |
| v2.7.1 (two control boots) | 66.7 / 66.5 | 113.1 / 112.9 | 111.9 / 111.7 | 99.5 / 100.8 | 71.8 / 71.7 | 92.6 / 90.1 | 90.1 / 89.9 |

v2.7.2 runs: all 89.46 / 90.37, prose 66.09 / 67.19, code 112.40 / 113.56, structured 111.15 / 112.39, agentic 98.57 / 102.50, multilingual 71.03 / 72.35, long context 92.84 / 88.30. Median time to first token 166.2 / 165.8 ms (controls 165.9-167.5 ms); 73 of 73 prompts valid in both runs, none failed. Agentic and long context vary most from run to run.

**Under load** (v2.6, one fresh boot of the v2.6 image and profile, the last full load sweep; speedbench load: closed loop, 90 s per level, 256 output tokens, thinking off)

| concurrency | block tok/s | per stream, median tok/s | TTFT, median |
|---|---:|---:|---:|
| c1 | 77.3 | 83.3 | 207 ms |
| c4 | 131.7 | 38.7 | 310 ms |
| c16 | **211.0** | 16.0 | 539 ms |

Block tok/s is all completion tokens over the wall time of the level, including request start and prefill, so the c1 value sits below the decode geomean above. The v2.7.2 release gate ran load at c16 only: 228.6 aggregate tok/s, median TTFT 468 ms, per-stream median 16.7 tok/s.

**Prefill, cold** (v2.7.2 release gate, the gate's own probes, not speedbench: 64K, 131K and 1M are the median of three flushed prompts of 62K, 127K and 968K tokens; 388K is the first token of a cold 388,000-token prompt, median of three)

| tok/s | 64K | 131K | 388K | 1M (968K tokens) |
|---|---:|---:|---:|---:|
| v2.7.2 | **6,488** | **6,611** | **6,497** (59.7 s) | **5,485** |
| v2.7.1 (two control boots) | 6,264 / 6,395 | 6,379 / 6,393 | 6,180 / 6,212 (62.8 / 62.5 s) | 5,233 / 5,247 |

The needle tests at 248K and 969K tokens passed (38.3 s and 177.2 s). The four Sparks serve the abliterated variant in production ([`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) `517efab`); its edit touches only `attn.wo_b` and `shared_experts.w2` of the main layers (the DSpark draft layers are identical), so the two run within a few percent of each other. The v2.7.1 and v2.7 tables are in [docs/history.md](history.md).

#### v2.7.2 release gate

v2.7.1 / v2.7.2 / v2.7.1, three fresh boots in that order on the same four Sparks, official 1017 `1bae0e1` on all three, GPU clock capped at 2200 MHz, 2026-11-09/10 (run `run-20261109-191418-2004513`, 342 min including the restore of production). The controls are production v2.7.1's image and profile; the v2.7.2 boot ran a gate image of `1993630` (production v2.7.1's image with this release's changed files added) and the production profile, whose only difference is `DSV41_PREFILL_CED_SKIP=1`; KV pool 8,541,952 tokens on all three. Verdict: **PASS**, every rule (decode, sampled, load, cold prefill, memory, qeval, idle CPU, idle first token, inter-token gaps, long-prompt decode), 0 safety, 0 invalid. qeval 71 / 71 / 71 with the same four failed tasks on every boot, all 75 T = 0 prompts inside the controls' bracket, KLD 0.0057 short / 0.0037 long against the controls (limit 0.035), the sampled-token distribution test p 0.994, the 16 thinking-on reasoning prompts the same length on all three boots. Draft acceptance pooled over the speedbench prompts (the statistic since this gate) x1.001 / x0.997 / x1.009 / x0.999 of the lower control at c1 / c4 / c8 / c16, and on qeval identical on the three boots (3.979). Long-prompt decode (new in this gate: flushed 32K and 131K documents, each followed by a 512-token greedy continuation) 92.9 tok/s (controls 92.3 / 92.9) at 3.64 tokens per step (3.59 / 3.61). Head memory low-water 8.41 GiB (controls 8.48 / 8.54), during the cold 1M prompts 8.56 GiB (8.62 / 8.54); idle first token 194 ms median (190 / 193); load c16 p99 inter-token gap 282 ms (296 / 290); a short prompt beside a cold 131K prefill got its first token in 0.33 s (0.37 / 0.36); the issue-16 soak 313 / 313 requests ok and its companion load 226 requests, 0 failed; no Xid on any boot. Every node logged the CED skip armed and its skip receipts. A deliberately cold boot of v2.7.2 (tuning caches off) was ready in 171 s with no Xid, core dump or fault line.
## v2.7.1 release, 2026-11-09, speedbench (the README's current results until v2.7.2)

The README's v2.7.1 results and release gate, moved here (headings demoted, links adjusted, text unchanged) when v2.7.2 was released on 2026-11-10. The v2.7.2 numbers are in the [README](../README.md#current-results).

**v2.7.1** (2026-11-09, in production on the four Sparks since 2026-11-09 14:48) is v2.7 plus five lossless changes in the production profile: the two K = 5120 dense GEMMs of a prefill chunk run on 4096-row blocks that stay in L2 (`DSV41_PREFILL_GEMM_ROW_SPLIT=4096 DSV41_PREFILL_GEMM_ROW_SPLIT_N=1792,1152`, every shape checked bit for bit against the stock GEMM on every rank at its first call); the L2 window beside the attention core warms `wo_b` instead of `wo_a` (`DSV41_L2_PREFETCH_WOA_MB=0 DSV41_L2_PREFETCH_WOB_MB=6`); extends without host syncs and MoE launches without the PyTorch dispatcher (`DSV41_EXTEND_NOSYNC=1`, `DSV41_MOE_LAUNCH_PREPARED=plan,direct`); for sampled requests, the draft-KV commit reused from the verify graph and the draft probabilities computed while it runs (`DSV41_SAMPLED_COMMIT_REUSE=1 DSV41_SAMPLED_DRAFT_PROBS_EARLY=1`); and a wider prefill bypass (`DSV41_PREFILL_BYPASS=8192`, was 256: a 500-token prompt beside a cold 131K prefill answers in 1.3-1.7 s instead of 26-28 s). In its release gate against two v2.7 control boots on the official checkpoint: greedy decode c1 **+1.0 %**, c4 **+2.6 %**, c8 +3.2 % and c16 **+3.6 %** (aggregate, against the mean of the controls), sampled c1 +2.7 % and c4 +3.4 %, speedbench load at c16 +6.3 %, median first token at c1 166 ms instead of 207 ms, and cold prefill **+4.1 to +4.7 %** at 64K, 131K and 1M. Outputs: qeval 71 / 71 / 71, all 75 T = 0 qeval prompts inside the controls' own spread, KLD 0.007 short / 0.006 long (limit 0.035), draft acceptance on qeval identical on the three boots, reasoning length identical. The image stack, the KV pool (8,541,952 tokens at fraction 0.85) and the launch line are v2.7's.

#### speedbench

[speedbench](https://github.com/knapcio/speedbench) 0.3 (`6733653`), full mode: 73 frozen prompts in six categories (prompt set speedbench-v1), 256 output tokens, thinking off, run on the head. v2.7.1 release-gate boot (2026-11-09, official 1017 checkpoint, KV pool pinned at 8,541,952 tokens as in the production profile), mean of two full runs per greedy cell and one run per sampled cell; c1 is the decode tok/s geometric mean over the prompts, c4 to c16 the aggregate tok/s:

| tok/s | all 73 prompts, c1 | prose, c1 | c4 aggregate | c8 aggregate | c16 aggregate |
|---|---:|---:|---:|---:|---:|
| greedy (temperature 0) | **89.8** | 66.7 | **135.2** | 171.8 | **216.2** |
| sampled (temperature 0.7, top-p 0.95, seed 1) | 87.1 | 64.8 | 133.6 | – | – |

The two v2.7 control boots of the same gate gave greedy c1 89.0 / 88.8, c4 131.3 / 132.2, c8 166.2 / 166.8, c16 209.6 / 207.6 and sampled c1 84.4 / 85.3, c4 128.7 / 129.8 tok/s. Speedbench load at c16 gave 229.5 aggregate tok/s, median TTFT 466 ms, per-stream median 17.3 tok/s (controls 216.0 / 216.0 tok/s, 539 / 533 ms).

Per category, greedy c1 (same boots, mean of two full runs):

| tok/s | prose | code | structured | agentic | multilingual | long context | all 73 prompts |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2.7.1 | **66.7** | 113.0 | 111.8 | 98.0 | 71.8 | 92.4 | **89.8** |
| v2.7 (two control boots) | 65.8 / 65.9 | 111.7 / 111.9 | 110.6 / 110.8 | 100.4 / 97.9 | 70.8 / 70.9 | 89.4 / 90.0 | 89.0 / 88.8 |

v2.7.1 runs: all 89.52 / 90.07, prose 66.12 / 67.24, code 112.48 / 113.57, structured 111.05 / 112.50, agentic 98.40 / 97.60, multilingual 71.18 / 72.34, long context 93.25 / 91.46. Median time to first token 166.4 / 166.3 ms (controls 207.3-207.5 ms); 73 of 73 prompts valid in both runs, none failed. Agentic and long context vary most from run to run.

**Under load** (v2.6, one fresh boot of the v2.6 image and profile, the last full load sweep; speedbench load: closed loop, 90 s per level, 256 output tokens, thinking off)

| concurrency | block tok/s | per stream, median tok/s | TTFT, median |
|---|---:|---:|---:|
| c1 | 77.3 | 83.3 | 207 ms |
| c4 | 131.7 | 38.7 | 310 ms |
| c16 | **211.0** | 16.0 | 539 ms |

Block tok/s is all completion tokens over the wall time of the level, including request start and prefill, so the c1 value sits below the decode geomean above. The v2.7.1 release gate ran load at c16 only: 229.5 aggregate tok/s, median TTFT 466 ms, per-stream median 17.3 tok/s.

**Prefill, cold** (v2.7.1 release gate, the gate's own probes, not speedbench: median of three flushed prompts of 62K, 127K and 968K tokens)

| tok/s | 64K | 131K | 1M (968K tokens) |
|---|---:|---:|---:|
| v2.7.1 | **6,308** | **6,400** | **5,253** |
| v2.7 (two control boots) | 6,028 / 6,045 | 6,096 / 6,134 | 5,022 / 5,067 |

The needle tests at 248K and 968K tokens passed. The four Sparks serve the abliterated variant in production ([`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) `517efab`); its edit touches only `attn.wo_b` and `shared_experts.w2` of the main layers (the DSpark draft layers are identical), so the two run within a few percent of each other. The v2.7 tables (abliterated checkpoint) are in [docs/history.md](history.md).

#### v2.7.1 release gate

v2.7 / v2.7.1 / v2.7, three fresh boots in that order on the same four Sparks, official 1017 `1bae0e1` on all three, GPU clock capped at 2200 MHz, 2026-11-09 (run `run-20261109-085149-628062`, 341 min including the restore of production). The controls are production v2.7's image and profile with v2.7.1's `DSV41_PREFILL_BYPASS=8192`, which every boot ran; KV pool 8,541,952 tokens on all three. Every speed, prefill, memory, qeval, idle and latency rule passed, 0 safety, 0 invalid. qeval 71 / 71 / 71 with the same four failed tasks on every boot, all 75 T = 0 prompts inside the controls' bracket, KLD 0.0069 short / 0.0061 long against the controls (limit 0.035), the sampled-token distribution test p 0.994, the 16 thinking-on reasoning prompts the same length on all three boots. Head memory low-water 8.32 GiB (controls 8.34 / 8.36), idle first token 189.6 ms (187.0 / 187.2), load c16 p99 inter-token gap 283 ms (355 / 354). Every node logged the eight new keys armed, and the row split's first-call check passed for both widths on every rank. A deliberately cold boot of v2.7.1 (tuning caches off) was ready in 171 s with no Xid, core dump or fault line.

The verdict text was **FAIL on one rule, draft acceptance at c4**: the median over the 73 speedbench prompts of tokens per stream step was 3.091 for v2.7.1 against 3.168 / 3.250 for the controls, below the rule's 0.98 band. With 255 decoded tokens per prompt that median moves in steps of 255/n (one stream step is about 1.2 %), and the same rule fails a control at c16 (0.976). The pooled acceptance (all tokens over all steps) is equal: c4 2.899 against 2.898 / 2.931, c1, c8 and c16 within 0.3 %; on the 45 prompts whose output was identical in all six runs, tokens per step are identical (3.033), and qeval's T = 0 acceptance is identical on the three boots (3.979). Greedy text at c4 depends on which requests share a batch, which the faster admission shifts. Accepted on review as release-equivalent.

## v2.7 release, 2026-11-08, speedbench (the README's current results until v2.7.1)

The README's v2.7 results and release gate, moved here (headings demoted, links adjusted, text unchanged) when v2.7.1 was released on 2026-11-09. The v2.7.1 numbers are in the [README](../README.md#current-results).

**v2.7** (2026-11-08, in production on the four Sparks since 2026-11-08) is v2.6.5 plus FlashInfer 0.7.2 with DeepSeek-V4.1's own KV layouts (`DSV41_KV_LAYOUT=v41 SGLANG_SM120_V41_PRECISION=bf16`: the sliding-window cache in the 528-byte V4.1 layout as DeepSeek's reference stores it, the compressed caches in the 288-byte V4.1 FP4 layout, both read in place by FlashInfer's SM120 DSv4.1 kernels with exact on-chip dequantization), the SWA prompt-tail keep (`DSV41_SWA_PROMPT_TAIL_KEEP=1024`: a revisit of a document or chat with a new question reuses the shared prefix on its 2nd visit instead of its 3rd), NCCL 2.30.7 rebuilt with the NVIDIA/nccl #2393 AArch64 fence in the canary-roce image, and a launcher fix for `NCCL_OVERLAY_PIP=1` on a switched fabric. In its release gate against two v2.6.5 control boots: decode at parity (greedy c1 x1.004 / x1.005, c4 x0.997 / x0.999, c16 x0.999 / x0.995), draft acceptance x1.006-1.046, 2k / 6k revisits answered in 0.24 / 0.25 s instead of 0.53 / 1.14-1.16 s, a cold 1M prompt at x0.994 of the slower control; **cold prefill from 64K to ~400K context is ~1.7 % slower** (FlashInfer 0.7.2's BF16 sparse-MLA prefill kernel). It is a numerics change against v2.6.5 (the window keys' RoPE part is stored in FP8, as in the reference): T = 0 outputs differ on some prompts, KLD against v2.6.5 0.011 short / 0.006 long (limit 0.035), qeval unchanged (71 / 75). The KV pool is unchanged: 8,541,952 tokens at fraction 0.85, eight 1M-token contexts at once. The gate's verdict text was FAIL on one rule (reasoning length: v2.7 reasoned longer on the mean of 16 deterministic prompts, with identical answers), accepted on review; details in [CHANGELOG.md](../CHANGELOG.md).

#### speedbench

[speedbench](https://github.com/knapcio/speedbench) 0.3 (`6733653`), full mode: 73 frozen prompts in six categories (prompt set speedbench-v1), 256 output tokens, thinking off, run on the head. v2.7 release-gate boot (2026-11-08, KV pool pinned at 8,541,952 tokens as in the production profile), mean of two full runs per greedy cell and one run per sampled cell; c1 is the decode tok/s geometric mean over the prompts, c4 and c16 the aggregate tok/s:

| tok/s | all 73 prompts, c1 | prose, c1 | c4 aggregate | c16 aggregate |
|---|---:|---:|---:|---:|
| greedy (temperature 0) | **89.9** | 66.0 | **132.1** | **208.1** |
| sampled (temperature 0.7, top-p 0.95, seed 1) | 87.2 | 63.4 | 128.7 | – |

The two v2.6.5 control boots of the same gate gave greedy c1 89.6 / 89.5, c4 132.5 / 132.2, c16 208.4 / 209.1 and sampled c1 86.8 / 87.3 tok/s. Speedbench load at c16 gave 216.6 aggregate tok/s, median TTFT 542 ms (controls 210.9 / 214.9 tok/s, 545 / 537 ms).

Per category, greedy c1 (same boots, mean of two full runs):

| tok/s | prose | code | structured | agentic | multilingual | long context | all 73 prompts |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2.7 | **66.0** | 108.8 | 110.2 | 106.7 | 72.1 | 91.8 | **89.9** |
| v2.6.5 (two control boots) | 65.9 / 66.0 | 112.6 / 112.7 | 107.3 / 107.4 | 103.1 / 104.2 | 70.6 / 70.6 | 93.3 / 90.6 | 89.6 / 89.5 |

v2.7 runs: all 89.40 / 90.43, prose 65.46 / 66.47, code 108.21 / 109.44, structured 109.53 / 110.77, agentic 106.15 / 107.17, multilingual 71.49 / 72.63, long context 91.71 / 91.98. Median time to first token 204.4 / 209.1 ms; 73 of 73 prompts valid in both runs, none failed. The numerics change alters the greedy text and with it the draft acceptance per prompt, so categories move against v2.6.5 in both directions (code -3.4 %, structured / agentic / multilingual +2 to +3 %) while the all-prompt geomean and prose stay equal. Long context varies most from run to run.

**Under load** (v2.6, one fresh boot of the v2.6 image and profile, the last full load sweep; speedbench load: closed loop, 90 s per level, 256 output tokens, thinking off)

| concurrency | block tok/s | per stream, median tok/s | TTFT, median |
|---|---:|---:|---:|
| c1 | 77.3 | 83.3 | 207 ms |
| c4 | 131.7 | 38.7 | 310 ms |
| c16 | **211.0** | 16.0 | 539 ms |

Block tok/s is all completion tokens over the wall time of the level, including request start and prefill, so the c1 value sits below the decode geomean above. The v2.7 release gate ran load at c16 only: 216.6 aggregate tok/s, median TTFT 542 ms, per-stream median 16.6 tok/s.

**Prefill, cold** (v2.7 release gate, the gate's own probes, not speedbench: 64K and 1M are the median of three flushed prompts; 388K is the first token of a cold 388,000-token prompt, median of three)

| tok/s | 64K | 388K | 1M (968K tokens) |
|---|---:|---:|---:|
| v2.7 | 6,026 | 5,917 (65.6 s) | 5,151 |
| v2.6.5 (two control boots) | 6,145 / 6,165 | 6,023 / 6,062 (64.4 / 64.0 s) | 5,180 / 5,312 |

The steady-state screen before the gate put v2.7 at x0.983 of the slower v2.6.5 control at both 64K and 388K (every 40K-token stretch of the 388K prompt between x0.982 and x0.987) and located the difference in FlashInfer 0.7.2's BF16 sparse-MLA prefill kernel. In the kernel screens that kernel is faster than v2.6.5's attention from ~380K context on, so a cold 1M prompt is close to parity.

The official checkpoint ([`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) `1bae0e1`, the default weights profile in `.env.tp4.example`) was last measured in the v2.6.2 release gate (2026-10-29, same speedbench protocol, KV pool pinned at 4,194,304 tokens): full c1 all-prompt geomean 88.4, c4 130.7 and c16 207.6 aggregate tok/s, cold prefill 64K 6,054 and 1M 5,310 tok/s. v2.6.3 to v2.7 moved greedy all-prompt decode speed by about 1 % or less in their release gates on the abliterated checkpoint; the official checkpoint has not been re-measured since. The abliterated edit touches only `attn.wo_b` and `shared_experts.w2` of the main layers (the DSpark draft layers are identical), so the two run within a few percent of each other.

#### v2.7 release gate

v2.6.5 / v2.7 / v2.6.5, three fresh boots in that order on the same four Sparks, abliterated 1017 `517efab`, GPU clock capped at 2200 MHz, 2026-11-08 (run `run-20261108-010847-479202`, 334 min including the restore of production). The controls are production v2.6.5 as it serves; the v2.7 boot ran this release's production profile; KV pool 8,541,952 tokens on all three. Every speed, prefill, memory, qeval, idle and latency rule passed, 0 safety, 0 invalid. Speedbench full: the numbers above; draft acceptance (accepted tokens per verify step) x1.006 on qeval and x1.020-1.046 on speedbench c1 / c4 / c16 against the lower control. Cold prefill 64K x0.981 / x0.977 and 1M x0.994 / x0.970 of the two controls (the gate's bound for this release: x0.97). qeval 71 / 71 / 71 with the same four failed tasks on every boot; at T = 0 the v2.7 text differs from the controls' on 26 of the 74 prompts on which the controls agree, as expected from the numerics change. KLD against v2.6.5: short 0.0111, long 0.0064 (control vs control 0.0061 / 0.0026; limit 0.035); T = 1 distribution check PASS (12,000 samples); needles at 256K (41.4 s) and 1M (969,022 tokens, 187.9 s) found. Revisits with a new question: 2k cached 2,048 tokens (first token 0.244 s; controls 0 cached, 0.537 / 0.534 s), 6k 6,144 tokens (0.247 s; controls 0, 1.135 / 1.157 s). An evicted 302,048-token resume point replayed one 8,192-token chunk on every boot (first token 2.19 / 2.22 / 2.21 s), and the v2.7 output (16 greedy tokens) equals both controls'. Head MemAvailable low-water over the boot 6.47 / 8.73 / 6.95 GiB, NVRM out-of-memory lines 12 / 0 / 13; soak ([issue #16](issue-16-swa-crash.md) reproduction plus a c2 companion load) 321 / 317 / 320 requests with 0 failures. A deliberately cold boot of v2.7 with no tuning caches, before the gate, was ready in 242 s with no Xid or fault line on any node (the boot fault fixed in this release, see [CHANGELOG.md](../CHANGELOG.md)).

The verdict text was **FAIL on one rule, reasoning length**: the mean and median completion tokens and the mean reasoning characters of 16 thinking-on T = 0 reasoning prompts must stay within ±3 % of the controls (a guard against less thinking). v2.7 reasoned longer: mean 139.4 against 133.4 tokens (+4.5 %), median 124.5 against 103.5, reasoning characters 514 against 454; per prompt it was longer on 6, shorter on 9 and equal on 1, and one prompt (+337 tokens) carries the mean. The two controls are byte-identical, so the band had no control noise to compare against. All 16 answers are correct and identical on the three boots. Accepted on review as release-equivalent: the direction is more reasoning, not less.

## v2.6.5 release, 2026-11-07, speedbench (the README's current results until v2.7)

The README's v2.6.5 results, its note on the v2.6.6 candidate and the v2.6 correctness checks, moved here (links adjusted, text unchanged) when v2.7 was released on 2026-11-08 (v2.6.6 was not released on its own; its changes shipped in v2.7). The v2.7 numbers are in the [README](../README.md#current-results) and its release gate in [CHANGELOG.md](../CHANGELOG.md).

**v2.6.6 (candidate)** is v2.6.5 plus `DSV41_SWA_PROMPT_TAIL_KEEP=1024` in the production profile (a prompt end keeps 1024 tokens of live sliding-window cache, so a revisit of a document or chat with a new question reuses the shared prefix on its 2nd visit instead of its 3rd: 2k / 6k revisits 97.0 / 99.0 % cached where v2.6.5 cached 0, first token 0.24 instead of 0.53 / 1.16 s, in a screen on the four Sparks), NCCL 2.30.7 rebuilt with the NVIDIA/nccl #2393 AArch64 fence in the canary-roce image (speed and reduction results equal to stock in an A/B on the four Sparks) and a launcher fix for `NCCL_OVERLAY_PIP=1` on a switched fabric. The tables below are v2.6.5's until the v2.6.6 gate; details in [CHANGELOG.md](../CHANGELOG.md).

**v2.6.5** is v2.6.4 plus the V4.1 FP4 KV layout in the production profile (`DSV41_KV_LAYOUT=v41c DSV41_V41C_ARM=1 DSV41_V41C_READER=native`, "v41n": the compressed KV records take 912.25 instead of 1670.75 bytes per token and are read in place, outputs in the V4 exactness class), with the full KV pool pinned at `MAX_TOTAL_TOKENS=8541952` and `MEM_FRACTION_STATIC=HEAD_MEM_FRACTION_STATIC=0.85`: **8,541,952 tokens, +69 % against v2.6.2's ~5.06M** (x2.3-2.5 against v2.6.4's 3.44-3.70M), eight 1M-token contexts at once instead of three. The trade-off is the head's free host memory: its lowest MemAvailable was 5.72 GiB in the memory ladder on the four Sparks (eight 1M-token prompts resident at once, a cold 1M prefill and a long-prompt soak) against 8.9-9.3 GiB with a 5.13M pool at fraction 0.78, and 7.15 GiB in the release gate (v2.6.2 controls 8.77 / 10.22). The rollback is the v2.6.4 profile values (see [Rollback](../README.md#rollback)). It also adds `DSV41_TOKENIZER_HEAP=trim2` (the head's tokenizer process returns the GiBs of free heap that back-to-back long prompts leave behind; allocation only) and the fix for #66 (a loop-aborted request ends as a named error; broken tool calls in the history no longer get HTTP 400). Speed is unchanged within the gate's noise (greedy c1 +1.1 % / +1.3 % against the two v2.6.2 control boots).

Since v2.6.4 the profile has a decode attention that reads the sliding-window cache in place instead of copying it into smaller pages first (bit-identical outputs, greedy c1 decode about 1.5 % faster in two in-boot A/Bs) and a decode yield: while one prompt prefills in chunks, the other running streams keep about 40 % of their decode rate instead of 8 %, and that prompt's first token comes about 1.5x later. It also keeps idle agent sessions' prefix-cache resume points (superseded SWA checkpoints are evicted first and the SWA pool holds 384 prefix tails instead of 128: resumed 300-330K-token sessions answered in 3.9-6.5 s instead of 61-64 s; the larger SWA reserve is paid from the full KV pool), and a resume point that is evicted anyway costs a one-chunk replay (~2 s) instead of a whole-context re-prefill (details in [CHANGELOG.md](../CHANGELOG.md)). The default weights are the official checkpoint (below); the v2.6.5 release gate and the per-category tables were measured on its abliterated variant, which `.env.tp4.example` keeps as an optional profile. Setup: image `Dockerfile.canary-roce` with this release's adapters and RoCE runtime (SGLang `dsv4.1` `f80c91a4b` + the credited backports, without #45251), the "Production profile" block of [`.env.tp4.example`](../.env.tp4.example) with `EP_SIZE=1`, 1M context, checkpoint [`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) at `517efab`, GPU clock capped at 2200 MHz, fresh boots with the host memory compacted first and the first run after each boot discarded. Speculative decoding makes tok/s depend on the generated text, so other DeepSeek releases give somewhat different numbers on the same prompts.

#### speedbench

[speedbench](https://github.com/knapcio/speedbench) 0.3 (`6733653`), full mode: 73 frozen prompts in six categories (prompt set speedbench-v1), 256 output tokens, thinking off, run on the head. v2.6.5 release-gate boot (2026-11-06, KV pool pinned at 8,541,952 tokens as in the production profile), median of two full runs per greedy cell and one run per sampled cell; c1 is the decode tok/s geometric mean over the prompts, c4 the aggregate tok/s:

| tok/s | all 73 prompts, c1 | prose, c1 | c4 aggregate |
|---|---:|---:|---:|
| greedy (temperature 0) | **89.9** | 66.1 | **132.1** |
| sampled (temperature 0.7, top-p 0.95, seed 1) | 85.8 | 62.6 | 129.2 |

The sampled cells are single runs; the two v2.6.2 control boots of the same gate gave 84.85 and 85.00 tok/s at c1 (release gate below). The gate did not repeat the full c16 run; speedbench load at c16 gave 217.4 aggregate tok/s (controls 213.7 / 216.8).

Per category, greedy c1 (v2.6.5 release-gate boot, 2026-11-06, median of two full runs):

| tok/s | prose | code | structured | agentic | multilingual | long context | all 73 prompts |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1 | **66.1** | 112.9 | 107.7 | 103.1 | 70.8 | 94.6 | **89.9** |

Runs: all 89.38 / 90.44, prose 65.58 / 66.63, code 112.15 / 113.56, structured 107.09 / 108.30, agentic 102.30 / 103.95, multilingual 70.22 / 71.40, long context 94.85 / 94.26. Median time to first token 208.5 / 207.7 ms; 73 of 73 prompts valid in both runs, none failed. Long context varies most from run to run: the two v2.6.2 control boots of the same gate gave 92.85 / 89.15 and 88.83 / 93.09.

**Under load** (v2.6; speedbench load: closed loop, 90 s per level, 256 output tokens, thinking off; one fresh boot of the same image and profile)

| concurrency | block tok/s | per stream, median tok/s | TTFT, median |
|---|---:|---:|---:|
| c1 | 77.3 | 83.3 | 207 ms |
| c4 | 131.7 | 38.7 | 310 ms |
| c16 | **211.0** | 16.0 | 539 ms |

Block tok/s is all completion tokens over the wall time of the level, including request start and prefill, so the c1 value sits below the decode geomean above. The v2.6.5 release gate ran load at c16 only: 217.4 aggregate tok/s (212.1 block tok/s), median TTFT 538 ms (its v2.6.2 control boots: 213.7 / 216.8 aggregate tok/s, 538 / 532 ms).

**Prefill, cold, tok/s** (v2.6; 4k and 64k: speedbench longctx, unique salt per request, median of three documents per size, same boot as the load table; 388K: one cold varied-text prompt on the release-gate boot; v2.6.2 made cold 1M prefill about 5-8 % faster. The v2.6.5 release gate measured cold 64K at 6,198 tok/s, median of three flushed prompts, and a cold 388K-token prompt's first token at 64.1 s, median of three)

| 4k | 64k | 388K |
|---:|---:|---:|
| 5309 | 5970 | 5770 (65.2 s) |

The official checkpoint ([`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) `1bae0e1`, the default weights profile in `.env.tp4.example`) was last measured in the v2.6.2 release gate (2026-10-29, same speedbench protocol, KV pool pinned at 4,194,304 tokens): full c1 all-prompt geomean 88.4, c4 130.7 and c16 207.6 aggregate tok/s, cold prefill 64K 6,054 and 1M 5,310 tok/s. v2.6.3 to v2.6.5 move greedy decode speed by about 1 % or less in their release gates, so these carry over. The abliterated edit touches only `attn.wo_b` and `shared_experts.w2` of the main layers (the DSpark draft layers are identical), so the two run within a few percent of each other.

**Correctness checks** (v2.6 release-gate boot; the v2.6.5 gate's checks are in the next section). Needle retrieval PASS at 124k and 388k tokens (passphrase at 50 % depth; 19.8 s and 66.6 s); cold prefill of a 388K-token varied-text prompt 65.2 s (5,770 tok/s); `scripts/qeval.py` 71 / 75 (code 24/25, format 7/7, json 6/8, math 13/14, prose 5/5, reason 16/16), the same score and split as v2.5; T > 0 scan (`ds_gate.py tscan`, 35 outputs) 0 garbled, 0 errors; `/v1/messages` with the image placeholder token in text and with thinking enabled without a budget: both answered; the issue #16 accelerated reproducer (`scripts/issue16_repro.py`: 190K-900K cached prefixes with 512-4096 fresh tokens, alternating c1/c4) 410 requests in 15 minutes, 0 failures ([docs/issue-16-swa-crash.md](issue-16-swa-crash.md)). Exactness against v2.5: 75 qeval prompts at T = 0 with top-20 logprobs and cold prefills, on four fresh boots (v2.5, two-shot only, the full v2.6 set, v2.5 again): every v2.6 output byte-identical to v2.5. Sampling at T > 0 (v2.5 profile, 12,000 samples): the second token's counts match the verify row's own probabilities (chi-square p 0.61).

#### v2.6.5 release gate

v2.6.2 / v2.6.5 / v2.6.2, three fresh boots in that order, abliterated 1017 `517efab`, 2026-11-06 (run `run-20261106-214411-640735`, 277 min including the restore of production): PASS, 0 KILL, 0 INVALID. The controls are production v2.6.2 as it serves (tails 128, no KV pin, fraction 0.78); the v2.6.5 boot ran the production profile of this release (tails 384, pool pinned at 8,541,952, fraction 0.85). Speedbench full, median of two runs: greedy c1 all-prompt geomean 88.91 / 89.91 / 88.77 and prose 65.49 / 66.11 / 65.34 tok/s, c4 131.39 / 132.08 / 131.33 aggregate tok/s; sampled (T 0.7, top-p 0.95, one run per cell) c1 84.85 / 85.77 / 85.00 and prose 62.26 / 62.58 / 62.73 tok/s, c4 127.62 / 129.24 / 127.99 aggregate tok/s; speedbench load c16 213.65 / 217.44 / 216.81 aggregate tok/s. Cold 64K prefill 6182 / 6198 / 6210 tok/s; a cold 388K-token prompt's first token 63.5 / 64.1 / 63.3 s; a short request beside a 131,072-token prefill answered in 0.44 / 0.45 / 0.43 s. qeval 71 / 71 / 71; at T = 0 the v2.6.5 outputs equal the controls' on all 74 of the 75 prompts on which the two controls agree. KLD long 0.0049 and short 0.00459 (control vs control 0.00485 / 0.00531; PASS), T = 1 distribution check PASS (12,000 samples), 1M needle (968,707 tokens) found in 187 s. Every rank of the v2.6.5 boot logged the v41n pool (912.25 B per full token, 8,541,952 tokens) and native reader, SWA thin, the cone replay, the decode yield, the #66 named exit and the history repair as installed, and trim2 armed in the head's tokenizer process (10 trims, 12,101 MiB returned, longest 493 ms), with no fallback or error; the FlashInfer #7676 fence was present on every rank.

KV pool 4,316,160 / 8,541,952 / 4,552,192 tokens (the controls size theirs from the free memory at boot; 4,997,632 / 5,101,824 in the v2.6.4 gate). Head MemAvailable low-water over the boot 8.77 / 7.15 / 10.22 GiB, NVRM out-of-memory lines 0 / 0 / 0; at the soak's 15th minute the head's tokenizer process held 4.12 / 1.63 / 3.13 GiB.

Evicted resume point: a 302,048-token prompt of natural text, first the document, then 800 distinct 1K-token prompts that pushed its SWA window out, then the document plus 2,048 tokens. On the v2.6.5 boot every node replayed one 8,192-token chunk (0 reverts) and the first token came after 2.21 s, with 291,328 tokens cached; the v2.6.2 controls answered the same request with a full prefill in 53.3 / 50.8 s. The v2.6.5 output equals the second control's token for token (16 greedy tokens; the two controls differ from each other after 9). Soak (issue #16 reproduction with a seeded document order plus a c2 companion load): 268 / 307 / 261 requests with 0 failures, companion 39.6 / 49.3 / 38.9 tok/s.

#### Production-profile rows retired in v2.7

The README's "What the production profile runs" rows for the FlashInfer 0.6.18 attention modules, moved here unchanged when v2.7 replaced them with `DSV41_KV_LAYOUT=v41 SGLANG_SM120_V41_PRECISION=bf16` on FlashInfer 0.7.2:

| Layer | Setting | What it does |
|---|---|---|
| Prefill attention pages | `DSV41_PREFILL_NATIVE_EXTRA=1` (since v2.6.2) | SM120 sparse-MLA prefill reads the extra (compressed) cache in SGLang's own 128 / 256-token pages through a module generated from FlashInfer's sources at image build (approved source closure, sha256-pinned .so, no JIT at boot) instead of a persistent 64-token copy of the whole pool: 2.45-2.76 GiB less allocated after pool sizing per node, cold 1M prefill -5 %, bit-identical |
| Decode attention pages | `DSV41_SWA_NATIVE_DECODE=1` (since v2.6.4) | every decode attention call (up to 64 query rows: the DSpark verify and draft forwards) reads the sliding-window cache in SGLang's own 256-token pages; before, each call copied every touched page into four 64-token pages first (43 copy chains per c1 step), because FlashInfer's decode kernel was instantiated for 64-token pages only. Same kernel body at 256-token pages (its address math is generic in the page size), same tactic, same output bits; other shapes keep the copy. Module built at image build (`runtime/flashinfer_v41c`, sha256-pinned .so). Two in-boot A/Bs on the four Sparks: greedy c1 +1.36 / +1.86 %, c4 -0.06 / +0.85 %, sampled c1 -1.95 / +0.79 % (sampled run-to-run spread up to 3 %); T = 0 text identical on every stable prompt. Built on FlashInfer's SM120 sparse-MLA decode kernel; the 256-token instantiation is ours |
| KV layout | `DSV41_KV_LAYOUT=v41c`, `DSV41_V41C_READER=native`, `MAX_TOTAL_TOKENS=8541952` at fraction 0.85 (since v2.6.5; the v2.7 candidate replaces the v41n keys, `DSV41_SWA_NATIVE_DECODE` and `DSV41_PREFILL_NATIVE_EXTRA` with `DSV41_KV_LAYOUT=v41 SGLANG_SM120_V41_PRECISION=bf16` on FlashInfer 0.7.2, same pin) | the ratio-1/2 compressed KV records are stored in SGLang's own V4.1 FP4 layout (288 B instead of the 584-byte V4 slot: 912.25 instead of 1670.75 B per full token) and the sparse-MLA readers convert them to the V4 operand bytes in shared memory while the previous chunk computes, so the attention math sees v2.6's bytes. Capacity A/Bs on the four Sparks (tails 128): pool 4.89-5.05M → 9.05M tokens, 8 x 1M needles resident at once; with v2.6.4's tails 384 and the fraction 0.85 the pool is pinned at 8,541,952 tokens (3.44-3.70M without v41n), measured SAFE with eight 1M prompts resident at once (head low-water 5.72 GiB, 8,000,000: 5.96 GiB); c16 -0.4..-0.8 %, c1 / c4 / prose within ±0.1 %, cold 1M prefill +1.6 %; KLD inside the control-vs-control noise, qeval unchanged. Release gate: pool 8,541,952, speed within the controls' noise, a replayed 302K-token resume (2.21 s) equal token for token to a v2.6.2 control's full prefill (51-53 s). Built on FlashInfer's SM120 sparse-MLA kernels and SGLang's V41_FP4 writers; the in-place readers are ours (design idea: Christopher Owen's spark-ds41f) |

## Official 1017 checkpoint on the v2.5 image, 2026-10-26, speedbench

The README's note on the official checkpoint from v2.6 until v2.6.3, moved here unchanged (the v2.6.2 release gate on the official checkpoint is in [CHANGELOG.md](../CHANGELOG.md)):

On the v2.5 image on 2026-10-26, same speedbench protocol: c1 all 88.4 (prose 66.0, code 108.4, structured 111.0), c4 132.3, c16 200.0 block tok/s, cold prefill 4k 4,605 and 128k 5,783 tok/s. This release's [v2.6's] switches leave c1 decode unchanged (T = 0 outputs byte-identical), so its c1 carries over; its c16 and short-prefill gains were measured on the abliterated checkpoint only.

## Production 2026-10-23 (v2.5 + NVMe keepalive, main `1059dfe`), sparkDash 1.8.8 and speedbench

The README's results from 2026-10-23 until v2.6, moved here unchanged. sparkDash has not been used since 2026-10-25; later numbers are speedbench only.

Image `Dockerfile.canary-roce` built from `f9a1de4` (SGLang `dsv4.1` `f80c91a4b` + the credited backports, without #45251) plus this release's Engram NVMe keepalive (`adapter/row_store.cpp`), the "Production profile" block of [`.env.tp4.example`](.env.tp4.example) with `EP_SIZE=1`, 1M context, checkpoint [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at `1bae0e1`, GPU clock capped at 2200 MHz, measured 2026-10-23 on one fresh boot per checkpoint with the host memory compacted first. The abliterated variant of the same checkpoint is a second weights profile in `.env.tp4.example`; its numbers on the same image, profile and protocol follow each table. Speculative decoding makes tok/s depend on the generated text, so other DeepSeek releases give somewhat different numbers on the same prompts.

#### sparkDash 1.8.8

256 new tokens, temperature 0, thinking off.

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **91.3** | 115.0 (58.1) | 174.0 (44.8) | 242.8 (32.0) | 359.7 (24.2) |
| code | 142.9 | 186.1 (94.1) | 255.2 (66.0) | 341.4 (45.5) | 471.6 (31.9) |
| structured | 160.8 | 222.2 (114.3) | 215.7 (67.1) | 248.7 (41.7) | 393.5 (35.2) |
| json | 130.6 | 184.9 (96.3) | 322.8 (82.5) | 461.4 (59.6) | 726.9 (47.2) |

**Prefill, cold, tok/s** (measured on the same image without the NVMe keepalive, which leaves prefill unchanged: cold 388K 67.8 s with it, 67.4 s without)

| 4k | 16k | 32k | 64k | 128k |
|---:|---:|---:|---:|---:|
| 4798 | 5976 | 6196 | 6241 | 6291 |

Abliterated checkpoint ([`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) `517efab`), same image, profile and protocol, c1: prose 89.0, code 145.2, structured 160.1, json 131.4; c16 aggregate: prose 355.7, code 479.1; prefill 4k-128k 4855-6214 tok/s. Its edit touches only `attn.wo_b` and `shared_experts.w2` of the main layers (the DSpark draft layers are identical), so the two run within a few percent of each other.

**How these were measured.** sparkDash DecodeBench after two discarded warm-up prose c1 runs; c1 is the median of three runs, c4 the median of two, c2 / c8 / c16 one run each (c1 runs: prose 91.2 / 91.4 / 91.3; code 141.7 / 142.9 / 143.3; structured 155.1 / 160.8 / 160.9; json 130.1 / 130.6 / 131.4). sparkDash's prose c1 is a single prompt, and the non-prose types use a different prompt at each concurrency, so per-stream values are not comparable across columns. Prefill: sparkDash prefill bench, cold (salted prompts), after a 4k warm-up, median of the second and third of three runs per length; its filler is one repeated token, so the Engram row cache inflates these numbers ([MiaAI-Lab#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)).

#### speedbench

[speedbench](https://github.com/knapcio/speedbench) 0.3 (`6733653`), full mode: 73 frozen prompts in six categories (prompt set speedbench-v1), c1, temperature 0, 256 output tokens, thinking off, run on the head against `127.0.0.1:8888`. Decode tok/s, geometric mean over the prompts of each category:

| tok/s | prose | code | structured | agentic | multilingual | long context | all 73 prompts |
|---|---:|---:|---:|---:|---:|---:|---:|
| official 1017 | 64.7 | 106.8 | 109.5 | 98.3 | 70.0 | 91.0 | 87.5 |
| abliterated | 64.3 | 110.2 | 105.1 | 100.6 | 68.4 | 92.2 | 87.6 |

Median time to first token 205 ms (abliterated 208 ms); 73 of 73 prompts valid on both, none failed.

**Correctness checks on the same boot.** Needle retrieval PASS at 124k and 388k tokens (passphrase at 50 % depth; 21 s and 68 s); cold prefill of a 388K-token varied-text prompt 67.8 s (5,720 tok/s); `scripts/qeval.py` 71 / 75 (code 24/25, format 7/7, json 6/8, math 13/14, prose 5/5, reason 16/16); T > 0 scan (`ds_gate.py tscan`, 35 outputs) 0 garbled, 0 errors; `/v1/messages` with the image placeholder token in text and with thinking enabled without a budget: both answered; native `/generate` with top-10 logprobs on 16.5k-40k-token prompts answered. The abliterated boot passed the same checks (qeval 71/75 with the same split, cold 388K 69.0 s), plus `ds_gate.py kld` long and short panels against the stable 1017 image (continuation KL 0.0075 / 0.0126) and the issue #16 accelerated reproducer (`scripts/issue16_repro.py`: 190K-900K cached prefixes with 512-4096 fresh tokens, alternating c1/c4): 1,018 requests in 30 minutes, 0 failures, no CUDA error or Xid on any rank ([docs/issue-16-swa-crash.md](docs/issue-16-swa-crash.md)).

## Production 2026-10-22 (v2.5, image `f9a1de4`, before the NVMe keepalive), sparkDash 1.8.8

The v2.5 README table: official 1017 checkpoint `1bae0e1`, production profile, one fresh boot, same protocol as the 2026-10-23 sparkDash table above.

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | 91.0 | 114.8 (58.0) | 169.4 (43.7) | 232.9 (30.6) | 343.3 (23.0) |
| code | 140.3 | 182.0 (92.1) | 248.1 (64.4) | 321.6 (42.8) | 454.9 (30.7) |
| structured | 159.7 | 219.8 (113.3) | 214.4 (66.9) | 238.7 (39.6) | 373.3 (33.2) |
| json | 130.5 | 179.5 (93.3) | 318.1 (81.1) | 444.6 (57.6) | 694.3 (45.2) |

c1 runs: prose 91.1 / 91.0 / 89.8; code 137.5 / 140.4 / 143.4; structured 152.5 / 160.8 / 159.7; json 127.8 / 131.1 / 130.5. Abliterated `517efab`, same image: c1 prose 88.6, code 144.9, structured 158.4, json 128.1; c16 prose 337.6, code 456.9. speedbench 0.3 full (prose / code / structured / agentic / multilingual / long context / all): official 63.9 / 105.4 / 107.4 / 96.9 / 69.0 / 88.3 / 86.1, abliterated 63.3 / 108.7 / 103.2 / 97.3 / 67.7 / 89.9 / 85.9; median TTFT 218 / 219 ms. Cold 388K 67.4 s (abliterated 68.2 s); abliterated KL long/short 0.0062 / 0.0061; issue #16 reproducer 813 requests in 30 min (1,445 in 40 min on an earlier boot), 0 failures.

## Conservative fallback profile 2026-10-22 (1017 checkpoint), sparkDash 1.8.8

Served 2026-10-22 between the issue #16 quarantine and the re-qualification of the v2.4 levers: the "Conservative fallback profile" block of `.env.tp4.example` (v2.1 feature set, 4096-token chunks, memory fraction 0.80), `EP_SIZE=1`, checkpoint `mrfakename/DeepSeek-V4.1-Flash-1017-abliterated` at `517efab`, image of `57a9888` with the SGLang tree of `030813a` (no #45251), GPU clock cap 2200 MHz, one fresh boot, same protocol as the README.

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | 84.8 | 122.3 (61.9) | 166.6 (43.1) | 235.7 (31.2) | 336.7 (21.8) |
| code | 138.8 | 182.7 (92.3) | 242.9 (64.3) | 317.6 (42.0) | 457.8 (30.6) |
| structured | 153.8 | 217.7 (112.9) | 218.2 (72.6) | 275.3 (42.2) | 375.3 (32.1) |
| json | 126.0 | 176.4 (89.1) | 298.5 (76.1) | 476.9 (61.9) | 684.8 (44.7) |

Prefill, cold, tok/s: 4k 4805, 16k 5814, 32k 5938, 64k 5963, 128k 5964. speedbench 0.3 full: prose 61.2, code 105.3, structured 100.1, agentic 93.6, multilingual 65.9, long context 86.6, all 83.1. Cold 388K prefill 77.0 s (5,037 tok/s). qeval 71/75, tscan 0/35.

## Production 2026-10-12 (v2.4, image before the issue #16 fix), sparkDash 1.8.8

These measurements remain useful as historical performance data, but their image predates the issue #16 fix. Stack v2.4 (the historical `EXTRA_CONTAINER_ENV` line of [`.env.tp4.example`](../.env.tp4.example) with its settings block, `EP_SIZE=1`, `Dockerfile.canary-roce` image), built from a clean tree on all four nodes, sparkDash 1.8.8: 256 new tokens, temperature 0, thinking off, idle fleet, 2026-10-12. KV pool 3.67M tokens, 1M context. Raw output: [`docs/results/validation-20261012-v24.txt`](results/validation-20261012-v24.txt). Measurements from other fleets are linked, with their dates, in [`docs/community-results.md`](community-results.md).

#### sparkDash, GPU clock capped at 2200 MHz

All results here are at the 2200 MHz SM clock cap this fleet runs with (`nvidia-smi -lgc 0,2200`, re-applied by a timer), which keeps the nodes cooler.

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **89.9** | 123.3 (63.3) | 164.9 (41.7) | 237.2 (30.8) | 341.3 (22.2) |
| code | 131.0 | 175.8 (88.8) | 246.8 (63.2) | 306.8 (41.0) | 440.3 (29.4) |
| structured | 158.7 | 178.9 (103.7) | 240.4 (70.2) | 292.2 (43.5) | 575.6 (44.7) |
| json | 128.4 | 175.1 (90.3) | 314.0 (79.4) | 465.0 (59.4) | 632.8 (41.7) |

**Prefill, cold, tok/s**

| 4k | 16k | 32k | 64k | 128k |
|---:|---:|---:|---:|---:|
| 4756 | 5790 | 5941 | 5969 | 5896 |

**How these were measured.** sparkDash DecodeBench after two discarded warm-up prose c1 runs; c1 is the median of three runs, c4 the median of two, c2 / c8 / c16 one run each. Each cell is the better of two full sweeps on two boots of the same final image (c1 runs: prose 89.9 / 89.0 / 90.0; code 131.0 / 129.3 / 131.4; structured 152.8 / 159.3 / 158.7; json 123.4 / 129.5 / 128.5). This fleet ran without the optional `DSV41_ENGRAM_DRM_NODE` (its display reservation is used by another tenant), which costs a few percent at c8-c16. Prefill: sparkDash prefill bench, cold, one run per length; its filler is one repeated token, so the Engram row cache inflates these numbers ([MiaAI-Lab#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)).

#### Long context and multi-user behaviour

Varied-text prompts, cold (unique salt per prompt), [`scripts/v24_probe.py`](../scripts/v24_probe.py):

| | v2.4 |
|---|---:|
| cold prefill, 388K-token prompt | 77.8 s (4,976 tok/s) |
| 969K-token needle (passphrase found) | 235 s |
| short chat sent 8 s into a cold 400K prefill, time to answer | 4.6 s |
| return to a 388K session after a second 388K session ran | 1.5 s (100 % cached) |
| lowest MemAvailable during the 1M prefill, head / workers | 5.7 / 7.1-7.7 GiB |

What v2.4 changed against v2.3 is in [CHANGELOG.md](../CHANGELOG.md).

**Quality and correctness gates** (final image, against the v2.3 settings booted the same morning as the reference):

- `scripts/qeval.py` (75 auto-scored tasks, c1, temperature 0): 72 / 72 / 72 on v2.4, the reference 72 / 72 / 72; both miss the same three tasks.
- Greedy-continuation KL against the reference (`scripts/ds_gate.py kld`): mean 0.0045 over 239 positions (long prompts, 16.5k-40k) and 0.0061 over 559 positions (short, 1-3k), PASS against the 0.035 limit.
- T > 0 scan (`ds_gate.py tscan`, 35 outputs): 0 garbled, 0 errors (reference 0 of 35).
- Prefix cache (`scripts/prefix_scan.py`): PASS, cold-vs-warm logprob drift 0.0 on all eight hits (reference identical, including its one false anomaly).
- 60-minute soak at 16 concurrent clients (mixed prompts, T 0 / 0.6 / 1.0, 30 % thinking, ~5 % cold 22K-45K-token prompts): 2,285 completions, 0 errors, 0 empty, 0 container restarts or OOM, 0 kernel Xid / `NV_ERR`, MemAvailable never below 12.9 GiB.
- `/v1/messages` with the image placeholder token in message text and with `thinking: {"type": "enabled"}` without a budget: both answered.
- Required and named `tool_choice` now use SGLang's V4.1 structural tag. The stock default image admitted a bare DSML invoke that the parser returned as `message.content`; the build-time backport and parser/grammar test require a proper calls block and typed `tool_calls`. `auto` was unaffected, and the canary source already carried the fixed detector.

Every earlier measurement, the per-stage tables and the experiments that were tried and not adopted are in [docs/history.md](history.md); what changed when is in [CHANGELOG.md](../CHANGELOG.md).

## Production 2026-09-29 (v2.3), sparkDash 1.8.8

Production stack v2.3 (the last `EXTRA_CONTAINER_ENV` line of [`.env.tp4.example`](../.env.tp4.example) with `EP_SIZE=1`, `Dockerfile.canary-roce` image), built from a fresh clone of this repository on all four nodes, booted once with empty caches (healthy in 556 s; KV pool 6.80M tokens, 1M context), sparkDash 1.8.8: 256 new tokens, temperature 0, thinking off, idle fleet, 2026-09-29. Raw output: [`docs/results/validation-20260929-v23.txt`](results/validation-20260929-v23.txt). Measurements from other fleets are linked, with their dates, in [`docs/community-results.md`](community-results.md).

#### sparkDash, GPU clock capped at 2200 MHz

All results here are at the 2200 MHz SM clock cap this fleet runs with (`nvidia-smi -lgc 0,2200`, re-applied by a timer), which keeps the nodes cooler; DeepSeek-V4.1 decode gains almost nothing from higher clocks. SM clock under load median 2184 MHz (p90 2190), hottest GPU 68 °C, no throttling samples.

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **89.7** | 125.2 (64.2) | 165.9 (41.9) | 244.6 (31.7) | 357.2 (23.2) |
| code | 131.9 | 181.4 (91.7) | 248.0 (63.5) | 328.6 (44.0) | 456.3 (30.5) |
| structured | 157.8 | 184.6 (106.9) | 242.1 (70.6) | 300.6 (44.6) | 603.4 (47.1) |
| json | 130.4 | 179.9 (92.8) | 319.8 (80.9) | 477.3 (61.0) | 693.2 (44.8) |

**Prefill, cold, tok/s**

| 4k | 16k | 32k | 64k | 128k |
|---:|---:|---:|---:|---:|
| 4688 | 5583 | 5650 | 5664 | 5398 |

**How these were measured.** sparkDash DecodeBench after two discarded warm-up prose c1 runs; c1 is the median of three runs, c4 the median of two, c2 / c8 / c16 one run each (c1 runs: prose 89.7 / 89.0 / 90.2; code 131.5 / 131.9 / 132.1; structured 157.8 / 160.1 / 155.7; json 130.7 / 130.4 / 130.3). sparkDash's prose c1 is one prompt: greedy text is identical run to run on this stack, but a change in rounding anywhere (another stack, another clock) can follow a different greedy text there, so compare step time or many prompts across stacks. The non-prose types use a different prompt at each concurrency, so per-stream values are not comparable across columns. Prefill: sparkDash prefill bench, cold (salted prompts), after a 4k warm-up, median of the second and third of three runs per length. sparkDash's prefill filler is one repeated token, so every filler token hits the same Engram row and the row cache inflates these numbers ([MiaAI-Lab#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)); on real text v2.2 measured 3915 / 4743 / 4822 / 4946 / 4800 tok/s at ~4k / ~15k / ~30k / ~56k / ~123k tokens, and v2.3 does not change prefill.

**Quality and correctness gates** (same boot, against the v2.2 production tree booted the same afternoon as the reference):

- `scripts/qeval.py` (75 auto-scored tasks: code run against hidden asserts, JSON schema checks, numeric answers, format constraints; no LLM judge; c1, temperature 0, number extractor fixed in this release): 72 / 75 on v2.3; the v2.2 reference scored 72 / 72 / 72 in three runs. Both miss the same three tasks (`code_interval_intersect`, `json_escape`, `json_count`).
- Greedy-continuation KL against the reference (`scripts/ds_gate.py kld`, 256 greedy tokens after cold prompts of 16.5k-40k tokens and of 1-3k tokens, KL up to and including the first divergent token): mean 0.0031 over 178 positions (long) and 0.0071 over 190 positions (short), PASS against a 0.035 limit and at the A/A floor: the same candidate collected twice in one boot gives 0.0046 / 0.0035, because cold prefills of these prompts are not bit-reproducible run to run and the greedy texts part within a few tokens either way. Prompt-token logprobs are not available on this launch (see Known limits), so the panel scores the continuation.
- T > 0 scan (`ds_gate.py tscan`, 35 outputs at T 1.0 / 0.6 and mixed c4 batches, thinking on): 0 garbled outputs, 0 errors (reference: 0 of 35).
- Prefix cache (`scripts/prefix_scan.py`, ~100k-token shared prefix, c8 cold then warm at T 0.8, plus a greedy drift test with 4 full and 4 mid-document cache hits): PASS; cold-vs-warm logprob drift 0.0 over 128 greedy tokens on all eight hits, 0 degenerate outputs.
- The certified head is exact by construction and was checked on the device: its exact kernel matches the stock matmul bit for bit on every logit of all four rank shards, and check mode counted 0 mismatches over 3,355 live verify steps; armed on all four ranks in this boot. Prompt-logprob requests answered HTTP 400 and the server kept serving.

## Production 2026-09-25 (v2.2), sparkDash 1.8.8 (GPU clock cap then at 3003 MHz, effectively the driver default boost)

Production stack v2.2 (the last `EXTRA_CONTAINER_ENV` line of [`.env.tp4.example`](../.env.tp4.example) with `EP_SIZE=1`, `Dockerfile.canary-roce` image), built from a fresh clone of this repository on all four nodes and measured on that image with sparkDash 1.8.8: 256 new tokens, temperature 0, thinking off, idle fleet, 2026-09-25. Two boots, both healthy on the first try in 180 s and 160 s; KV pool 5.98-6.49M tokens (1M context). Raw output: [`docs/results/validation-20260925-v22.txt`](results/validation-20260925-v22.txt).

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **89.8** | 122.0 (62.6) | 164.1 (41.5) | 238.8 (31.0) | 340.9 (22.2) |
| code | 132.4 | 177.8 (89.8) | 252.7 (64.1) | 308.9 (41.4) | 436.4 (29.2) |
| structured | 156.9 | 181.6 (105.4) | 239.6 (70.2) | 295.7 (44.2) | 565.5 (44.3) |
| json | 124.3 | 177.6 (91.6) | 311.2 (78.7) | 468.8 (59.8) | 653.0 (42.2) |

The c1, c4 and c16 columns are from the v2.2 fresh-clone boot; c2 and c8 were measured on the same image the same evening, first run of each cell after two warm-ups ([`docs/results/validation-20260925-v22.txt`](results/validation-20260925-v22.txt)). Prose c1 is the median of five runs, 89.76 (88.97-90.21), after the warm-ups; the second boot gave 89.04 / 90.42 / 89.47. Code c1 is the median of three runs (124.95 / 132.62 / 132.37). Greedy output is byte-identical to v2.1's (hash check of 15 outputs on the first boot and 6 on the second). With the deterministic MoE reduction the greedy text is identical run to run, so the sparkDash numbers repeat within about ±1 tok/s; sparkDash's prose c1 is one prompt, and a stack that sums in a different order (another fabric, another all-reduce) follows a different greedy text there, so compare step time or a many-prompt benchmark across stacks. On 45 varied prompts (prose, structured and other catalogs, c1 greedy) the same image runs 61.1 / 99.0 / 74.7 tok/s; decode step 31.2-32.5 ms on prose and 37.4-37.8 ms on code at c1. sparkDash uses a different set of prompts at each concurrency for the non-prose types, so per-stream values are not comparable across columns. Sampled chat at the model card's T=1 / top_p=0.95 with thinking (c1, 18 requests x 800 tokens on two prompt sets) runs 67.2 / 65.9 tok/s (measured on the 2026-09-24 stack; sparkDash benches are greedy, where the draft temperature and block verification do not act).

**Prefill, cold, tok/s by prompt length** (one pass, same boot)

| 4k | 16k | 32k | 64k | 128k | 262k |
|---:|---:|---:|---:|---:|---:|
| 3944 | 5789 | 5855 | 5850 | 5734 | 5286 |

sparkDash's prefill filler is one repeated token, so every filler token hits the same Engram row and the row cache inflates these numbers (reported by koldfrontier in [MiaAI-Lab#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)); on real text (documentation and source code, a unique prefix per prompt so nothing comes from the prefix cache) the same image measured 3915 / 4743 / 4822 / 4946 / 4800 tok/s at ~4k / ~15k / ~30k / ~56k / ~123k tokens. Needle retrieval (a single phrase in varied filler at 37 % depth): PASS at 1,011,084 tokens on this image (324.3 s, head `MemAvailable` low-water 6,029 MiB); earlier stacks passed at 99k, 198k and 746k as well. A harder list-lookup needle ("value of item N" in a list of up to 197,000 items) passes for about half the keys at 129k and 259k on the stacks it was run on, a limit of the model rather than of either stack.

**Quality.** `scripts/qeval.py` runs 75 auto-scored tasks (code executed against hidden asserts, JSON schema-checked, numeric answers matched, format constraints enforced, prose checked for degeneration; no LLM judge), one request at a time, temperature 0. qeval was not re-run for v2.2: its greedy output is byte-identical to v2.1's, which scores 72 of 75 (earlier stacks 71-72); the three misses (`code_interval_intersect`, `json_escape`, `math_m9`) fail identically on the upstream example. Every speed change here is meant to be lossless: same weights, every draft token verified by the target, and each adapter either bit-identical to the stock path (checked at boot or in the in-image tests) or exact in distribution (draft temperature, block verification). RoCEnante sums in a different order than NCCL, so the numerics are not bit-identical to an NCCL run, which is why the profile is also scored. Run qeval from a worker, not from the head (it executes model-generated Python).

Every earlier measurement, the per-stage tables and the experiments that were tried and not adopted are in [docs/history.md](history.md); what changed when is in [CHANGELOG.md](../CHANGELOG.md).

## Production 2026-09-25 (v2, with the v2.1 c1 and prefill), sparkDash 1.8.8

Production stack (the last `EXTRA_CONTAINER_ENV` line of [`.env.tp4.example`](../.env.tp4.example) with `EP_SIZE=1`, `Dockerfile.canary-roce` image), built from a fresh clone of this repository on all four nodes and measured on that image with sparkDash 1.8.8: 256 new tokens, temperature 0, thinking off, idle fleet, 2026-09-25. Engine start to healthy 160-170 s; KV pool 6.0-6.6M tokens (1M context). Raw output: [`docs/results/validation-20260925-v2.txt`](results/validation-20260925-v2.txt).

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **87.7** | 120.4 (61.9) | 163.6 (41.4) | 237.6 (30.8) | 342.7 (22.2) |
| code | 124.8 | 175.0 (88.4) | 246.8 (63.2) | 309.8 (41.4) | 438.3 (29.3) |
| structured | 152.4 | 177.6 (103.3) | 240.4 (70.4) | 295.5 (44.2) | 572.2 (44.4) |
| json | 118.9 | 174.2 (89.8) | 301.7 (76.2) | 471.5 (60.2) | 659.9 (42.7) |

Prose and code c1 are from the latest fresh-clone boot (v2.1, prose median of five runs 86.7-87.7 after two discarded warm-ups; the v2 boot gave 86.5 and 122.6); the other columns are from the v2 sweep, whose decode path is identical. With the deterministic MoE reduction the greedy text is identical run to run, so the sparkDash numbers repeat within about ±1 tok/s; sparkDash's prose c1 is one prompt, and a stack that sums in a different order (another fabric, another all-reduce) follows a different greedy text there, so compare step time or a many-prompt benchmark across stacks. On 45 varied prompts (prose, structured and other catalogs, c1 greedy) the same image runs 58.4 / 94.3 / 71.8 tok/s; decode step 32.4-33.6 ms on prose and 38.9-39.4 ms on code at c1. sparkDash uses a different set of prompts at each concurrency for the non-prose types, so per-stream values are not comparable across columns. Sampled chat at the model card's T=1 / top_p=0.95 with thinking (c1, 18 requests x 800 tokens on two prompt sets) runs 67.2 / 65.9 tok/s (measured on the 2026-09-24 stack; sparkDash benches are greedy, where the draft temperature and block verification do not act).

**Prefill, cold, tok/s by prompt length** (two passes; fresh clone of the MXFP8 attention-gather commit, [`docs/results/validation-20260925-v21.txt`](results/validation-20260925-v21.txt), which also measured prose c1 87.7 (86.7-87.7), code c1 124.8, qeval 72/75 and the 1,011,084-token needle PASS in 325 s)

| 4k | 16k | 32k | 64k | 128k | 262k |
|---:|---:|---:|---:|---:|---:|
| 4119 / 4840 | 5891 / 5846 | 5893 / 5868 | 5936 / 5903 | 5793 / 5761 | 5355 / 5360 |

sparkDash's prefill filler is one repeated token, so every filler token hits the same Engram row and the row cache inflates these numbers (reported by koldfrontier in [MiaAI-Lab#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)); on real text (documentation and source code, a unique prefix per prompt so nothing comes from the prefix cache) the same image measured 4202-4685 / 4824-4922 / 4827-4830 / 4942-4970 / 4822-4871 tok/s at ~4k / ~15k / ~32k / ~62k / ~120k tokens. Needle retrieval (a single phrase in varied filler at 37 % depth): PASS at 99k, 198k, 746k and 1,011,084 tokens (the last one in 322 s, head `MemAvailable` low-water 7 GiB). A harder list-lookup needle ("value of item N" in a list of up to 197,000 items) passes for about half the keys at 129k and 259k on this stack and on the previous one alike, a limit of the model rather than of either stack.

## Production 2026-09-24 evening (EP1 release), sparkDash 1.8.8

Production stack (the last `EXTRA_CONTAINER_ENV` line of [`.env.tp4.example`](../.env.tp4.example) with `EP_SIZE=1`, `Dockerfile.canary-roce` image), built from a fresh clone of this repository on all four nodes and measured on that image with sparkDash 1.8.8: 256 new tokens, temperature 0, thinking off, idle fleet, 2026-09-24. Engine start to healthy 170 s; KV pool 6.56M tokens (1M context). Raw output: [`docs/results/validation-20260924-ep1.txt`](results/validation-20260924-ep1.txt).

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **84.6** | 118.0 (60.6) | 160.4 (40.5) | 231.6 (30.1) | 339.8 (22.1) |
| code | 120.9 | 172.7 (87.2) | 241.6 (61.9) | 303.3 (40.6) | 436.9 (29.2) |
| structured | 146.0 | 172.5 (100.6) | 232.8 (68.4) | 286.4 (42.9) | 565.5 (44.6) |
| json | 119.2 | 168.5 (86.9) | 299.4 (75.7) | 461.3 (58.9) | 658.7 (42.7) |

Prose c1 is the median of seven runs (84.2-85.2) after two discarded warm-ups; other boots of the same stack gave medians of 85.0-85.2. With the deterministic MoE reduction the greedy text is identical run to run, so the sparkDash numbers repeat within about ±1 tok/s. On 45 varied prompts (prose, structured and other catalogs, c1 greedy) the same image runs 57.4 / 92.9 / 70.8 tok/s. sparkDash uses a different set of prompts at each concurrency for the non-prose types, so per-stream values are not comparable across columns. Sampled chat at the model card's T=1 / top_p=0.95 with thinking (c1, 18 requests x 800 tokens on two prompt sets) runs 67.2 / 65.9 tok/s (the previous stack 62.3 / 59.8); sparkDash benches are greedy, where the draft temperature and block verification do not act.

**Prefill, cold, tok/s by prompt length** (two passes)

| 4k | 16k | 32k | 64k | 128k | 262k |
|---:|---:|---:|---:|---:|---:|
| 2449 / 3986 | 4784 / 4753 | 4870 / 4805 | 4815 / 4762 | 4721 / 4722 | 4264 / 4447 |

The 4k point of the first pass is the first request after the benches. sparkDash's prefill filler is one repeated token, so every filler token hits the same Engram row and the row cache inflates these numbers (reported by koldfrontier in [MiaAI-Lab#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)); on real text (documentation and source code, a unique prefix per prompt so nothing comes from the prefix cache) the same image measured 4003-4077 / 4154-4255 / 4144-4258 / 4138-4189 / 4092-4093 tok/s at ~4k / ~15k / ~29k / ~60k / ~113k tokens. Needle retrieval (a list lookup): passes at 129k tokens; at 259k some keys pass and some miss on both this stack and the previous FlashInfer/EP2 stack (key 17777 misses on both, 20001 and 3333 pass on both), a limit of the model at that length rather than of either stack. Engine start to ready is ~3 minutes with the fast loader.

## Production 2026-09-24 morning (EP2, FlashInfer MoE), sparkDash 1.8.8

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | 74.8 | 101.9 (53.5) | 143.9 (37.4) | 192.9 (25.6) | 318.5 (20.8) |
| code | 113.0 | 151.4 (75.7) | 212.6 (55.3) | 280.7 (37.3) | 391.8 (26.7) |
| structured | 137.1 | 156.0 (88.1) | 196.1 (57.1) | 201.7 (35.7) | 517.2 (44.2) |
| json | 105.6 | 164.8 (82.4) | 261.1 (65.6) | 398.2 (52.1) | 648.9 (41.8) |

Prefill 4k / 16k / 32k / 64k / 128k / 262k: 3119 / 4001 / 4681 / 4675 / 4575 / 4206 tok/s. Raw output:
[`results/sweep-20260924-types.txt`](results/sweep-20260924-types.txt), [`results/prodbench-20260924-current.txt`](results/prodbench-20260924-current.txt).

## Production stack as of 2026-09-24, with the full rationale per switch

One image, one env file. Everything in the tables below labelled **production** is this stack. What changed when is in [CHANGELOG.md](../CHANGELOG.md):

| Layer | Setting | Status | Why |
|---|---|---|---|
| Image | `Dockerfile.canary-roce` = upstream `dsv4.1` branch at `f80c91a4b` + RoCEnante overlay + all adapters | **on** | fastest decode of the three images (branch kernels + RDMA all-reduce) |
| Slots | `MAX_RUNNING_REQUESTS=16` | on | adds the c16 tier; c1–c8 unchanged |
| Experts | `EP_SIZE=2`, `--enable-deepseek-v4-fp4-indexer` | on | straggler wait halved; kernel path |
| Engram | `DSV41_CACHE_GIB=4`, `DSV41_CACHE_WAYS=16`, `DSV41_ENGRAM_PREFETCH=1` | on | row cache on NVMe; the row lookups run on a side stream right after the hasher instead of stalling the graph before each gather: step 51.7 → 49.5 ms, real-text prose c1 +5 %, rows bit-identical ([docs/upstream-watch.md](upstream-watch.md)) |
| Draft | `DSPARK_BLOCK_SIZE=5`, `SGLANG_DSPARK_FOLDED_SAMPLING=2` | on | k=5 wins on code, ties on prose; forced fold keeps sampled decode equal to greedy on the branch |
| Prefill | `CHUNKED_PREFILL_SIZE=4096` + `DSV41_INDEXER_CHUNKED=1` (v3) + `SPARK_PREFILL_TP_SPLIT=1` | on | bounded indexer transient (sglang#39187) plus the SG18 row split across ranks; 985k prompt leaves 5 GiB on the head |
| Shared expert | `DSV41_SHARED_PAD_K=1` | on | keeps the K=576 shape on the b12x kernel, −0.9 ms/step, bit-identical |
| wo_a | `DSV41_WO_A_W8=1` | on | verify/draft `wo_a` reads the checkpoint's fp8 bytes (exact twin of the bf16 copy) in the stock tiling: 43 layers 3.43 → 2.20 ms, step 52.9 → 51.7 ms at c1 |
| Draft temperature | `DSV41_DRAFT_TAU=0.7` | on | sampled requests only; exact by construction (same q for proposal and acceptance). At T=1 / top_p=0.95 with thinking (c1, 18 x 800 tokens per arm), 0.7 beat 0.8 on two disjoint prompt sets: 62.3 vs 61.2 and 59.8 vs 58.6 tok/s (+1.8 %, +2.0 %; accepted tokens per step +3.3 %, +2.2 %); 0.6 and 0.9 were below 0.7. (0.8 was the offline pick on 2026-09-23, +1.2 % over no scaling.) |
| Draft LM head | `DSV41_DRAFT_HEAD_FP8=1` | on | the draft reads an fp8 copy of the shared LM head (target logits untouched): 1405 → 721 us per step, acceptance unchanged |
| Verification | `DSV41_BLOCK_VERIFY=1` | on | block verification for sampled rows: exact output distribution, +1.8 % (prose) / +2.5 % (coding with thinking) accepted tokens per step |
| Folded results | `DSV41_FOLDED_FENCE=1` | on | correctness: folded (all-greedy) verify results cloned before the overlapped D2H copy, closes the sglang#40919 race |
| Verify length | `DSV41_VERIFY_CAP=conf:0.1` | on | per request and step, only the leading drafts whose running product of the draft confidence head's survival stays >= 0.1 are verified; the other verify rows are routed to the anchor row's experts, so they add no expert reads (each such row saves ~2 ms at c1), and acceptance is capped at the verified drafts through the engine's own cutoff. Exact: greedy outputs identical, sampled rows go through block verification with the dropped positions removed. Prose c1 +5 %, prose c4 +6 %, sampled thinking traffic +5 %, code and structured flat |
| Engram cache (optional) | `DSV41_ENGRAM_DRM_NODE=/dev/dri/card0` | off by default | one Engram layer's row cache in the GB10 display reservation (outside `MemAvailable`): ~1.8 GiB of runtime headroom per node, same speed and hit rate; needs a host change, see [docs/display-reserve.md](display-reserve.md) |
| Autotune cache | `DSV41_AUTOTUNE_KEEP=1` | on | keeps FlashInfer's MoE autotune cache across boots under EP (sglang#40320: the stock gate deleted it on every boot and re-drew the tactics, 26 re-tunes per start); kept only while the launch configuration matches |
| wo_a at c2+ / KV | `DSV41_WO_A_W8_MID=1`, `DSV41_WO_A_W8_DROP=1` | on | verify/draft `wo_a` at 9-192 rows also reads the fp8 twin (per-stream +5 % at c4, +3 % at c8 by verify step time); then the bf16 copy is released (722 MB per rank incl. the draft) and prefill dequantizes per call, bit-identical |
| Replicated linears | `DSV41_REPLICATED_SPLIT=wqkv_a,engram.wkv` | on | `engram.wkv` (6144 → 25600, 183 MB MXFP8) and `wqkv_a` (5120 → 1792) are `ReplicatedLinear`: every rank streamed the whole weight for the same output (the two Engram projections alone 2 × 756 us per step at c1). Each rank now runs the same quantized linear on its 128-row weight tiles and the columns are all-gathered; enabled per layer only when the slice is bit-identical to the stock layer on all ranks (checked at boot). Step probe prose 38.2 → 36.8 ms, code 44.9 → 43.6 ms; greedy outputs identical |
| Router remap | `DSV41_ROUTER_LIVE=1` | on | the verify-length remap (dead rows take the anchor row's experts) folded into the router kernel: dead rows load the anchor's scores, so the router computes the anchor's ids and weights for them and the separate remap kernel (40 launches, 0.21 ms/step) is gone. Built from the engine's own router source with three text substitutions; bit-identical outputs (checked with and without per-token bias); greedy text identical. Varied prompts +0.3 %, sparkDash unchanged within noise |
| Draft main_proj | `DSV41_DRAFT_MAIN_PROJ_SPLIT=1` | on | the draft's replicated `main_proj` (15360 → 5120, 105 MB) as a 1/4 column shard in fp8 with the checkpoint's exact weight bytes + all-gather: 430 → ~150 us per step; draft only, the target is untouched |
| Transport | `SGLANG_ROCE_ALLREDUCE=1`, `SGLANG_ROCE_MAX_SIZE=2097152`, `B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0`, `DSV41_ROCE_GATHER=2097152` | on | TP SUM all-reduces up to 2 MiB over RDMA on both rails; 2 MiB covers the 16-slot step (983 KB). `DSV41_ROCE_GATHER` also routes TP all-gathers up to 2 MiB per rank (the draft's vocab-parallel logits, 6 per step, and the splits above) through the one-shot kernel: 600 → 400 us per step at c1, a byte copy |
| NCCL | `IB_HCA=rocep1s0f0,roceP2p1s0f0` | on | neutral within noise, kept for the remaining collectives |
| Fabric | switched RoCE, tree reachable | on | every default assumes a switch; a switchless ring sets `NCCL_SWITCHLESS_RING_ONLY=1` instead ([optional-setups.md](optional-setups.md#switchless-ring-no-roce-switch)) |
| Serving | `--enable-cache-report`, `--sleep-on-idle`, `--min-free-slots-delay 1`, `DSV41_MAX_NEW_TOKENS`, loop abort, thinking alias | on | cached-token usage; sleep-on-idle takes the head scheduler from 47 % to 14 % CPU when idle with no change to first-response latency (0.2 s) or decode; engine `/metrics` is an opt-in because its overhead has not been benchmarked on this profile |
| Weight loading | `DSV41_FAST_LOAD=1` (+ `--model-loader-extra-config {"num_threads":1}`) | **on** | engine start 343 s → 111–124 s, bytes identical, decode/prefill/needle unchanged; costs 3–13 % of the KV pool (6.71–7.27 M vs 7.47–7.82 M tokens on the same image), the one trade-off in this table ([docs/fast-load.md](fast-load.md)) |
| Rust image processor | `SGLANG_RUST_BUILD_MODE=never` | off | the branch's `cargo` probe can hang the head before the HTTP server starts; PIL path is used |
| Adaptive chunk sizer | `DSV41_ADAPTIVE_CHUNK` | off | superseded by the bounded indexer; it would only shrink chunks needlessly |
| DSpark SPS table / ragged verify | `DSPARK_SPS_TABLE` | ignored unless `DSPARK_SPS_ALLOW_ENGRAM=1` | compact ragged verify crashes the Engram path on this model; the guard keeps the verify-all schedule even if a stale/profiled table exists |
| NVFP4 checkpoint (`nvidia/DeepSeek-V4.1-Flash-NVFP4`) | – | not used | corrected 2026-11-02 ([#64](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/issues/64); this row said "routed experts only"): the routed experts are the official MXFP4 values re-encoded losslessly with twice the scale bytes (+16 GiB in the file, about +4 GiB per rank on the GPU and about 6 % more expert bytes read per decode step); the smaller file comes from the two Engram tables re-quantized from FP8 to FP4 (lossy; -76 GiB for the later `nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`), which saves nothing here because the engine serves Engram rows from NVMe; DSpark unvalidated |

Rollback to any earlier point is an env change: `DSV41_FAST_LOAD=0` restores the stock loader, `SGLANG_ROCE_ALLREDUCE=0` drops the RDMA transport, `SPARK_PREFILL_TP_SPLIT=0` the row split, `IMAGE=dsv41-4x-spark:canary` the RoCEnante overlay, `IMAGE=dsv41-4x-spark:local` the branch.

## Decode, prefill and boot time by stage

sparkDash decode bench, 256 new tokens, temperature 0, thinking off, idle fleet, no foreign traffic (checked against the engine's `#running-req` log). Four DGX Spark, TP4/EP2, driver 580.x, `lmsysorg/sglang:dev-dsv41` base. Run-to-run spread between boots of the same configuration is about ±2 % on c1, so differences inside that band are noise. Both images were rebuilt from a fresh clone of this repository on 2026-09-17 and re-measured (base: prose c1 52.7 / c8 170, code c1 100.5; canary: prose c1 55.2 / c8 177, code c1 100.8). Full record with every intermediate step: [`docs/window-20260916.md`](window-20260916.md), raw outputs in [`results/window-20260916/`](results/window-20260916/).

### Prose decode, aggregate tok/s (per stream in brackets)

| Profile | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| upstream TP4 example (from its README) | 45.4 | 72.9 | 103.1 (26.7) | 114.1 (23.2) | 134.2 (22.0) |
| this profile, `Dockerfile` (base image) | 51.6 | 76.7 | 109.3 (28.5) | 160.9 (20.8) | 248.8 (16.7) |
| this profile, `Dockerfile.canary` (upstream dsv4.1 branch) | 55.4 | 80.9 | 118.9 (30.7) | 178.2 (24.0) | 277.5 (18.6) |
| production on 2026-09-18 (`Dockerfile.canary-roce`, 2 MiB route, prefill TP split, both rails, fast load, Engram prefetch) | 61.0 | 85.9 | 125.2 (33.2) | 178.9 (24.3) | 292.9 (19.3) |
| production at noon 2026-09-23 (the 2026-09-18 stack + `wo_a` fp8 twin, fp8 draft LM head, draft temperature, block verification, folded fence; 2026-09-23) | 66.0 | 86.2 | 125.4 (32.9) | 190.6 (25.5) | 307.8 (20.4) |
| production 2026-09-23 evening (the noon stack + adaptive verify length `DSV41_VERIFY_CAP=conf:0.1` + kept autotune cache) | 69.7 | 91.4 | 133.5 (34.7) | 195.1 (25.7) | 310.3 (20.5) |
| production 2026-09-23 night (+ `wo_a` fp8 twin at 9-192 rows and the bf16 copy released) | 69.1 | 97.5 | 138.3 (36.2) | 189.6 (25.2) | 314.1 (20.5) |
| the same + RoCE all-gathers + draft `main_proj` split, uncapped GPU clock (2026-09-24; the rows above ran under a local 2200 MHz cap, which [costs ~1.5 % on prose](upstream-watch.md)) | 72.0 | 101.0 | 141.1 (36.7) | 192.9 (25.5) | 318.1 (20.8) |
| **production** (+ `engram.wkv` and `wqkv_a` column-split, bit-identical; 2026-09-24) | **73.9** | **102.4** | **142.3 (37.1)** | **191.5 (25.3)** | **319.2 (21.2)** |

### Code and structured decode, aggregate tok/s

| Profile | code c1 | code c8 | code c16 | structured c1 |
|---|---:|---:|---:|---:|
| this profile, base image | 96.7 | 446.7 | 595.3 | 104.5 |
| this profile, canary image | 100.4 | 513.3 | 838.6 | 108.0 |
| production on 2026-09-18 | 113.3 | 548.3 | 882.4 | 124.1 |
| production at noon 2026-09-23 | 118.3 | 543.8 | 879.7 | 128.8 |
| production 2026-09-23 evening | 114.5 | 540.8 | 878.9 | 126.4 |
| production 2026-09-23 night | 113.8 | 556.3 | 885.7 | 125.5 |
| RoCE all-gathers + draft `main_proj` split, uncapped clock (2026-09-24) | 120.8 | 565.9 | 900.5 | 129.5 |
| **production** (+ replicated-linear split, 2026-09-24) | **119.0** | **573.5** | **914.3** | **131.5** |

### Prefill, cold, tok/s by prompt length

| Profile | 4k | 16k | 32k | 64k | 128k | 262k |
|---|---:|---:|---:|---:|---:|---:|
| upstream example (chunk 1024) | 3350 | 3782 | 3768 | 3531 | 3251 | – |
| this profile, base image (chunk 4096 + indexer backport) | 3532 | 4006 | 4038 | 3917 | 3230 | 2724 |
| this profile, canary image | 3174 | 3982 | 4180 | 4010 | 3499 | 2701 |
| production on 2026-09-18 (canary-roce + prefill TP split + fast load + Engram prefetch) | 3202 | 3497 | 4665 | 4674 | 4539 | 4241 |
| production at noon 2026-09-23 | 3619 | 4516 | 4501 | 4500 | 4465 | 4169 |
| production 2026-09-23 evening (4k/16k repeated after the first cold pass) | 4086 | 4510 | 4566 | 4530 | 4446 | 4125 |
| production 2026-09-23 night (4k/16k/32k/128k from a repeated pass) | 3983 | 4592 | 4575 | 4539 | 4405 | 4082 |
| **production** (2026-09-24, single cold pass; the 4k point is the first request of the pass) | **3159** | **3638** | **4783** | **4804** | **4677** | **4337** |

**Caveat on the prefill table:** sparkDash's prefill filler is one repeated token, so every filler token hits the same Engram row and the row cache (`DSV41_CACHE_GIB=4`) inflates those numbers (reported by koldfrontier in [MiaAI-Lab#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)). The same canary engine on random-word text, cold, one request per size, `prompt_tokens / TTFT`:

| random text | 11.8k | 23.8k | 47.3k | 94.3k | 188.7k |
|---|---:|---:|---:|---:|---:|
| canary, tok/s | 3330 | 3666 | 3347 | 3187 | 2657 |
| production, tok/s | 2879 | 3612 | 3776 | 4006 | 3092 |

Use these rows for real prompts; the sparkDash column overstates by 9–20 % at 16k–128k. The v3 row's 12k value is a single cold request right after boot (the split does not engage below 32k).

Long-context checks: needle retrieval PASS at 131k, 262k and **985k** tokens on the canary image (985k cold prefill 732 s, head `MemAvailable` low-water 6.6 GiB) and on production (131k 25.6 s, 262k 58 s, **985k 585 s**, low-water 5.0 GiB). The split without the chunked scoring (SG18 as published, base image) reached 503 s at 985k but left only 1.9 GiB on the head, which is why v3 keeps the 2 GiB logits budget inside each rank's partition.

### Boot time

| | stock loader | fast load (`DSV41_FAST_LOAD=1`) |
|---|---:|---:|
| target `load_weight` (rank 0 / 1 / 2 / 3) | 225–246 / 95 / 246 / 114 s | 71–74 / 83 / 73 / 71 s |
| draft `load_weight` | 38–50 s | 3–8 s |
| engine start to ready (`scheduler_e2e`) | 343–354 s | 124–129 s |
| `max_total_num_tokens` (KV pool, same image, same night) | 7.47–7.82 M | 7.27 M and 6.71 M on two boots (pinned buffers); 6.2–6.8 M with the earlier mmap buffers |

Same image, gate on versus off, 2026-09-18. Decode, prefill and the needle test are unchanged. The KV pool is 3–13 % smaller (it also varies more from boot to boot): SGLang sizes it from the head's `MemAvailable` right after the loads, and ~0.8 GB less is available then with the fast loader (with pageable buffers it was 1.5 GB, traced to driver staging memory; the remainder shows only as mapped file pages of the scheduler process). Flip `DSV41_FAST_LOAD=0` if the last 0.5–1 M tokens of pool matter more than 220 s per boot. Profile, dead ends and raw snapshots: [docs/fast-load.md](fast-load.md).

## Adaptive verify length (2026-09-23 evening)

Same boots, sparkDash c1 medians of three; step probe = greedy 400-token requests with the engine's own `spec_verify_ct`:

| `DSV41_VERIFY_CAP` | prose c1 | code c1 | structured c1 | step probe prose (ms/step, tok/s) | step probe code (ms/step, tok/s) |
|---|---:|---:|---:|---:|---:|
| unset | 66.1 / 65.5 | 118.2 / 115.8 | 126.4 / 124.9 | 47.4, 48.3 | 48.5, 82.0 |
| `5` (machinery, all rows live) | 65.1 | 116.6 | 124.7 | 47.7, 48.1 | 48.8, 81.4 |
| `3` (fixed) | 60.5 | 86.0 | 72.0 | 44.9, 48.4 | 43.7, 71.3 |
| `2` (fixed) | 56.5 | 69.2 | 70.4 | 40.7, 49.9 | 41.2, 63.9 |
| `conf:0.05` | 67.5 | 115.7 | 123.9 | 42.4, 51.9 | 48.0, 82.8 |
| `conf:0.1` (two boots) | **69.3 / 68.8** | 115.9 / 113.6 | 124.3 | 41.0, **52.4** | 47.4, 82.6 / 80.9 |
| `conf:0.2` | 65.7 | 115.9 | 124.5 | 40.1, 49.1 | 48.0, 78.4 |
| `conf:0.35` | 61.8 | 116.4 | 124.3 | 38.6, 48.9 | 44.5, 80.0 |

Sampled thinking traffic (T=1, top_p 0.95, 6 technical prompts x 2, 800 tokens, c1), boots in the order conf / unset / conf / unset: 53.8, 53.2, 56.7, 52.0 tok/s, i.e. 55.3 vs 52.6 (+5 %); accepted tokens per step 2.62 vs 2.72, step 47.4 vs 51.9 ms. qeval 71 and 72 of 75 (primary 53 of 55; `math_m9` hits the 640-token cap on most images, `prose_p2` as noted below). A fixed cap loses: the head is what makes the cut pay. Rebuilt from a fresh clone of this repository and booted with the `.env.tp4.example` production line: prose c1 70.2 (median of 62.8 / 70.2 / 70.2), code c1 117.9, structured 125.7; qeval 73 of 75.

`DSV41_REPLICATED_SPLIT`, `DSV41_DRAFT_MAIN_PROJ_SPLIT`, `DSV41_ROCE_GATHER` (2026-09-24, uncapped clock, greedy step probe = 3 prompts x 400 tokens, median of the 2nd/3rd run after boot): the six per-step draft vocab all-gathers over RoCE instead of NCCL 600 -> 400 us, prose step 38.9 -> 38.4 ms; the draft `main_proj` shard prose 38.4 -> 38.2 ms, code 45.3 -> 44.9 ms; `engram.wkv` split prose 38.2 -> 37.1 ms, code 44.9 -> 43.9 ms (sparkDash prose c1 71.8 -> 74.1, code c1 118 -> 121); `wqkv_a` split prose 37.1 -> 36.8 ms (sparkDash prose c1 74.1 -> 74.7). Every split layer was bit-identical on all four ranks at boot (40 `wqkv_a` + 2 `engram.wkv`, plus the draft's), and greedy outputs match the unsplit engine byte for byte. Splitting the indexer's `wq_b` and the compressor's `wkv_gate` the same way gave nothing (prose 36.8 -> 36.9 ms) and is not in the production line. Cost: the column slices are extra copies next to the full weights (which prefill still uses), ~210 MB per rank allocated after the KV pool is sized. Fresh clone of this state, built on all four nodes and booted with the `.env.tp4.example` production line: every split layer ON and bit-identical, greedy outputs identical to the pre-split engine, sparkDash prose c1 74.7 (74.72 / 74.68 / 74.05), code c1 120.1 / 124.4; qeval 71 of 75 (`code_interval_intersect`, `json_escape`, `math_m9` fail on every image here, `json_count` has failed before).

`DSV41_WO_A_W8_MID`, verify step time with 400-token greedy requests on different prompts (two boots per arm, on / off): c2 59.0 / 58.4 ms, c4 78.3 / 82.1 ms, c8 118.1 / 120.0 ms; greedy c1 text identical. sparkDash's fixed prompts are not a usable A/B for kernels that change rounding order: any change to the prefill numerics flips a near-tie somewhere in the 256 tokens and moves c1/c2 by +-10 %, which is why the mid path leaves prefill on the stock kernel. Fresh clone of this state (all switches of the production line): prose c1 69.2 (69.6 / 66.8 / 69.2), code c1 116.9, structured 123.1, prose c4 137.8; qeval 72 of 75. `DSV41_WO_A_W8_DROP`: greedy outputs byte-identical to the bf16 path (after the first request of a boot, which drifts with or without it), prefill and decode unchanged; KV pool 6.10-6.43 M tokens over four boots against 5.93-6.34 M without.

Note on measuring: a dashboard polling `nvidia-smi` every 2 s on every node cost 0.6 ms per decode step here (47.1 vs 46.5 ms/step with it paused); the sparkDash instance used for these tables polls every 10 s since (`POLL_INTERVAL_GPU=10000`, `POLL_INTERVAL_BANDWIDTH=10000`).

## Measured 2026-09-23 (sparkDash, same method as the tables above)

After `DSV41_WO_A_W8`, `DSV41_DRAFT_TAU`, `DSV41_DRAFT_HEAD_FP8`, `DSV41_BLOCK_VERIFY`, `DSV41_FOLDED_FENCE`: **prose c1 65.3** (65.26 / 65.37 / 65.28, was 61.0), code c1 113.8 (113.3), structured c1 124.7 (124.1). sparkDash benches run greedy, where the draft temperature and block verification do not act; at the model card's T=1 / top_p=0.95 (6 technical prompts, thinking on, 12 × 800 tokens, c1) the same stack went 50.9 → 54.2 tok/s end to end. qeval over three runs (two on the production checkout, one from a fresh clone of this repository): 72, 71 and 73 of 75; primary (code + reasoning + math) 54, 53 and 54 of 55 (53 before these adapters); `json_escape` fails as it also does on earlier production images; `prose_p2` came out at 88 words against a 100-word minimum on the first two runs and passed on the third. Fresh-clone boot: sparkDash prose c1 65.42 / 65.47 / 65.34, code c1 107.1 / 115.9 / 117.0 / 111.0 (median 113.5; the code tier is the noisiest), structured c1 124.7, prose c4 32.3 per stream (122.9 aggregate).

## Quality gate, 2026-09-17

| Run (2026-09-17, same day, same fleet) | pass | broke | fixed | p |
|---|---:|---:|---:|---:|
| base image, same env (`dsv41-4x-spark:local`, chunk 4096, indexer backport) | 71/75 | – | – | – |
| **production** (branch + RoCEnante + prefill TP split) | **72/75** | 0 | 1 (`json_count`) | 1.000 |
| reference: upstream example, 2026-09-11 | 71/75 | | | |

The three tasks that fail on every stack (`code_interval_intersect`, `json_escape`, `math_m9`) fail identically on the upstream example. Raw results: [`results/quality-20260917/`](results/quality-20260917/). Run it from a worker, not from the head (it executes model-generated Python).

## Measurement notes for the dated rows

- `scripts/window-20260916.sh` is the runbook that produced the tables (preflight, build, two boots, benches, rollback).
- The 2026-09-23 production rows come from one boot right after a power cycle (`prodbench` as below; prose c1 is the median of 65.24 / 65.97 / 66.23). The draft temperature and block verification act only on sampled requests, so they are not visible in these greedy benches; the `wo_a` and draft-head kernels take 2.0 ms (~4 %) off the step, and the rest of the prose c1 difference is within the boot-to-boot spread (FlashInfer re-draws its MoE tactics on every boot under EP, sglang#40320).
- The 2026-09-18 production rows were re-measured on 2026-09-18 on the Engram-prefetch boot (`results/fastload-20260918/prodbench-prefetch-20260918.txt`): two warm-up prose c1 runs discarded, then one run per cell; prose c1 is the median of three runs (61.0, 61.2, 61.0); the 4k and 16k prefill cells are single cold points that swing 2.4–3.8k between boots.

## Tested and not adopted

### 2026-09-25

Measured on the fleet while building v2.2.

- **L2 prefetch v3/v4 sub-gates**: `AHEAD`, `ENGRAM`, `DRAFT` and `LMHEAD` flat; `SKIP_N` +0.17 ms/step; `WOB_MB=4` +0.12 ms/step; `MOE` slows the MoE stream it overlaps (~12.6 us per 10 MB) and its lines do not survive to the next layer. Only `WOA` shipped.
- **Fused MXFP8 quantization into q_norm / wo_a / hc**: flat.
- **Layer-14 Engram lookup on a side stream during verify**: ~0.
- **Fused hc prefill kernel**: slower than the current kernels.
- **MoE restructures**: closed; the MoE phases already run at 223 GB/s live, near the DRAM limit.
- **CPU pinning**: flat.

### 2026-09-23

All measured against the live engine or with an offline draft forward whose per-position acceptance matches the engine within 0.016.

- **Fine-tuning the DSpark draft on our own traffic** (~10 M captured tokens, markov + norms + hyper-connection mixers + attention, fp8 quantization-aware, exported in the checkpoint format and swapped in at load): +3.4 % accepted tokens on held-out requests, but 0 % on new prompts and on new coding problems. Agent sessions resend near-identical context, so a request-level split leaks; split by session or date.
- **Multi-pass drafting** (a second draft forward that sees the tokens drafted so far, trained for it): +2.4 % (2 passes) to +3.9 % (5) accepted tokens; each extra pass costs ~9 % of a step.
- **Expert-sharing routing** (a verify row's last two experts swapped to one another row already streams when the router scored it within 0.05): 20.5 → 18.4 experts per layer at bs 1, but the step went 52.9 → 56.6 ms from the extra ops, and it changes routing. Reverted.
- **n-gram lookup over the request's own output** on coding-with-thinking traffic: even an oracle choosing per step is +1 % (3.17 → 3.20 tokens/step); DSpark already covers the repeats.
- **Relaxed acceptance** (not exact): accepting any draft token inside the target's top-p is only +20 % accepted tokens; on free text the draft's deeper proposals are mostly wrong, not merely differently distributed.
- **Kernels**: split-K MXFP8 for the small-M dense projections was 1.5–3x slower than b12x (which already reads at ~200 GB/s); the hyper-connection mix kernel stays at 23–28 µs whatever the slicing or weight layout; the draft's LM head in fp8 would save ~0.5 ms with no acceptance loss (3.0882 → 3.0875), not done yet.
- Where a coding answer with thinking spends its steps: 99.4 % inside the thinking at 3.15 tokens/step, 0.6 % in the code at 5.6 tokens/step.

### 2026-09-18

- **`SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=0`** (Markov W2 replicated instead of TP-sharded, drops the 0.5 ms vocab all-gather): the bf16 vocab GEMM grows 3.04 → 4.31 ms/step, net step 49.7 → 50.3 ms, prose c1 unchanged. Kept sharded.
- **Fast-load knobs `DSV41_FAST_LOAD_INFLIGHT_GB=12` + `num_threads=2`**: load_weight 79 → 69 s but start-to-ready only 115 → 113 s. Defaults kept.
- **Parallel Engram misses in `row_store.cpp`** (probe first, pool for ≥2 misses): gather gap 2.44 → 2.38 ms, i.e. nothing; the side-stream prefetch above is what removed it.

### 2026-09-17

- **sglang#39704** (mHC/metadata overhead for medium batches) applied onto the pinned branch: every column within ±2 % of production on this fleet (its gain is at 32–64 concurrent requests on GB300). Kept out.
- **Newer `dsv4.1` heads (from 2026-09-16 22:15, #39671)** drop the torch candidate indexer and gate the DeepGEMM one on SM100; on SM121 DeepGEMM then rejects the 256-token KV pages (`block_kv == 64`). The pin stays at `f80c91a4b` until upstream has an SM12x candidate path again.
- `CHUNKED_PREFILL_SIZE=8192`, split threshold 16k, NVFP4 experts, `DSV41_CACHE_GIB` above 4, k≠5, NCCL channel/algorithm tuning: measured, no gain or worse.
