# Results from other fleets

Measurements other people published for this profile (or Mia's upstream TP4 line it was
merged into). Linked, not copied: each was taken on its own hardware, fabric, checkout,
clock policy and harness, at the date shown, and may not match the current release. The
numbers in the [README](../README.md) are the only ones measured here.

| Date | Who | Fabric | Line | What it covers | Link |
|---|---|---|---|---|---|
| 2026-09-25 | ecohash-co | switched | PR #36 line (v2.1) | decode c1-c8 vs GLM-5.3-Flash EXL3 on the same boxes, own prompts | [Mia #38](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/38) |
| 2026-09-27 | ChrisLou-bioinfo | switchless ring | v2 line | ring bring-up, RoCEnante on the ring, prefill with chunk 4096 | [Mia #40](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/40) |
| 2026-09-27/28 | ZackO2o | switchless ring | canary-roce line | decode sweep with `benchmarks/decode_window.py`, KV pool vs `DSV41_CACHE_GIB`, C1 conditions (answer length, context, contention) | [Mia #41](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/41), [their notes](https://github.com/ZackO2o/DeepSeek-V4.1-Flash-4x-GB10-1M-Full-Recipe/blob/main/results/2026-09-28-sglang-switchless-ring-tp4.md), [cross-project table](https://github.com/ZackO2o/DeepSeek-V4.1-Flash-4x-GB10-1M-Full-Recipe/blob/main/results/2026-09-28-cross-project-4x-gb10-reference.md) |
| 2026-09-29 | LECYWZA | switchless ring | canary-roce line | 1M multi-session prefix-cache eviction and `--swa-full-tokens-ratio` | [Mia #43](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/43) |
| 2026-10-01/06 | hushengkai | switched | v2.1 and v2.3 | small-request TTFT behind long prefills, `--prefill-decode-interval`, chunk 8192, SWA ratio | [Mia #44](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/44), [#9](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/issues/9) |
| 2026-10-03 | hushengkai | switched | v2.1 vs v2.3 | independent v2.3 evaluation: decode, prefill, 1M needle, clock cap A/B | [#10](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/issues/10) |
| 2026-10-03 | xueyinmei-dot | switchless ring | v2.3 ("R5" profile) | full sparkDash matrix on a ring with hardware-forwarded diagonal paths, chunk 8192 | [#11](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/11) |
