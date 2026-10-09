# R5 wired switchless-ring deployment: independent four-Spark profile

This opt-in deployment record was contributed by xueyinmei-dot on 2026-10-07. **R5 is a local profile name, not an upstream release.** It freezes an independently deployed four-GB10 configuration and reports a recorded R5 measurement. It does not change this repository's production defaults or claim a new kernel or transport implementation.

## Provenance and pinned assets

The serving design and Engram row store originate with MiaAI-Lab; this repository's tuned stack is knapcio's work. RoCEnante and fused MoE come from b12x/local-inference-lab, the SG17 overlay and prefill split from rhys101, and switchless hardware forwarding and cumulative NCCL patches from FujitsuPolycom/sparkring. See the main [credits](../README.md#credits) and [ring guide](switchless-ring.md).

The local R5 profile kept the preceding wired R4 capacity settings and incorporated `5f11a97172bb650e9f296ad6ecb95a61e8d3170d` (certified head and replay guard). The subsequent `58f232155917d388b6053eca079c617fd306c33c` changes documentation only. These are upstream commits, not new optimizations authored by this contributor.

| Asset | Reference |
|---|---|
| Repository source | `58f232155917d388b6053eca079c617fd306c33c` |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash`, revision `fb2764a5cf321eaa5070ca8f9e892818f477c16d` |
| Model config SHA256 | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| Checkpoint | 48 safetensors shards, 510310549975 bytes (~475.26 GiB) |
| Image recipe | `Dockerfile.canary-roce`, Linux arm64 / SM121 |
| SGLang canary | `dsv4.1` at `f80c91a4b`; resolve and retain its full SHA |
| b12x_next | `a7d7d29b2ef8869086e0ceaa787321f17544e3c9`, renamed and patched by the repository's build script |
| NCCL source | 2.30.7, NVIDIA commit `73cf112295c33aee2b895f329f592f2a9b4b0f97` plus sparkring's cumulative dual-PCI-domain patch |
| Mesh planner/marker reference | sparkring `f16b5f4`; resolve and retain full SHA |
| Reference NCCL library | CUDA 13.0 build, 61419672 bytes, SHA256 `933c0cecfd68b25511ab49fd8cf1cb33c75f471d9d0f6db35ae302d56ebbe925` |

The original image was local: image ID `sha256:5a081f4ff5f4890fe1f79dea85338e9b6a0a71ce5e6df0c3181a5c78021d5fe2`, with no registry digest. **That ID is not a pullable public image.** The mutable `lmsysorg/sglang:dev-dsv41` base tag alone cannot reconstruct it. A new builder must pin an arm64 base digest, dependencies and staged canary tree and record a new image receipt. Reference runtime packages included sglang-kernel 0.4.7, sgl-deep-gemm 0.2.0 and FlashInfer 0.6.18. A rebuilt image is a separately validated variant, not proven byte-identical.

This contribution does not distribute images, weights, NCCL binaries or caches. Fetch sources and validate licenses via their respective upstreams. The NCCL patch is cumulative: apply it to its intended clean base, **do not layer it over switchless-cycle/skip-tree patches or add experimental bidirectional patches**. Compile in the target arm64 toolchain, run `tests/routing_handle/compat.cc`, retain the resulting hash, and use the same library on all ranks.

For offline deployments, stage the image, model, pinned sources, wheels and build dependencies on a download workstation first. Inspect the full Dockerfile/launcher/JIT chain for implicit downloads; `HF_HUB_OFFLINE=1` does not make pip or a Docker build offline. Verify each asset after transfer. Existing verified assets can be reused.

## Physical wiring and local storage

Four DGX Sparks, four compatible 200GbE DAC/AOC cables, no switch or diagonal cable:

| Cable | Endpoints |
|---|---|
| 1 | rank0 f0 ↔ rank1 f1 |
| 2 | rank1 f0 ↔ rank2 f1 |
| 3 | rank2 f0 ↔ rank3 f1 |
| 4 | rank3 f0 ↔ rank0 f1 |

Keep a separate wired management network. Determine rank order from actual wiring. On this reference each physical port exposes functions under two PCI domains: **four logical HCA functions do not mean four independent physical 200GbE links**. Confirm OEM port labels and mappings with `ibdev2netdev` and sysfs rather than assuming left/right order.

| HCA index | Netdev | RDMA HCA |
|---:|---|---|
| 0 | `enp1s0f0np0` | `rocep1s0f0` |
| 1 | `enp1s0f1np1` | `rocep1s0f1` |
| 2 | `enP2p1s0f0np0` | `roceP2p1s0f0` |
| 3 | `enP2p1s0f1np1` | `roceP2p1s0f1` |

Names are case-sensitive. Allocate one /24 per cable per PCI-domain plane; the following addresses are synthetic examples, not contributor addresses:

| Rank | f0 primary | f1 primary | f0 secondary | f1 secondary |
|---:|---|---|---|---|
| 0 | 10.241.1.1/24 | 10.241.7.2/24 | 10.241.2.1/24 | 10.241.8.2/24 |
| 1 | 10.241.3.1/24 | 10.241.1.2/24 | 10.241.4.1/24 | 10.241.2.2/24 |
| 2 | 10.241.5.1/24 | 10.241.3.2/24 | 10.241.6.1/24 | 10.241.4.2/24 |
| 3 | 10.241.7.1/24 | 10.241.5.2/24 | 10.241.8.1/24 | 10.241.6.2/24 |

Reference Ethernet MTU was 9000, active RoCE MTU 4096, IPv4-mapped RoCEv2 GID index 3 on all four HCAs. Verify the selected GID's address, type and netdev; a slot's existence is insufficient. If interface ordering or GID slots differ, regenerate all relevant settings and record the change.

Use a local complete checkpoint and local per-rank packed Engram on every node (`NFS_SHARE=0` when using the repository launcher). Pack with `--tp 4 --rank N`, not the packer's TP3 default, and do not copy rank0's packed shard to every rank. Budget ~476 GiB for weights per node plus packed Engram, image and JIT storage; inspect pack sizes and disk space before starting. Keep model and packed files read-only, state and autotune paths writable.

## Transport activation and validation

Follow [the upstream ring/mesh setup](switchless-ring.md#rocenante-on-the-ring-hardware-forwarded-opposite-node-paths) while the engine is stopped. This profile uses:

- Patched NCCL Ring-only connectivity, four advertised IPv4 listener GIDs and PCI-domain-preserving fallback; channels fixed at four.
- A read-only overlay **over the image's pip NCCL library**, not a second NCCL on LD_PRELOAD/LD_LIBRARY_PATH. Check `/proc/<server-pid>/maps` and the loaded file's hash, not only Docker mounts.
- RoCEnante for small collectives with a 262144-byte all-reduce/gather cap, hardware-forwarded opposite-node paths, hairpin queues 8192 and two-wave scheduling disabled.
- NIC profile `hairpin_num_queues=4`, `flow_steering_mode=hmfs`, eswitch legacy and `hw-tc-offload on`; every forwarding rule must actually report `in_hw`.

The marker's flow label 16383 / EtherType 0x88b5 and the neighbor's hardware MAC/EtherType rewrite implement the diagonal path. Ordinary Linux IP routing is not a substitute for the RDMA path. Reserve the marker port on the fabric. Queue driverinit changes reinitialize NICs and invalidate QPs: do not reinitialize interfaces or stop mesh/markers with the engine running.

Run `scripts/ring_mesh/plan.py` against the recipient's actual hosts, MACs, links and rank order. Install its scripts and boot units, then regenerate/check the plan once queue size 8192 is active: a plan made at queue size 1024 can retain the 81920-byte cap. The peer-HCA map below is a reference for exactly the stated rank/HCA order, **not a universal map for any four machines**.

Validate before serving: four HCAs ACTIVE with correct GIDs; `ListenerRouting ... advertised=4`; final QP routing retains the selected PCI domain; both domains' byte counters rise; markers active, tc rules `in_hw`, and the diagonal path's hardware counters rise without corresponding software forwarding. Seeing HCA 2/3 in a channel log alone does not prove traffic uses them.

## Capacity and inference settings

| Parameter | R5 value |
|---|---|
| TP / EP / nodes | 4 / 1 / 4 |
| Context | 1048576 |
| Total token pool | 6500000 |
| Memory fraction | 0.80 |
| Prefill chunk | **8192**, distinct from upstream's 4096 production profile |
| Request slots / decode graph max BS | 16 / 16 |
| DSpark | block 5, draft tau 0.7, confidence verify cap `conf:0.1` |
| Engram | NVMe, cache 4 GiB / 16 ways, 96 IO threads |
| Fast load | on, TP slice auto, inflight 6 GiB |
| Prefill SP / FP8 | on / on |
| Certified head | on, M=6,12,18,24,30,36,42,48 |
| Native boot flow | WARMUP=1; head SKIP_SMOKE=0; workers SKIP_SMOKE=1; SMOKE_QUICK=1 |

All other explicit settings are below. The KV capacity is recorded as a separate deployment setting. Context is a configured limit, not proof of long-context quality at every concurrency. Reasoning effort 75 is not a 75-token thinking budget.

`DSPARK_SPS_TABLE` / `DSPARK_STS_TABLE` paths existed in the reference environment, but the actual launch argv did **not** load these tables. Do not generate/load new tables and claim it is the same reference. Likewise the reference did not add graph-tier alignment or `--skip-server-warmup`.

### Complete non-secret common environment

This is **Docker env-file syntax, not a shell source file** (some values contain spaces). Copy to a site-local file and adapt network/path identity settings. API_KEY is deliberately absent; generate a recipient-specific key in a separate restricted file, distribute securely and never commit it.

```ini
# Docker --env-file format; do NOT source this file in Bash.
# Identity/IP values are examples; API_KEY must be supplied privately.
B12X_COMPILE_CACHE_DIR=/state/b12x-compile
B12X_ROCE_CACHE_DIR=/state/b12x-roce
B12X_ROCE_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1
B12X_ROCE_PEER_HCA_MAPS=1=0/2,2=0/3,3=1/3;0=1/3,2=0/2,3=0/3;0=1/2,1=1/3,3=0/2;0=0/2,1=1/2,2=1/3
B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES=0
CHUNKED_PREFILL_SIZE=8192
CONTEXT_LENGTH=1048576
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_GRAPH_MAX_BS_DECODE=16
CUDA_HOME=/usr/local/cuda
DGX_BASELINE_ID=DSF41-R5-PUBLIC-REFERENCE
DIST_INIT_ADDR=10.241.1.1:20000
DSPARK_BLOCK_SIZE=5
DSPARK_SPS_TABLE=/state/dspark_sps.json
DSPARK_STS_TABLE=/state/dspark_sts.json
DSV41_AUTOTUNE_KEEP=1
DSV41_BLOCK_VERIFY=1
DSV41_CACHE_GIB=4
DSV41_CACHE_WAYS=16
DSV41_CERT_HEAD=1
DSV41_CERT_HEAD_M=6,12,18,24,30,36,42,48
DSV41_DRAFT_HEAD_FP8=1
DSV41_DRAFT_MAIN_PROJ_SPLIT=1
DSV41_DRAFT_TAU=0.7
DSV41_EAGER_GLUE=all
DSV41_ENGRAM_PREFETCH=1
DSV41_FAST_LOAD=1
DSV41_FAST_LOAD_EP_SIZE=1
DSV41_FAST_LOAD_INFLIGHT_GB=6
DSV41_FAST_LOAD_N_EXPERTS=384
DSV41_FAST_LOAD_TP_SLICE=auto
DSV41_FOLDED_FENCE=1
DSV41_HC_FUSED=1
DSV41_INDEXER_CHUNKED=1
DSV41_IO_THREADS=96
DSV41_L2_PREFETCH=1
DSV41_L2_PREFETCH_WOA=1
DSV41_LOOP_ABORT=1
DSV41_LOOP_LINE_REPEATS=8
DSV41_LOOP_NGRAM=32
DSV41_LOOP_REPEATS=4
DSV41_MAX_NEW_TOKENS=32768
DSV41_MOE_B12X_NEXT=1
DSV41_MOE_B12X_NEXT_DETERMINISTIC=1
DSV41_MXFP8_BACKEND=b12x
DSV41_PACKED_DIR=/engram
DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192
DSV41_PREFILL_SP=1
DSV41_PREFILL_SP_FP8=1
DSV41_REPLICATED_SPLIT=wqkv_a,engram.wkv
DSV41_RESIDENT_SCALES=0
DSV41_ROCE_GATHER=262144
DSV41_ROCE_RING=1
DSV41_ROUTER_LIVE=1
DSV41_SHARED_PAD_BUF_ROWS=256
DSV41_SHARED_PAD_K=1
DSV41_SOURCE=/models/DeepSeek-V4.1-Flash
DSV41_SPEC_SYNC_FREE=all
DSV41_SPLIT_COMPACT_GATHER=1
DSV41_STATS_SECONDS=60
DSV41_TP_PAD=0
DSV41_VERIFY_CAP=conf:0.1
DSV41_WO_A_W8=1
DSV41_WO_A_W8_DROP=1
DSV41_WO_A_W8_MID=1
EP_SIZE=1
EXTRA_SGLANG_ARGS=--fp8-gemm-backend flashinfer_cutlass --watchdog-timeout 1800 --enable-metrics --min-free-slots-delay 1 --enable-deepseek-v4-fp4-indexer --enable-cache-report --sleep-on-idle
HF_HUB_OFFLINE=1
HOST=0.0.0.0
LANG=en_US.UTF-8
LANGUAGE=en_US:en
LC_ALL=en_US.UTF-8
LD_LIBRARY_PATH=
LIBRARY_PATH=/usr/local/cuda/lib64/stubs
MAX_RUNNING_REQUESTS=16
MAX_TOTAL_TOKENS=6500000
MEM_FRACTION_STATIC=0.80
MODEL_PATH=/models/DeepSeek-V4.1-Flash
NCCL_ALGO=Ring
NCCL_BUFFSIZE=1048576
NCCL_CROSS_NIC=1
NCCL_CUMEM_ENABLE=0
NCCL_CUMEM_HOST_ENABLE=0
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,ENV,NET
NCCL_IB_DISABLE=0
NCCL_IB_EXTENDED_IPV4_GIDS=1
NCCL_IB_GID_INDEX=3
NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1
NCCL_IB_MERGE_NICS=0
NCCL_IB_PRESERVE_PCI_DOMAIN=1
NCCL_IB_QPS_PER_CONNECTION=1
NCCL_IB_RETRY_CNT=7
NCCL_IB_ROUTE_DIAGNOSTICS=1
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_IB_SUBNET_PREFIX_LEN=24
NCCL_IGNORE_CPU_AFFINITY=1
NCCL_LL128_BUFFSIZE=262144
NCCL_MAX_NCHANNELS=4
NCCL_MIN_NCHANNELS=4
NCCL_NET=IB
NCCL_NET_PLUGIN=none
NCCL_OVERLAY_PIP=1
NCCL_P2P_DISABLE=0
NCCL_P2P_LEVEL=SYS
NCCL_PROTO=^LL128
NCCL_SET_THREAD_NAME=1
NCCL_SHM_DISABLE=1
NCCL_SWITCHLESS_RING_ONLY=1
NNODES=4
NVIDIA_DRIVER_CAPABILITIES=compute,utility
NVIDIA_VISIBLE_DEVICES=all
OFFLOAD_MODE=nvme
PATH=/opt/sglang/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PYTHONPATH=/opt/dsv41/adapter:/opt/b12x:/opt/b12x_next
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
SERVED_MODEL_NAME=deepseek-v4.1-flash
SERVER_PORT=8888
SGLANG_DSPARK_FOLDED_SAMPLING=2
SGLANG_DSV41_REASONING_EFFORT=75
SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0
SGLANG_ROCE_ALLREDUCE=1
SGLANG_ROCE_MAX_SIZE=262144
SGLANG_RUST_BUILD_MODE=never
SKIP_PREPARE=1
SKIP_VERIFY=1
SMOKE_QUICK=1
SPARK_PREFILL_TP_MIN_CONTEXT=32768
SPARK_PREFILL_TP_MIN_ROWS=1024
SPARK_PREFILL_TP_SPLIT=1
SPEC_ALGO=DSPARK
STATE_PATH=/state
TP_SIZE=4
WARMUP=1
```

### Per-rank overrides

The control sockets in this reference bind to the wired fabric. Verify IP connectivity to the head from all ranks, including diagonal routes. Moving GLOO/NCCL sockets to a separate management network is a legitimate site adaptation, but document it as a change.
#### rank0

```ini
# EXAMPLE: verify the local NIC/IP mapping before use.
NODE_RANK=0
HOST_IP=10.241.1.1
VLLM_HOST_IP=10.241.1.1
GLOO_SOCKET_IFNAME=enp1s0f0np0
NCCL_SOCKET_IFNAME==enp1s0f0np0
SKIP_SMOKE=0
```

#### rank1

```ini
# EXAMPLE: verify the local NIC/IP mapping before use.
NODE_RANK=1
HOST_IP=10.241.1.2
VLLM_HOST_IP=10.241.1.2
GLOO_SOCKET_IFNAME=enp1s0f1np1
NCCL_SOCKET_IFNAME==enp1s0f1np1
SKIP_SMOKE=1
```

#### rank2

```ini
# EXAMPLE: verify the local NIC/IP mapping before use.
NODE_RANK=2
HOST_IP=10.241.3.2
VLLM_HOST_IP=10.241.3.2
GLOO_SOCKET_IFNAME=enp1s0f1np1
NCCL_SOCKET_IFNAME==enp1s0f1np1
SKIP_SMOKE=1
```

#### rank3

```ini
# EXAMPLE: verify the local NIC/IP mapping before use.
NODE_RANK=3
HOST_IP=10.241.7.1
VLLM_HOST_IP=10.241.7.1
GLOO_SOCKET_IFNAME=enp1s0f0np0
NCCL_SOCKET_IFNAME==enp1s0f0np0
SKIP_SMOKE=1
```

## Container launch and acceptance

Example paths below are synthetic. Prepare the image and patched library locally on every node, install and validate the fabric, and create the model/Engram/state directories before launch. Start **rank1 → rank2 → rank3 → rank0**. `boot.py run` retains the native initialization/smoke/warmup flow; do not replace it with an unqualified direct server invocation.

```bash
ROOT=/srv/dsf41-r5
IMAGE=dsf41-r5:58f2321-arm64   # your already-built, validated local image
N=0                         # this node's logical rank
mkdir -p "$ROOT/state/rank$N" "$ROOT/autotune/rank$N"
docker run -d --pull never --runtime runc --gpus all \
 --name "dsf41-r5-r$N" --restart no --network host --ipc host \
 --privileged --cap-add IPC_LOCK --device /dev/infiniband \
 --shm-size 32g --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
 --env-file "$ROOT/config/common.env" --env-file "$ROOT/config/rank$N.env" \
 --env-file "$ROOT/secrets/api.env" \
 -v "$ROOT/models/DeepSeek-V4.1-Flash:/models/DeepSeek-V4.1-Flash:ro" \
 -v "$ROOT/engram/rank$N:/engram:ro" \
 -v "$ROOT/state/rank$N:/state" \
 -v "$ROOT/autotune/rank$N:/root/.cache/sglang/flashinfer/autotune" \
 -v "$ROOT/overlay/libnccl.so.2:/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2:ro" \
 "$IMAGE" run
```

Verify the image's pip NCCL path before using the overlay; another base may differ. Host IPC means `--shm-size` does not provide a private host-/dev/shm budget. Reference containers had no Docker memory limit, CPU pinning or automatic restart. Privileged networking is required by this historical recipe; a reduced-permission variant needs its own validation.

Bound waits and preserve failure logs (reference budgets: ranks ready 900 s, API verification 120 s, stop 120 s). Before replacing existing containers verify ownership, image, mounts and processes. Stop only your serving instance; keep monitoring running. Do not erase shared memory or global caches indiscriminately.

Acceptance requires all four ranks ready with their intended parameters, native smoke returning 42, a key-authenticated `/v1/models` request and one real generation, plus the transport checks above. Inspect actual process env/argv, not only the launcher. Native warmup can emit warnings without failing readiness, so check its actual completion. Preserve a rollback point for the prior service and network configuration.

Explicit reference rank0 server argv (API key omitted; deployment should still use `boot.py run`):

```json
[
  "/opt/sglang/bin/python3",
  "-m",
  "sglang.launch_server",
  "--model-path",
  "/models/DeepSeek-V4.1-Flash",
  "--served-model-name",
  "deepseek-v4.1-flash",
  "--trust-remote-code",
  "--load-format",
  "safetensors",
  "--tp",
  "4",
  "--ep-size",
  "1",
  "--attention-backend",
  "dsv4",
  "--moe-runner-backend",
  "flashinfer_mxfp4",
  "--mem-fraction-static",
  "0.80",
  "--chunked-prefill-size",
  "8192",
  "--context-length",
  "1048576",
  "--max-running-requests",
  "16",
  "--cuda-graph-max-bs-decode",
  "16",
  "--random-seed",
  "0",
  "--enable-decoder-swa-bounded-replay",
  "--tool-call-parser",
  "deepseekv41",
  "--reasoning-parser",
  "deepseek-v41",
  "--host",
  "0.0.0.0",
  "--port",
  "8888",
  "--speculative-algorithm",
  "DSPARK",
  "--speculative-dspark-block-size",
  "5",
  "--nnodes",
  "4",
  "--node-rank",
  "0",
  "--dist-init-addr",
  "10.241.1.1:20000",
  "--max-total-tokens",
  "6500000",
  "--fp8-gemm-backend",
  "flashinfer_cutlass",
  "--watchdog-timeout",
  "1800",
  "--enable-metrics",
  "--min-free-slots-delay",
  "1",
  "--enable-deepseek-v4-fp4-indexer",
  "--enable-cache-report",
  "--sleep-on-idle"
]
```

## Independent observations and limits

### sparkDash results: contributor R5 vs the author's published v2.3

The following uses the author's table layout. **These are different fleets and sampling schedules: percentages are observed differences from published values, not an isolated optimization gain or a formal benchmark ranking.** Negative differences are retained as well as positive ones. R5 data are the archived initial run already included in this PR; no new measurement is implied.

Author source: [README at 58f23215](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/58f232155917d388b6053eca079c617fd306c33c/README.md#current-results), measured 2026-09-29. Contributor source: [sanitized R5 result JSON](results/r5-wired-20261007.json), measured 2026-10-03. Difference = (R5 / author − 1) × 100, calculated from the displayed values.

**Contributor R5 Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | 84.44 | 119.90 (61.10) | 156.30 (40.58) | 224.74 (29.53) | 347.41 (22.98) |
| code | 129.86 | 182.57 (92.21) | 238.28 (62.85) | 323.61 (43.34) | 453.12 (30.19) |
| structured | 156.14 | 174.54 (89.39) | 267.76 (74.12) | 257.64 (44.93) | 537.83 (43.17) |
| json | 135.26 | 188.32 (96.68) | 295.06 (74.49) | 440.73 (56.85) | 673.80 (44.15) |

**Author Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | 89.7 | 125.2 (64.2) | 165.9 (41.9) | 244.6 (31.7) | 357.2 (23.2) |
| code | 131.9 | 181.4 (91.7) | 248.0 (63.5) | 328.6 (44.0) | 456.3 (30.5) |
| structured | 157.8 | 184.6 (106.9) | 242.1 (70.6) | 300.6 (44.6) | 603.4 (47.1) |
| json | 130.4 | 179.9 (92.8) | 319.8 (80.9) | 477.3 (61.0) | 693.2 (44.8) |

**Aggregate Decode: R5 vs author, observed difference**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | -5.86% | -4.23% | -5.79% | -8.12% | -2.74% |
| code | -1.55% | +0.64% | -3.92% | -1.52% | -0.70% |
| structured | -1.05% | -5.45% | +10.60% | -14.29% | -10.87% |
| json | +3.73% | +4.68% | -7.74% | -7.66% | -2.80% |

**Per-stream Decode: R5 vs author, observed difference**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | -5.86% | -4.83% | -3.15% | -6.85% | -0.95% |
| code | -1.55% | +0.56% | -1.02% | -1.50% | -1.02% |
| structured | -1.05% | -16.38% | +4.99% | +0.74% | -8.34% |
| json | +3.73% | +4.18% | -7.92% | -6.80% | -1.45% |

**Prefill, salted/cold-prefix repeated-token prompts, tok/s**

| fleet | 4k | 16k | 32k | 64k | 128k | 262k |
|---|---:|---:|---:|---:|---:|---:|
| Author, switched | 4688.00 | 5583.00 | 5650.00 | 5664.00 | 5398.00 | Not published in this table |
| Contributor R5, ring | 3142.73 | 5287.15 | 5694.98 | 5888.80 | 5462.13 | 5112.71 |
| R5 vs author, observed difference | -32.96% | -5.30% | +0.80% | +3.97% | +1.19% | N/A |

“Cold” refers to salted prefix prompts, not an empty Engram row cache. The repeated-token filler can favor Engram caching on both fleets. The 4k result is displayed for completeness and should not determine the overall conclusion; 262k has no matching entry in the author's current table. See the JSON for actual prompt tokens and TTFT. A positive aggregate difference does not necessarily mean a positive per-stream difference.

**Configuration and measurement differences**

| Item | Author's published v2.3 | Contributor R5 |
|---|---|---|
| Hardware / parallelism | Four DGX Spark GB10, TP4/EP1 | Four DGX Spark GB10, TP4/EP1 |
| Source line | v2.3 at 58f23215 | Same upstream source ref, local wired capacity profile |
| Fabric | Switched RoCE | Four-cable switchless ring, hardware-forwarded diagonal paths |
| Prefill chunk | 4096 | 8192 |
| RoCEnante all-reduce/gather cap | 2097152 bytes | 262144 bytes |
| Context / token pool | 1M context; reported boot pool 6.80M | 1048576 context; configured max-total-tokens 6500000 |
| GPU clock | Explicit 2200-MHz SM cap | Equivalence to that cap not established in the exported measurement record |
| Benchmark / generation | sparkDash1.8.8, max256 tokens, temperature0, thinking off | Same core benchmark/version and generation settings |
| Decode discarded warmup | Two prose c1 runs before sweep | Manager-internal32-token warmup each job; two extra discarded jobs at each type's c2/c8 |
| Decode repetitions | Each c1 median of3, c4 median of2; c2/c8/c16 once | Prose c1 median of5, code c1 median of3, remaining cells once |
| Prefill warmup / samples | 4k warmup; median of runs2 and3 of three runs per length | Manager512-token warmup; one six-length sweep |
| Native initialization | Not specified in the published comparison protocol | WARMUP=1, head quick smoke, workers skip smoke |
| Monitoring | README recommends10-second GPU/bandwidth polling | sparkDash remained running; polling equivalence not established here |


The accompanying [sanitized measurements](results/r5-wired-20261007.json) contain the initial local R5 sweep. These are **sparkDash diagnostic observations, not a formal benchmark ranking, a universal performance guarantee, or proof of root cause**. They are not an equal-protocol comparison against the upstream README's switched fleet.

Protocol: sparkDash 1.8.8, commit `754f40a7454fed1d26d47d1d68dd7bef930ce5ca`, original DecodeBench/PrefillBench managers driven from a workstation. The private orchestration runner SHA256 is `c00e8ae3ec50422b0d5475a9417a2016f1fb76d6ab2779c6a278d7841aac2d53`; it is identified here for provenance, not distributed as a public runner. Prompt types in order: prose, code, structured, json; concurrency 1,2,4,8,16 per type. Prose c1 repeats five times, code c1 three, others once; each c2/c8 has two extra discarded warmup jobs. Every decode job has the manager's internal 32-token warmup; Prefill has its internal 512-token warmup and one sweep over 4096/16384/32768/65536/131072/262144. Total 42 decode jobs plus one Prefill job. Native model warmup remains as above. Greedy/temperature 0, thinking off, decode max output 256. Per-cell repeated statistics are medians; per-stream decode and aggregate decode are separate fields.

The historical decode prompts are fixed and repeated, so prefix caching may occur. Prefill uses a UUID-salted prompt plus repeated ` the` filler, which warms the Engram row cache and is not representative natural-text Prefill. Actual token counts are included. The initial R5 structured c8 extra warmup contained one 97-token output; it was excluded from formal cell statistics. All formal R5 streams completed 256 tokens. Monitoring remained running during benchmark windows. Preserve its sampling configuration when repeating a comparison.

Initial local R5 c1 observations were prose 84.44, code 129.86, structured 156.14 and json 135.26 tok/s. Prefill was 4k 3142.73, 16k 5287.15, 32k 5694.98, 64k 5888.80, 128k 5462.13 and 262k 5112.71 tok/s. The complete concurrency matrix, actual prompt lengths and TTFT are in the accompanying JSON. These are results of this recorded run, **not an isolated measurement of the benefit of any individual switch or a claim of superiority over another fleet**.
The recorded boot/API and parameter checks are deployment validation. They do not substitute for the repository's qeval, KL, sampling, prefix-cache and long-context quality gates on a new recipient fleet.

## Sharing and rollback

All host addresses in examples are synthetic. No private API key, SSH credential, real MAC, user home, GPU UUID, container identity or original state/autotune cache is included. Source and artifact hashes identify versions and are not access credentials.

Before changing fabric configuration or disabling mesh, stop the model normally and verify QPs/processes are gone. Retain the prior profile and network rollback assets. Follow the recipient's site policies; this document is deployment information, not authorization to disrupt a running service.
