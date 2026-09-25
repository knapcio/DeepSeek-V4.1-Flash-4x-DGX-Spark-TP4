# Switchless ring: four Sparks, no RoCE switch

`NCCL_SWITCHLESS_RING_ONLY=1` boots this stack on four DGX Sparks cabled as a ring
(a-b-c-d-a, one DAC per adjacency per rail, no switch in between). It is **off by
default and changes nothing when off**: the switch only decides whether the ring
NCCL environment and the overlay mount are injected.

The production line's RoCEnante needs a path to the opposite node as well; how to build it in the
neighbours' ConnectX-7 hardware is in
[RoCEnante on the ring](#rocenante-on-the-ring-hardware-forwarded-opposite-node-paths).

## Why the defaults do not boot on a ring

```
NCCL INFO Trees [0] 2/-1/-1->0->-1
NCCL INFO Connected all rings, use ring PXN 0 GDR 0
...
RuntimeError: NCCL error: unhandled system error
    at ncclTransportTreeConnect -> ncclCommInitRank
```

NCCL builds a **tree** in addition to the ring. In a four-node ring the tree wants a
direct fabric path between opposite nodes (rank0 ↔ rank2), and there is none: the
diagonal exists at the IP layer through a transit node, but RoCE queue-pair setup
does not follow IP routing. The tree therefore never connects and the boot dies
inside `ncclCommInitRank`. The NCCL counters and `/health` never come up.

The fix is two environment variables plus a patched NCCL:

| | |
|---|---|
| `NCCL_ALGO=Ring` | no tree, ring only |
| `NCCL_SKIP_TREE_CONNECT=1` | belt and braces: the patched build refuses the tree connect outright |
| patched NCCL | `sparkring`'s `switchless-cycle` / `skip-tree-pat` patches (FujitsuPolycom/sparkring) |

## Cabling

Four DACs, no switch and no diagonal links:

```text
  rank0                         rank1
  CX7-0 -------- cable 1 ------- CX7-1
  CX7-1                         CX7-0
    |                             |
  cable 4                       cable 2
    |                             |
  CX7-0                         CX7-1
  CX7-1 -------- cable 3 ------- CX7-0
  rank3                         rank2
```

```text
cable 1: rank0 CX7-0 <-> rank1 CX7-1
cable 2: rank1 CX7-0 <-> rank2 CX7-1
cable 3: rank2 CX7-0 <-> rank3 CX7-1
cable 4: rank3 CX7-0 <-> rank0 CX7-1
```

Use a separate `/24` per cable, or one `/24` with `NCCL_IB_SUBNET_PREFIX_LEN=24` and
`NCCL_IB_SUBNET_AWARE_ROUTING=1` (the default this switch relies on). Keep the
management LAN on a different interface (`GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`)
— the ring carries the data plane only.

## Setup

1. Build/install the patched NCCL into `NCCL_HOST_DIR` (default `$HOME/nccl-2.30.7`)
   on **every** node. The library must contain the `SWITCHLESS_RING_ONLY` marker;
   the preflight rejects a stock build, so a wrong library fails before any
   container is replaced rather than at NCCL init.

2. Put a local checkpoint on every node and set `NFS_SHARE=0`. A ring has no
   fabric-wide NFS path, so the tested configuration keeps its own copy of the
   weights on every Spark — see [Local weights](#local-weights).

3. Uncomment the ring block in `.env.tp4`:

   ```ini
   NCCL_SWITCHLESS_RING_ONLY=1
   NCCL_ALGO=Ring
   NCCL_P2P_LEVEL=SYS
   ```

4. Validate, then serve:

   ```bash
   ./start.sh doctor
   ./start-tp4.sh serve
   ```

## Local weights

`NFS_SHARE=1` (the default) exports the head's checkpoint and gives every worker an
NFS-backed docker volume. On a ring that is the wrong shape: the share would ride the
same four DACs the ring needs for collectives, opposite nodes would reach the head
through a transit node, and a dead exporter takes the whole serve down. `NFS_SHARE=0`
keeps `$NFS_VOLUME` (default `dsv41-weights`) local on each node instead.

It is a pre-existing switch in this repo, extended here to cover the whole profile.

### What it turns off

With `NFS_SHARE=0` none of the NFS apparatus is touched:

* no exporter container on the head, no `nfs_publish_model`, no `/etc/exports` rewrite;
* no per-worker docker volume creation, so no `nfs-common` on the workers and no client
  ACL to get wrong — the ACL is the usual reason a worker cannot see the checkpoint;
* `NFS_SERVER_IP`, `NFS_SERVER_IPS`, `WORKER_FABRIC_IPS` and the `NFS_SERVER_IP_<n>`
  legacy variables become unused. Nothing reads them, so leaving them set is harmless;
* `$COMMON_MODEL` (the head's symlink to `MODEL_DIR`) is still created, but nothing
  consumes it on a worker any more — see below.

`cmd_share` returns immediately under `NFS_SHARE=0`. It stays idempotent, so `serve` can
keep calling it unconditionally without knowing which profile it is in.

### What it checks, and when

`serve` validates **before** it replaces containers — a half-provisioned node otherwise
fails twenty minutes into a load, with four containers already down:

* the head: a readable, non-empty `config.json` plus at least `EXPECTED_SHARDS` readable,
  non-empty shards (`local_model_has_weights`);
* every worker: the same, inside its own `$NFS_VOLUME` (`nfs_worker_has_model`).

`doctor` runs the same per-worker check in **both** profiles, so a missing node shows up
in the pre-flight rather than in `serve`. It reports the weights source once:

```
[+] weights: NFS_SHARE=0 — head reads $HOME/NewModels/DeepSeek-V4.1-Flash; workers use local volume dsv41-weights
```

### What it fixed

* **`status` read the wrong path.** It tested `$COMMON_MODEL/config.json` on each worker,
  but `$COMMON_MODEL` is a **head-side symlink to `MODEL_DIR`** and never exists on a
  worker, so healthy workers reported `weights:MISSING` in this profile. It now inspects
  the worker's own volume and prints `weights:OK (dsv41-weights)`.
* **The probe could create what it was checking.** `docker run -v $NFS_VOLUME:/m` creates
  an empty volume on a typo, and the old probe pulled a mutable `alpine:latest` to do it.
  It now inspects the volume first, then runs the serving image already required by
  `serve` with `--pull=never --network none`, needs neither network nor GPUs, and checks
  the shard count instead of only `config.json`.

### Migrating from `NFS_SHARE=1`

A leftover NFS-backed volume **keeps the same name** and still contains `config.json`, so
the existence probe passes and the worker quietly keeps reading over NFS — the exact
thing this profile exists to avoid, and it fails confusingly the moment the exporter is
gone. `nfs_unmount_workers()` exists but is not wired to a command, so nothing removes it
for you. `serve` therefore refuses it outright:

```
$ grep -q '"type":"nfs"' <<< "$(docker volume inspect -f '{{json .Options}}' dsv41-weights)"
$ ssh spark2 docker volume rm dsv41-weights
[+] spark2: weights:OK (dsv41-weights)      # after copying the checkpoint locally
```

`Driver` cannot tell the two apart — a plain volume and an NFS volume both report
`local`. The mount options can: a plain volume reports `null`, an NFS one reports
`{"type":"nfs",...}`.

Provisioning is then a plain copy per node, for example from the head:

```bash
rsync -a --info=progress2 $MODEL_DIR/ spark2:$MODEL_DIR/
./start.sh doctor          # one weights line per worker
./start-tp4.sh serve
```

Two consequences worth planning for: the checkpoint costs **4× the disk** (476 GiB per
node here), and a model update has to be pushed to all four nodes before the next
`serve` — a node left behind fails the content check rather than serving stale weights.

## What the switch does

With `NCCL_SWITCHLESS_RING_ONLY=1`:

* injects `NCCL_SWITCHLESS_RING_ONLY=1`, `NCCL_ALGO=Ring`,
  `NCCL_SKIP_TREE_CONNECT=1`, `NCCL_IB_SUBNET_PREFIX_LEN=24`,
  `NCCL_MIN_NCHANNELS=4` and `NCCL_P2P_LEVEL=SYS` into the head **and every worker**
  (each value still overridable from the env file);
* mounts the patched library **over** the image's pip NCCL
  (`NCCL_PIP_SO`, default `/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2`)
  instead of putting it on `LD_LIBRARY_PATH`. This matters if you use DeepEP: two
  visible NCCL runtimes make `check_nccl_so()` abort before NCCL is initialised;
* validates the configuration and every rank's HCA/GID in `doctor`, and fatally in
  `serve` — the check sits before any container is replaced.

`NCCL_OVERLAY_PIP` defaults to following the switch, so the overlay can also be
enabled on a switched fabric on its own (`NCCL_OVERLAY_PIP=1`, switch off).

## Expected logs

```
NCCL INFO Connected all rings, use ring PXN 0 GDR 0
NCCL INFO NCCL_SWITCHLESS_RING_ONLY set by environment to 1.
NCCL INFO Tree transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
NCCL INFO PAT transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
```

`doctor` prints one line per rank:

```
[+] switchless ring: config OK (NNODES=4 TP=4 EP=2, IB_HCA=rocep1s0f0,rocep1s0f1)
[+] switchless ring: head preflight OK (RoCEv2 GID index 3)
[+] switchless ring: 10.0.0.2 preflight OK (RoCEv2 GID index 3)
[+] switchless ring: 10.0.0.3 preflight OK (RoCEv2 GID index 3)
[+] switchless ring: 10.0.0.4 preflight OK (RoCEv2 GID index 3)
```

## Devices past the second are never advertised

`NCCL_IB_HCA` accepts any number of devices, and NCCL says nothing when it cannot
use them. The switchless-cycle patch publishes at most **two** listener GIDs per
rank — `gidSlot < 2` in `net_ib/connect.cc`, in both the original and the patched
loop. Devices after the second are therefore absent from the handle every peer
receives, no peer can match their subnet, and their ports carry zero bytes. The
ring still forms and serves; it just runs on the first two devices.

The symptom is easy to miss because the channel plan looks right:

```
NCCL INFO NET/IB : Using [0]rocep1s0f0:1/RoCE [1]rocep1s0f1:1/RoCE \
                   [2]roceP2p1s0f0:1/RoCE [3]roceP2p1s0f1:1/RoCE [RO]
NCCL INFO Channel 02/0 : 0[0] -> 1[0] [send] via NET/IB/2
```

Both lines name device 2, and it still moves nothing. What actually happens is a
silent collapse onto the first device, visible only in the routing log:

```
NCCL INFO NET/IB: Subnet-aware routing: overriding dev 2 with dev 0
NCCL INFO NET/IB: Subnet-aware routing: overriding dev 3 with dev 0
```

`doctor` now says so instead of leaving it to be discovered by counters:

```
warning: IB_HCA lists 4 devices but listener GID publication is capped at 2
warning: only rocep1s0f0 and rocep1s0f1 can be selected by a peer; the rest stay at zero
warning: set NCCL_IB_EXTENDED_IPV4_GIDS=1 with a dual-PCI-domain NCCL build
```

### Raising the cap

A four-Spark board exposes its ConnectX-7 functions through **two PCI root
domains** (`0000:` and `0002:` here), which NCCL discovers as four separate
devices:

```
[0] rocep1s0f0    pciPath=/sys/devices/pci0000:00/.../0000:01:00.0
[1] rocep1s0f1    pciPath=/sys/devices/pci0000:00/.../0000:01:00.0
[2] roceP2p1s0f0  pciPath=/sys/devices/pci0002:00/.../0002:01:00.0
[3] roceP2p1s0f1  pciPath=/sys/devices/pci0002:00/.../0002:01:00.0
```

FujitsuPolycom/sparkring's cumulative
[`nccl-2.30.7-dual-pci-domain.patch`](https://github.com/FujitsuPolycom/sparkring/blob/main/spark_transport/nccl/DUAL_PCI_DOMAIN.md)
raises the bound to four behind a flag, and adds a fallback that substitutes a
device **within the same PCI root** rather than collapsing across domains. Apply
it **alone** — it already contains the switchless-cycle changes, so it must not be
layered over them.

```ini
NCCL_IB_EXTENDED_IPV4_GIDS=1     # publish up to four IPv4-mapped listener GIDs
NCCL_IB_PRESERVE_PCI_DOMAIN=1    # substitute within the selected PCI root
NCCL_IB_ROUTE_DIAGNOSTICS=1      # one record per final QP: which device it landed on
NCCL_IB_QPS_PER_CONNECTION=1
```

Set `IB_HCA` to all four devices and the ring uses both planes. Every value has to
reach every rank, head and workers alike.

The flags are read at NCCL init, so the effect is visible before any request:

```
NCCL INFO NET/IB ListenerRouting format=ipv4-v1 advertised=4 observed=4
```

`advertised=2` means the cap is still in force. The routing records then stop
collapsing: `overriding dev 3 with dev 2` stays inside the second PCI root instead
of reaching for `dev 0`.

### The second plane's network has to exist before those flags can reach it

The flags raise the *publication* bound; they do not give the second card a network.
On the four Sparks the second ConnectX-7 came cabled and up (200G, link detected) but
unconfigured: no IPv4, MTU 1500, and the RoCE v2 GID that `NCCL_IB_GID_INDEX` selects
(index 3 here) reading back all-zero. NCCL cannot match a subnet for a device that has
no GID at the selected index, so the channel plan names it and the port still carries
nothing:

```text
NET/IB : Using [0]rocep1s0f0 [1]rocep1s0f1 [2]roceP2p1s0f0 [3]roceP2p1s0f1
Channel 02/0 : 0[0] -> 1[0] [send] via NET/IB/2      <- moves nothing
```

`port_xmit_data` on `roceP2p1s0f0` stayed flat for the whole run while the first card
carried 100 % of inter-node traffic — the 0.00 GB column in the table below. Setting the
four flags without this step leaves the ports where they were: the flags decide whether a
device may be advertised, the addressing decides whether there is a GID to advertise.

Both ports of the second card are part of the ring, one cable each to the two neighbours,
with the same geometry as the first card (`f0` to the next rank's `f1`). Each node
therefore needs one address per cable, a 9000 MTU, and a route for the two /24s it does
not sit on: those subnets exist only at the IP layer, through a transit neighbour.

A minimal template, per node as `/etc/netplan/41-sparkring-plane2.yaml`. Cable `<i>` is
the leg rank`i` -> rank`(i+1) mod 4`; shown for rank0, the other three are the same file
with `i` shifted and the two /24s adapted (they are placeholders — use your own scheme):

```yaml
network:
  version: 2
  renderer: NetworkManager        # the Sparks' fabric ports are NetworkManager-managed
  ethernets:
    enP2p1s0f0np0:                # -> RDMA device roceP2p1s0f0
      addresses: [10.10.0.10/24]  # cable 0, this end
      dhcp4: false
      dhcp6: false
      mtu: 9000
      optional: true             # never block boot on it
      routes:
      - to: 10.10.1.0/24          # cable 1, one hop away
        via: 10.10.0.11           # rank1's f1, on this cable
    enP2p1s0f1np1:                # -> RDMA device roceP2p1s0f1
      addresses: [10.10.3.11/24]  # cable 3, this end
      dhcp4: false
      dhcp6: false
      mtu: 9000
      optional: true
      routes:
      - to: 10.10.2.0/24          # cable 2, one hop away
        via: 10.10.3.10           # rank3's f0, on this cable
```

`sudo netplan apply`, then confirm the four addresses and the MTU are up
(`ip -br addr`, `ip -d link show enP2p1s0f0np0`) and that the interface names still
map to the `IB_HCA` list you set. The network is only right when the counters agree:
with the addressing and the four flags in place the second root moved 63.60 GB under a
64k prefill plus 16 streams and took 49.7 % of inter-node traffic, and the routing
record reads

```text
NET/IB: Subnet-aware routing: overriding dev 3 with dev 2 preserving PCI root pci0002:00
```

instead of collapsing to `dev 0`. Keep that order when debugging — addressing first,
then the flags, then the counters. The `NET` subsys stays useful here: this record is
the only place that states which root a channel landed on.

Removing the plane takes two steps, not one: NetworkManager can write its own
`/etc/netplan/90-NM-*.yaml` stanzas for these interfaces, and any of those that survive
bring the addresses back on the next `netplan apply` (`grep -l enP2p1s /etc/netplan/*`).
If you are rolling the plane back completely, `rm` those too and flush the addresses
(`ip addr flush dev enP2p1s0f0np0 enP2p1s0f1np1`).

### Channel count

`NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS` decide how many channels share the
devices. Four channels over four devices gives one channel per device, which is
the mapping that reaches all of them:

```ini
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4
```

Eight channels over four devices still round-robins 0,1,2,3,0,1,2,3, so it is not
wrong, but a four-versus-eight comparison on this workload found no serving
benefit and 0.14 GiB more head-node shared memory
([sparkring#193](https://github.com/FujitsuPolycom/sparkring/issues/193)).

### What it is worth

Measured here on four Sparks, TP4 / EP2, DSpark k=5, one 64k prefill plus 16
concurrent streams, IB port counters before and after:

| port | PCI root | before | after |
|---|---|---:|---:|
| `rocep1s0f0` | 0000 | 65.45 GB | 32.20 GB |
| `rocep1s0f1` | 0000 | 65.45 GB | 32.19 GB |
| `roceP2p1s0f0` | 0002 | **0.00 GB** | **31.80 GB** |
| `roceP2p1s0f1` | 0002 | **0.00 GB** | **31.80 GB** |

Half the traffic moves to the second plane. The total is unchanged — this spreads
the same collectives over twice the ports, it does not make them smaller. The
ported case is bounded by what the ring was waiting on, not by cable bandwidth:
the ports ran at roughly 5 % of line rate under this load, so expect a low
single-digit prefill gain and no decode change, matching the
[contributor measurement](https://github.com/FujitsuPolycom/sparkring/blob/main/performance/records/transport/nccl-dual-domain-deepseek.md)
of +5.43–6.82 % prefill for this exact model and runtime. Verify with counters and
the routing records rather than trusting the channel plan.

### Diagnosing it

`ListenerRouting` and the routing records are logged at the `NET` level. With
`NCCL_DEBUG_SUBSYS=INIT,ENV` they never appear and the collapse is invisible:

```ini
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,ENV,NET    # add NET while validating; drop it afterwards
```

## Pitfalls

* **`EP_SIZE` is free, `TP_SIZE` is not.** The ring needs `NNODES == TP_SIZE == 4`
  (the ring spans the tensor-parallel group). `EP_SIZE` only decides how the MoE
  all-to-all is grouped, so `1 <= EP_SIZE <= TP_SIZE` is accepted and `EP_SIZE=2`
  is a common choice on this fabric.
* **A wrong GID index is the usual failure.** All ports listed in `IB_HCA` must
  share one nonzero IPv4-mapped RoCE v2 GID; the preflight finds it or validates
  your `NCCL_IB_GID_INDEX` override. Management IPs need not appear in the GID table.
* **Do not also set `NCCL_SWITCHLESS_RING_ONLY=1` with a switched fabric.** The ring
  skips the tree, which is a performance loss when the tree is reachable.
* **A ring is not a non-blocking fabric.** Opposite ranks talk through a transit
  node, so a four-node ring's bisection bandwidth is one link, not two. Without the
  opposite-node paths below, expect the decode numbers under Measured rather than the switched ones.

## RoCEnante on the ring: hardware-forwarded opposite-node paths

RoCEnante's one-shot all-reduce and all-gather write every rank's payload straight into every
peer's buffers, and a four-node ring has no link between opposite nodes, so out of the box a ring
runs the production line with `SGLANG_ROCE_ALLREDUCE=0` and the collectives go through the patched
NCCL (~56 us per decode-size all-reduce, ~90 of them per decode step). The missing path can be built
without a switch or extra cables, in the neighbours' ConnectX-7 hardware, with the design of
[FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring) (`cx7_hairpin_diagonal`,
commit `f16b5f4`):

- the sender's NIC re-tags the RDMA packets of the opposite-node queue pairs (flow label 16383, i.e.
  UDP source port 65535) from EtherType 0x0800 to 0x88b5 (an RDMA-TX flow rule, `mlx5-rdma-tx-rewrite-probe`);
- a `/32` route sends them to the neighbour on that cable;
- on the neighbour a `skip_sw` tc flower rule matches the tag and the two MACs, restores 0x0800,
  rewrites the destination MAC and redirects the packet out of its other port (mlx5 hairpin queues).

No CPU touches the forwarded packets and the kernel forwards nothing (`nstat IpForwDatagrams` stays
flat); a marked packet that misses the rule is dropped, never routed in software. Every rank uses
two paths per peer over its four RDMA functions: the neighbours over their own cable, the opposite
node through each neighbour (one path per PCIe domain). `DSV41_ROCE_RING=1` makes the SG17 overlay
load `b12x.comm.roce_ring`, sparkring's path-aware RoCEnante (provenance and local changes in
`runtime/b12x/roce_ring-provenance.json`), instead of `b12x.comm.roce`; unset, nothing changes.
`DSV41_L2_PREFETCH` hooks either package.

### What each step is worth

Same four-Spark ring, same day, one change at a time (qeval = `scripts/qeval.py`, 75 tasks at c1;
step time = the engine's `spec_verify_ct`, time to first token subtracted):

| Step | Effect |
|---|---|
| NCCL ring -> RoCEnante over the mesh (c5cee32 stack, 80 KB cap) | decode step -5.2 % (median over 51 qeval tasks, faster on 50), qeval median 76.3 -> 79.5 tok/s |
| proxy idle spins 200000 -> 20000000 (the SG17 value) | c1 step 40.6 -> 37.1 ms (-8.6 %), c2 -7 %, c4 -6 %, c8 -2 %; with 200000 the proxy was asleep in 27 % of samples during decode |
| `hairpin_queue_size` 8192, 256 KB cap, two-wave off | c2 -1 %, c4 -1.3 %; c1 unchanged (its collectives are 50-60 KB) |
| this README's v2 (prefill SP, L2 prefetch, draft head) | c1-c8 step -1 to -3 %, prefill +17-18 % at 16k-128k, +10 % at 262k |

### Results

The switched README's production line (v2, `7ac7123`) with the ring additions, built from this
repository (`Dockerfile.canary-roce`), measured on the four-Spark ring the same day, against the
README's v2 numbers (v2.1's `DSV41_PREFILL_SP_FP8` came later; it is fabric-independent and was not in
this run). Raw output: [`docs/results/ring-mesh-20260925.txt`](results/ring-mesh-20260925.txt).

| | Ring (this) | Switched (README at v2) |
|---|---:|---:|
| qeval median tok/s (75 tasks), pass | 84.7 (median of 3 runs), 72/75 | 83.5, 72/75 |
| decode step, prose-type prompts | 33.3 ms (2.0 tok/step) | 33.0 ms (2.27 tok/step) |
| decode step, code-type prompts | 38.1 ms (3.74 tok/step) | 39.2 ms (3.87 tok/step) |
| sparkDash 1.8.8 prose c1 / c16 | 80.9 / 345.1 | 86.5 / 342.7 |
| code c1 / c16 | 120.5 / 446.6 | 122.6 / 438.3 |
| structured c1 / c16 | 150.2 / 547.7 | 152.4 / 572.2 |
| json c1 / c16 | 134.2 / 671.5 | 118.9 / 659.9 |
| prefill 16k-128k / 262k | 5,393-5,567 / 4,971 | 5,644-5,818 / 5,214 |
| phrase needle | PASS at 1,030,651 tokens (305 s) | PASS at 1,011,084 tokens (322 s) |

The step-time prompts differ (the README's are not published), so the two columns are near but not
identical acceptance. Prose c1 on sparkDash is the single-prompt case under Mesh pitfalls below. Prefill
stays 4-5 % under the switched fabric at 16k-128k and 5 % at 262k: the large prefill collectives run on
NCCL over the ring's one-link bisection. The KV pool was 5.56 M tokens on this boot and 6.33-6.35 M on
the two before it (the fast loader's boot-to-boot spread).

### Setup

1. **The ring as above**, with both planes addressed (four RDMA functions per node, MTU 9000, the
   RoCEv2 GID at index 3), and the NIC profile sparkring's hardware forwarding was qualified on:
   `hairpin_num_queues` 4, `flow_steering_mode` `hmfs`, eswitch `legacy`, `hw-tc-offload on` on all
   four fabric netdevs (sparkring's
   [driver configuration notes](https://github.com/FujitsuPolycom/sparkring/blob/main/docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md#connectx-7-driver-configuration-for-hardware-forwarding)).
   `scripts/ring_mesh/inventory.sh` prints all of it per node; `plan.py` refuses a node whose links,
   MTU, GID, TC offload or steering mode do not match.
2. **The source marker**, built on every node from a sparkring checkout at `f16b5f4`
   (source sha256 `8684a696…`; it built to `2828c07e…` here, the binary sparkring records):

   ```bash
   git clone https://github.com/FujitsuPolycom/sparkring ~/sparkring && git -C ~/sparkring checkout f16b5f4
   sudo install -d /opt/dsv41-mesh/bin
   cc -O2 -Wall -Wextra ~/sparkring/spark_transport/fabric/cx7_hairpin_diagonal/native/mlx5_rdma_tx_rewrite_probe.c \
      -o /tmp/mlx5-rdma-tx-rewrite-probe -libverbs -lmlx5 && sudo install -m 755 /tmp/mlx5-rdma-tx-rewrite-probe /opt/dsv41-mesh/bin/
   ```
3. **The plan**, from the head, with the hosts in TP rank order (the head, then `WORKER_HOSTS`):

   ```bash
   python3 scripts/ring_mesh/plan.py --sparkring ~/sparkring --out ring-mesh spark1 spark2 spark3 spark4
   ```

   It inventories the nodes over SSH, reads the cabling from the fabric subnets, numbers the ring the
   way sparkring's planner needs it (every f0 port cabled to the next node's f1; it refuses anything
   else), and has sparkring's planner build the RoCEnante selection: per node two `/32` routes, two
   tc rules and two markers. It writes `mesh-up-<host>.sh` / `mesh-down-<host>.sh` and `env.txt`,
   the `EXTRA_CONTAINER_ENV` additions with the per-rank peer maps already translated to the TP rank
   order (sparkring numbers the ring by cabling direction, which need not match it).
4. **Install on every node** with the engine stopped (applying the hairpin size re-initialises each
   fabric function); `$HOST` is that node's name as given to `plan.py`:

   ```bash
   sudo install -m 755 ring-mesh/mesh-up-$HOST.sh /opt/dsv41-mesh/mesh-up.sh
   sudo install -m 755 ring-mesh/mesh-down-$HOST.sh /opt/dsv41-mesh/mesh-down.sh
   sudo install -m 755 scripts/ring_mesh/hairpin.sh /opt/dsv41-mesh/
   sudo install -m 644 scripts/ring_mesh/dsv41-mesh.service scripts/ring_mesh/dsv41-mesh-marker@.service /etc/systemd/system/
   sudo systemctl daemon-reload && sudo systemctl enable --now dsv41-mesh
   ```

   `dsv41-mesh.service` runs at every boot: it waits for the fabric links, sets `hairpin_queue_size`
   (a `driverinit` parameter that resets at boot) and applies the routes, rules and markers; `mesh-up.sh`
   refuses a rule that did not land in hardware. Start the engine after it is active.
5. **Verify the path** before booting the engine: every rule shows `in_hw` (`tc -s filter show dev
   <netdev> ingress`), and an RDMA write to the opposite node goes through the neighbour's rule, not its
   kernel (`ib_write_lat -d <dev> -x 3 --flow_label=16383` against the opposite node's port: ~10 us at
   61 KB here against 8.5 us to a direct neighbour; the neighbour's rule counters rise and its
   `IpForwDatagrams` does not).
6. **`.env.tp4`**: build `Dockerfile.canary-roce` as usual and append `env.txt` to the production
   `EXTRA_CONTAINER_ENV`, replacing its `B12X_ROCE_HCA`, `SGLANG_ROCE_MAX_SIZE` and `DSV41_ROCE_GATHER`.
   The boot log shows `RoCEnante ready: world=4 hcas=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1`
   and `DSV41_L2_PREFETCH: RoCE collectives prefetch the next weights into L2`.

### The size cap and the hairpin queues

The forwarded traffic crosses the neighbour in a hairpin queue, and at the driver default
(`hairpin_queue_size` 1024) a burst of more than ~100 KB per message overflows it:
`rx_out_of_buffer` rises on the forwarding ports, the far end counts `out_of_sequence` /
`packet_seq_err`, and go-back-N retransmits make a 120 KB all-reduce 3-4x slower than NCCL. Measured
in CUDA graphs, all four ranks, drops summed over all 16 functions:

| all-reduce | 60 KB | 100 KB | 120 KB | 240 KB | 480 KB | 960 KB |
|---|---:|---:|---:|---:|---:|---:|
| queue 1024: us/op | 23 | 36 | 111-141 | 91-109 | 167-191 | 378-449 |
| queue 1024: drops | 0 | 163 | many | many | many | many |
| queue 8192, two-wave off: us/op | 24 | 35 | 40 | 60 | 95 | 274 (two-wave on) |
| queue 8192: drops | 0 | 0 | 0 | 0 | 0 | 64 |

At 1024 keep `SGLANG_ROCE_MAX_SIZE=DSV41_ROCE_GATHER=81920` (every c1 decode collective is 50-60 KB, so
c1 loses nothing); at 8192 the 256 KB cap also moves c2-c4 and the draft's 129 KB vocabulary gathers.
`plan.py` picks the cap from the smallest queue it finds. The package's two-wave schedule (direct
paths first, forwarded paths after, from 128 KB) only costs once nothing drops:
`B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES=0` turns it off.

### Mesh pitfalls

- **Never re-initialise a fabric function, stop `dsv41-mesh` or a marker while the engine runs.**
  Opposite-node traffic stops, and a re-init drops every RDMA queue pair on that function; the RoCE
  health check then fails the step. `hairpin.sh` skips functions already at the value, so re-running
  it with the same value is safe.
- The marker rewrites every RDMA packet with UDP source port 65535 on its device, whatever the
  destination: keep that port reserved on the fabric.
- Without the mesh neither package has a path to the opposite node (`DSV41_ROCE_RING=1` or not): a
  ring without it runs `SGLANG_ROCE_ALLREDUCE=0` and no `DSV41_ROCE_GATHER`, as before.
- **Measuring.** The mesh changes the reduction order, so sparkDash's single greedy prose prompt takes
  a different text and its acceptance moves with it: here prose c1 read 72.0 on the mesh against 76.4 on
  NCCL while the step was 5 % faster. Compare step time (`spec_verify_ct`) or qeval's median over its
  75 tasks; that median itself moves 2-3 % from run to run (81.8 / 84.7 / 85.9 on one boot here), so take
  the median of several runs.
- CPU pinning does not help: the scheduler on the X925 cores measured -5.5 %, the proxy threads alone
  on dedicated X925 cores neutral.

Rollback: `DSV41_ROCE_RING=0 SGLANG_ROCE_ALLREDUCE=0` without `DSV41_ROCE_GATHER` in `.env.tp4`, then
`sudo systemctl disable --now dsv41-mesh` on every node (removes the markers, rules and routes).

## Measured

The ring without the opposite-node paths (collectives on NCCL): four GB10 Sparks in a ring (`a-b-c-d-a`, no switch), TP4 / EP2, 1M context,
DSpark k=5, local weights (`NFS_SHARE=0`), canary image, this switch on. sparkDash's
benchmark panel, one engine, no other load.

Boot, first time:

```
NCCL INFO Connected all rings, use ring PXN 0 GDR 0
NCCL INFO NCCL_SWITCHLESS_RING_ONLY set by environment to 1.
NCCL INFO Tree transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
NCCL INFO PAT transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
parallel: nnodes=4 TP=4 EP=2
```

All four ranks healthy, `doctor: ready`, `/health` 200, `--enable-cache-report` live
(cold request `prompt_tokens_details: None`, warm `{'cached_tokens': 1024}`).

### Prefill, cold

| context | 1k | 4k | 8k | 16k | 32k | 64k |
|---|---:|---:|---:|---:|---:|---:|
| prompt tokens | 1,041 | 4,116 | 8,213 | 16,405 | 32,793 | 65,555 |
| TTFT | 406 ms | 1.11 s | 2.01 s | 3.89 s | 7.67 s | 15.43 s |
| tok/s | 2563 | 3712 | 4081 | 4218 | 4274 | 4249 |

**Caveat, the same one as the README's prefill table:** sparkDash's prefill filler is one
repeated token, so every filler token hits the same Engram row and the row cache
(`DSV41_CACHE_GIB`) inflates the 16k-128k column by roughly 9-20 % (reported by
koldfrontier in MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks#21). Treat the shape as real and
the absolute numbers as an upper bound; a cold single request with a 53,613-token natural
prompt took **17 s** on the same boot, which is the same order as the 32k row above.

### Decode, 400 output tokens

Prose:

| concurrent streams | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| aggregate tok/s | 60.5 | 82.9 | 116.4 | 186.1 |
| per stream | 60.5 | 41.4 | 30.3 | 24.0 |
| TTFT | 182 ms | 209 ms | 263 ms | 286 ms |

Code:

| concurrent streams | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| aggregate tok/s | 107.5 | 189.6 | 326.4 | 521.6 |
| per stream | 107.5 | 94.8 | 81.6 | 65.2 |
| TTFT | 257 ms | 313 ms | 381 ms | 533 ms |

Decode is where a ring is the right trade: aggregate scales close to linearly through
eight streams (prose 60.5 → 186.1, code 107.5 → 521.6) while per-stream decay stays
gentle, which is what the 16-slot decoder and DSpark are for. Prefill is where the
bisection shows: ranks on opposite sides of the ring talk through a transit node, so a
four-node ring's bisection is one link, not two. That is the price of having no switch,
not a way to beat one.

### Reference

The ring configuration was originally contributed as
[MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks #3](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/3)
by [@Saolence](https://github.com/Saolence), carried forward with an all-rank preflight,
the overlay-mount strategy and regression fixtures in
[#19](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/19). The NCCL
transport patch itself is from
[FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring).
