// RDMA proxy for the b12x RoCE one-shot all-reduce, plus the pipelined two-shot
// all-reduce ported from flashinfer-ai/flashinfer#6915 (Luke Alonso,
// flashinfer/experimental/b12x/comm/roce/_roce_proxy.c, Apache-2.0) into the
// ds41 RoCEnante copy (local-inference-lab/b12x 081b2359, rhys101's SG17 overlay).
//
// One rank owns one pinned host region laid out as:
//
//   recv[src][slot]  (world * SLOTS * slot_bytes)  filled by peers' RDMA writes
//   flag[src][slot][hca] sequence number written on each HCA after that HCA's
//                         payload stripe
//   send[slot]       (SLOTS * slot_bytes)          staged by the local GPU kernel
//   ctrl             (CTRL_BYTES)                  {u32 seq, u32 nbytes, u32 error,
//                                                   u32 missing_peer} doorbell; the
//                                                  last two are set by the kernel
//                                                  when a wait times out; then the
//                                                  per-slot op records and the
//                                                  two-shot words below
//
// The GPU kernel stages its input into send[seq & 1], publishes nbytes and seq
// in ctrl, then spins on flag[peer][seq & 1][hca] for every peer and HCA.  The
// proxy thread
// below spins on ctrl.seq and, for every peer, stripes the payload across every
// HCA.  Each stripe is followed by its own 4-byte seq write on the same reliable
// QP, so its flag cannot become visible before its payload.  The GPU waits for
// every stripe flag before consuming the receive slot.  Nothing on the receive
// path involves the host.
//
// Messages wide enough for the wire to dominate take the pipelined two-shot
// path instead (ctrl kind[slot] = 1; every one-shot and all-gather kernel of this
// package writes kind[slot] = 0).  The region then also holds
//
//   pipe_send          (pipe_bytes)                 staged input; our reduced shard
//   pipe_recv[src][2]  (world * 2 * shard_cap)      peers' reduce-scatter and
//                                                   all-gather chunks
//   pipe_flag[2][src][chunk][hca]                   seq after each chunk stripe
//
// Shard j covers packs [j * P / world, (j + 1) * P / world) of a P-pack message and
// is split into chunks of chunk_packs.  The kernel publishes ready_rs[c] = seq
// once every block staged chunk c of every peer's shard and ready_ag[c] = seq once
// chunk c of its own shard is reduced; the proxy posts each chunk as its word
// turns, so staging, the wire and the reduction overlap.  The two-shot buffers
// need one copy only: a rank leaves op N after every peer's all-gather of N
// arrived, which every peer sent after reducing all of our N chunks, and a peer
// sends its N + 1 all-gather only after our N + 1 reduce-scatter, which we post
// after our N kernel finished reading.  pipe_send is the exception: our own
// all-gather of N reads our reduced shard from it, and that send is not part of
// what lets our kernel N finish.  So once every write of op N completed, this
// thread sets ctrl pipe_done = seq, and the kernel publishes its epoch only after
// seeing it; otherwise our next two-shot op could stage over the shard while it
// was still to be sent, and peers would gather that input as our reduced values.
//
// ds41: the ready words, pipe flags and pipe_done compare EQUAL to seq.  Until the
// 32-bit sequence wraps, every stale value is an earlier seq or 0 and cannot equal
// the current one; after a wrap it could.  So a two-shot op arriving after the
// sequence wrapped (2^32 collectives on one runtime) fails closed; one-shot ops are
// unaffected.  (flashinfer#6915 instead stops every op at INT32_MAX.)
//
// This file is compiled by b12x.comm.roce_twoshot._proxy at first use with the host
// gcc and libibverbs; it must stay plain C with no CUDA dependency.
//
// ds41 idle parking (DSV41_ROCE_IDLE_PARK=1, adapter/roce_idle_park.py): built with
// -DROCE_IDLE_PARK=1, the proxy blocks on a futex while the engine is idle instead of
// nanosleep-polling (see _roce_idle_park.h).  Every addition is inside
// "#if defined(ROCE_IDLE_PARK) && ROCE_IDLE_PARK"; without the define the compiled
// code is the v2.6 proxy's.

#define _GNU_SOURCE
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(ROCE_IDLE_PARK) && ROCE_IDLE_PARK
#include "_roce_idle_park.h"
static int park_tick(void *arg);
#endif

#define ROCE_MAX_PEERS 16
#define ROCE_MAX_HCAS 2
#define ROCE_SLOTS 2
#define ROCE_FLAG_STRIDE 128
#define ROCE_PORT 1
#define ROCE_SEND_DEPTH 256
// 3 is the stock RoCEnante proxy; 103 marks this two-shot copy (a different
// region layout and doorbell record), so neither package can load the other's.
#define ROCE_ABI_VERSION 103
#define ROCE_MAX_CHUNKS 64
#define ROCE_CTRL_BYTES 4096
// ctrl words (u32): the one-shot record, then per-slot op descriptors and the
// two-shot chunk-ready words.
#define CTRL_SEQ 0
#define CTRL_ERROR 2
#define CTRL_SLOT_NBYTES 4
#define CTRL_SLOT_KIND 8
#define CTRL_SLOT_CHUNK_PACKS 10
#define CTRL_SLOT_CHUNKS 12
#define CTRL_PIPE_DONE 48
#define CTRL_READY_RS 64
#define CTRL_READY_AG 128
#define KIND_ONESHOT 0
#define KIND_TWOSHOT 1
// Model graphs leave sub-millisecond gaps between collectives.  Keep the
// proxy hot across those gaps; sleeping there adds one scheduler wakeup to
// every collective on the graph's critical path.
#define ROCE_IDLE_SPINS 20000000

typedef struct {
    uint64_t region_addr;
    uint32_t rkey[ROCE_MAX_HCAS];
    uint16_t lid[ROCE_MAX_HCAS];
    uint8_t gid[ROCE_MAX_HCAS][16];
    uint32_t mtu[ROCE_MAX_HCAS];
    uint32_t qp_num[ROCE_MAX_HCAS][ROCE_MAX_PEERS];
} roce_blob_t;

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp[ROCE_MAX_PEERS];
    uint32_t outstanding[ROCE_MAX_PEERS];
    uint64_t writes_completed;
    uint64_t bytes_posted;
    union ibv_gid gid;
    uint16_t lid;
    enum ibv_mtu mtu;
} roce_hca_t;

typedef struct {
    int world;
    int rank;
    int n_hca;
    int gid_index;
    roce_hca_t hca[ROCE_MAX_HCAS];
    uint8_t *region;
    size_t region_bytes;
    size_t slot_bytes;
    size_t recv_off;
    size_t flag_off;
    size_t send_off;
    size_t ctrl_off;
    size_t pipe_bytes;
    size_t pipe_send_off;
    size_t pipe_recv_off;
    size_t pipe_flag_off;
    size_t shard_cap;
    uint32_t last_pipe_seq;
    int seq_wrapped;
    int started;
    uint64_t peer_addr[ROCE_MAX_PEERS];
    uint32_t peer_rkey[ROCE_MAX_HCAS][ROCE_MAX_PEERS];
    pthread_t thread;
    atomic_int running;
    atomic_int failed;
    uint32_t last_seq;
    uint64_t ops_posted;
    uint64_t writes_completed;
    char err[512];
#if defined(ROCE_IDLE_PARK) && ROCE_IDLE_PARK
    roce_park_t park;
#endif
} roce_ctx_t;

void roce_destroy(roce_ctx_t *c);

static void set_err(roce_ctx_t *c, const char *what, int e) {
    snprintf(c->err, sizeof(c->err), "%s: %s", what, e ? strerror(e) : "failed");
}

int roce_abi_version(void) { return ROCE_ABI_VERSION; }

int roce_layout(int world, uint64_t slot_bytes, uint64_t pipe_bytes, uint64_t *out) {
    // out = {recv_off, flag_off, send_off, ctrl_off, total_bytes, flag_stride, slots,
    //        pipe_send_off, pipe_recv_off, pipe_flag_off, shard_cap, max_chunks}
    if (world < 2 || world > ROCE_MAX_PEERS || slot_bytes == 0 || (slot_bytes % 4096) != 0 ||
        (pipe_bytes % 4096) != 0) {
        return -1;
    }
    // Reject a layout whose arithmetic would wrap; the caller sizes slots from
    // configuration, so a wrapped region must fail here rather than at the NIC.
    uint64_t recv_bytes, flag_bytes, send_bytes, send_off, ctrl_off, pipe_send_off;
    if (slot_bytes > ((uint64_t)1 << 40) || pipe_bytes > ((uint64_t)1 << 32) ||
        __builtin_mul_overflow((uint64_t)world * ROCE_SLOTS, slot_bytes, &recv_bytes) ||
        __builtin_mul_overflow(
            (uint64_t)world * ROCE_SLOTS * ROCE_MAX_HCAS,
            (uint64_t)ROCE_FLAG_STRIDE,
            &flag_bytes) ||
        __builtin_mul_overflow((uint64_t)ROCE_SLOTS, slot_bytes, &send_bytes) ||
        __builtin_add_overflow(recv_bytes, flag_bytes, &send_off) ||
        __builtin_add_overflow(send_off, send_bytes, &ctrl_off) ||
        __builtin_add_overflow(ctrl_off, (uint64_t)ROCE_CTRL_BYTES, &pipe_send_off)) {
        return -1;
    }
    // A shard holds at most ceil(packs / world) packs.
    uint64_t shard_cap = (pipe_bytes / 16 + (uint64_t)world - 1) / (uint64_t)world * 16;
    shard_cap = (shard_cap + 4095) / 4096 * 4096;
    uint64_t pipe_recv_off = pipe_send_off + pipe_bytes;
    uint64_t pipe_flag_off = pipe_recv_off + (uint64_t)world * 2 * shard_cap;
    uint64_t pipe_flag_bytes =
        pipe_bytes == 0
            ? 0
            : (uint64_t)2 * world * ROCE_MAX_CHUNKS * ROCE_MAX_HCAS * ROCE_FLAG_STRIDE;
    uint64_t total = pipe_flag_off + pipe_flag_bytes;
    uint64_t recv_off = 0;
    uint64_t flag_off = recv_off + recv_bytes;
    out[0] = recv_off;
    out[1] = flag_off;
    out[2] = send_off;
    out[3] = ctrl_off;
    out[4] = total;
    out[5] = ROCE_FLAG_STRIDE;
    out[6] = ROCE_SLOTS;
    out[7] = pipe_send_off;
    out[8] = pipe_recv_off;
    out[9] = pipe_flag_off;
    out[10] = shard_cap;
    out[11] = ROCE_MAX_CHUNKS;
    return 0;
}

uint64_t roce_blob_bytes(void) { return sizeof(roce_blob_t); }

static int open_hca(roce_ctx_t *c, int h, const char *name) {
    int num = 0;
    struct ibv_device **list = ibv_get_device_list(&num);
    if (list == NULL) {
        set_err(c, "ibv_get_device_list", errno);
        return -1;
    }
    struct ibv_device *dev = NULL;
    for (int i = 0; i < num; i++) {
        if (strcmp(ibv_get_device_name(list[i]), name) == 0) {
            dev = list[i];
            break;
        }
    }
    if (dev == NULL) {
        ibv_free_device_list(list);
        snprintf(c->err, sizeof(c->err), "RDMA device %s not found", name);
        return -1;
    }
    roce_hca_t *hca = &c->hca[h];
    hca->ctx = ibv_open_device(dev);
    ibv_free_device_list(list);
    if (hca->ctx == NULL) {
        set_err(c, "ibv_open_device", errno);
        return -1;
    }
    struct ibv_port_attr port;
    if (ibv_query_port(hca->ctx, ROCE_PORT, &port) != 0) {
        set_err(c, "ibv_query_port", errno);
        return -1;
    }
    if (port.state != IBV_PORT_ACTIVE) {
        snprintf(c->err, sizeof(c->err), "RDMA device %s port %d is not active", name, ROCE_PORT);
        return -1;
    }
    hca->lid = port.lid;
    hca->mtu = port.active_mtu;
    if (ibv_query_gid(hca->ctx, ROCE_PORT, c->gid_index, &hca->gid) != 0) {
        set_err(c, "ibv_query_gid", errno);
        return -1;
    }
    hca->pd = ibv_alloc_pd(hca->ctx);
    if (hca->pd == NULL) {
        set_err(c, "ibv_alloc_pd", errno);
        return -1;
    }
    hca->mr = ibv_reg_mr(hca->pd, c->region, c->region_bytes,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (hca->mr == NULL) {
        set_err(c, "ibv_reg_mr(pinned region)", errno);
        return -1;
    }
    hca->cq = ibv_create_cq(hca->ctx, ROCE_SEND_DEPTH * ROCE_MAX_PEERS, NULL, NULL, 0);
    if (hca->cq == NULL) {
        set_err(c, "ibv_create_cq", errno);
        return -1;
    }
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        struct ibv_qp_init_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.send_cq = hca->cq;
        attr.recv_cq = hca->cq;
        attr.qp_type = IBV_QPT_RC;
        attr.cap.max_send_wr = ROCE_SEND_DEPTH;
        attr.cap.max_recv_wr = 1;
        attr.cap.max_send_sge = 1;
        attr.cap.max_recv_sge = 1;
        attr.cap.max_inline_data = 16;
        hca->qp[p] = ibv_create_qp(hca->pd, &attr);
        if (hca->qp[p] == NULL) {
            set_err(c, "ibv_create_qp", errno);
            return -1;
        }
        struct ibv_qp_attr init;
        memset(&init, 0, sizeof(init));
        init.qp_state = IBV_QPS_INIT;
        init.pkey_index = 0;
        init.port_num = ROCE_PORT;
        init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        int rc = ibv_modify_qp(hca->qp[p], &init,
                               IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
        if (rc != 0) {
            set_err(c, "ibv_modify_qp(INIT)", rc);
            return -1;
        }
    }
    return 0;
}

roce_ctx_t *roce_create(int world, int rank, const char *const *hca_names, int n_hca,
                        int gid_index, void *region, uint64_t region_bytes,
                        uint64_t slot_bytes, uint64_t pipe_bytes, char *err,
                        uint64_t err_len) {
    uint64_t layout[12];
    if (roce_layout(world, slot_bytes, pipe_bytes, layout) != 0 || layout[4] > region_bytes ||
        rank < 0 || rank >= world || n_hca < 1 || n_hca > ROCE_MAX_HCAS) {
        snprintf(err, err_len, "invalid roce runtime geometry");
        return NULL;
    }
    roce_ctx_t *c = calloc(1, sizeof(*c));
    if (c == NULL) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    c->world = world;
    c->rank = rank;
    c->n_hca = n_hca;
    c->gid_index = gid_index;
    c->region = region;
    c->region_bytes = region_bytes;
    c->slot_bytes = slot_bytes;
    c->recv_off = layout[0];
    c->flag_off = layout[1];
    c->send_off = layout[2];
    c->ctrl_off = layout[3];
    c->pipe_send_off = layout[7];
    c->pipe_recv_off = layout[8];
    c->pipe_flag_off = layout[9];
    c->shard_cap = layout[10];
    c->pipe_bytes = pipe_bytes;
    for (int h = 0; h < n_hca; h++) {
        if (open_hca(c, h, hca_names[h]) != 0) {
            snprintf(err, err_len, "%s", c->err);
            roce_destroy(c);
            return NULL;
        }
    }
    return c;
}

int roce_local_blob(roce_ctx_t *c, void *out, uint64_t out_len) {
    if (out_len < sizeof(roce_blob_t)) {
        return -1;
    }
    roce_blob_t blob;
    memset(&blob, 0, sizeof(blob));
    blob.region_addr = (uint64_t)(uintptr_t)c->region;
    for (int h = 0; h < c->n_hca; h++) {
        blob.rkey[h] = c->hca[h].mr->rkey;
        blob.lid[h] = c->hca[h].lid;
        blob.mtu[h] = (uint32_t)c->hca[h].mtu;
        memcpy(blob.gid[h], c->hca[h].gid.raw, 16);
        for (int p = 0; p < c->world; p++) {
            blob.qp_num[h][p] = (p == c->rank) ? 0 : c->hca[h].qp[p]->qp_num;
        }
    }
    memcpy(out, &blob, sizeof(blob));
    return 0;
}

static int connect_qp(roce_ctx_t *c, int h, int p, const roce_blob_t *peer) {
    roce_hca_t *hca = &c->hca[h];
    struct ibv_qp_attr rtr;
    memset(&rtr, 0, sizeof(rtr));
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = (enum ibv_mtu)(peer->mtu[h] < (uint32_t)hca->mtu ? peer->mtu[h] : (uint32_t)hca->mtu);
    rtr.dest_qp_num = peer->qp_num[h][c->rank];
    rtr.rq_psn = 0;
    rtr.max_dest_rd_atomic = 1;
    rtr.min_rnr_timer = 12;
    rtr.ah_attr.is_global = 1;
    rtr.ah_attr.dlid = peer->lid[h];
    rtr.ah_attr.sl = 0;
    rtr.ah_attr.src_path_bits = 0;
    rtr.ah_attr.port_num = ROCE_PORT;
    memcpy(rtr.ah_attr.grh.dgid.raw, peer->gid[h], 16);
    rtr.ah_attr.grh.sgid_index = (uint8_t)c->gid_index;
    rtr.ah_attr.grh.hop_limit = 64;
    rtr.ah_attr.grh.traffic_class = 0;
    rtr.ah_attr.grh.flow_label = 0;
    int rc = ibv_modify_qp(hca->qp[p], &rtr,
                           IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                               IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc != 0) {
        set_err(c, "ibv_modify_qp(RTR)", rc);
        return -1;
    }
    struct ibv_qp_attr rts;
    memset(&rts, 0, sizeof(rts));
    rts.qp_state = IBV_QPS_RTS;
    rts.timeout = 14;
    rts.retry_cnt = 7;
    rts.rnr_retry = 7;
    rts.sq_psn = 0;
    rts.max_rd_atomic = 1;
    rc = ibv_modify_qp(hca->qp[p], &rts,
                       IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                           IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc != 0) {
        set_err(c, "ibv_modify_qp(RTS)", rc);
        return -1;
    }
    return 0;
}

int roce_connect(roce_ctx_t *c, const void *blobs, uint64_t blobs_len) {
    if (blobs_len < sizeof(roce_blob_t) * (uint64_t)c->world) {
        snprintf(c->err, sizeof(c->err), "peer blob buffer too small");
        return -1;
    }
    const roce_blob_t *all = (const roce_blob_t *)blobs;
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        c->peer_addr[p] = all[p].region_addr;
        for (int h = 0; h < c->n_hca; h++) {
            c->peer_rkey[h][p] = all[p].rkey[h];
            if (connect_qp(c, h, p, &all[p]) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

static int drain_cq(roce_ctx_t *c, int h) {
    struct ibv_wc wc[32];
    int n = ibv_poll_cq(c->hca[h].cq, 32, wc);
    if (n < 0) {
        set_err(c, "ibv_poll_cq", errno);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            snprintf(c->err, sizeof(c->err),
                     "RDMA write to rank %u failed: %s (vendor_err 0x%x, seq %u)",
                     (unsigned)wc[i].wr_id, ibv_wc_status_str(wc[i].status),
                     wc[i].vendor_err, c->last_seq);
            return -1;
        }
        c->hca[h].outstanding[wc[i].wr_id] -= 1;
        c->hca[h].writes_completed += 1;
        c->writes_completed += 1;
    }
    return 0;
}

static int drain_all(roce_ctx_t *c) {
    for (int h = 0; h < c->n_hca; h++) {
        if (drain_cq(c, h) != 0) {
            return -1;
        }
    }
    return 0;
}

// Write `packs` 16-byte packs from local `send` to peer p at region offset
// `remote_off`, striped across every HCA, each stripe followed on its QP by a
// 4-byte seq write to `flag_off + h * ROCE_FLAG_STRIDE`.  An empty payload
// still writes every flag, so receivers wait the same way for every chunk.
// The work requests are the stock one-shot's (data then inline signaled flag,
// a quarter of the send queue at most per peer and HCA).
static int post_striped(roce_ctx_t *c, int p, const uint8_t *send, uint64_t remote_off,
                        uint32_t packs, uint64_t flag_off, uint32_t seq) {
    uint32_t seq_copy = seq;
    uint32_t pack_offset = 0;
    uint64_t remote = c->peer_addr[p];
    for (int h = 0; h < c->n_hca; h++) {
        roce_hca_t *hca = &c->hca[h];
        // Two work requests per stripe; keep the queue at most a quarter
        // full so a provider that needs extra entries can never fail a post.
        while (hca->outstanding[p] >= ROCE_SEND_DEPTH / 4) {
            if (drain_cq(c, h) != 0) {
                return -1;
            }
            // A peer that stopped acknowledging keeps the QP retrying for
            // a long time; honour a stop request instead of blocking
            // roce_stop (and so teardown) behind it.
            if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
                snprintf(c->err, sizeof(c->err),
                         "RoCE proxy stopped with %u writes outstanding to rank %d",
                         hca->outstanding[p], p);
                return -1;
            }
        }
        uint32_t stripe_packs = packs / (uint32_t)c->n_hca;
        if ((uint32_t)h < packs % (uint32_t)c->n_hca) {
            stripe_packs += 1;
        }
        uint32_t stripe_bytes = stripe_packs * 16;
        uint64_t byte_offset = (uint64_t)pack_offset * 16;
        struct ibv_sge flag_sge = {
            .addr = (uint64_t)(uintptr_t)&seq_copy,
            .length = 4,
            .lkey = 0,
        };
        struct ibv_send_wr flag_wr;
        memset(&flag_wr, 0, sizeof(flag_wr));
        flag_wr.wr_id = (uint64_t)p;
        flag_wr.sg_list = &flag_sge;
        flag_wr.num_sge = 1;
        flag_wr.opcode = IBV_WR_RDMA_WRITE;
        flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        flag_wr.wr.rdma.remote_addr = remote + flag_off + (uint64_t)h * ROCE_FLAG_STRIDE;
        flag_wr.wr.rdma.rkey = c->peer_rkey[h][p];

        struct ibv_send_wr data_wr;
        struct ibv_sge data_sge;
        struct ibv_send_wr *first_wr = &flag_wr;
        if (stripe_bytes != 0) {
            data_sge = (struct ibv_sge){
                .addr = (uint64_t)(uintptr_t)(send + byte_offset),
                .length = stripe_bytes,
                .lkey = hca->mr->lkey,
            };
            memset(&data_wr, 0, sizeof(data_wr));
            data_wr.wr_id = (uint64_t)p;
            data_wr.next = &flag_wr;
            data_wr.sg_list = &data_sge;
            data_wr.num_sge = 1;
            data_wr.opcode = IBV_WR_RDMA_WRITE;
            data_wr.wr.rdma.remote_addr = remote + remote_off + byte_offset;
            data_wr.wr.rdma.rkey = c->peer_rkey[h][p];
            first_wr = &data_wr;
        }
        struct ibv_send_wr *bad = NULL;
        int rc = ibv_post_send(hca->qp[p], first_wr, &bad);
        if (rc != 0) {
            set_err(c, "ibv_post_send", rc);
            return -1;
        }
        hca->outstanding[p] += 1;
        hca->bytes_posted += stripe_bytes;
        pack_offset += stripe_packs;
    }
    return 0;
}

// Post one-shot op seq to every peer: the whole payload of send[seq & 1], one
// stripe plus its flag write per HCA (identical to the stock proxy's post_op).
static int post_op(roce_ctx_t *c, uint32_t seq, uint32_t nbytes) {
    if (nbytes == 0 || nbytes % 16 != 0 || nbytes > c->slot_bytes) {
        snprintf(c->err, sizeof(c->err),
                 "RoCE payload size must be a positive multiple of 16 bytes within the slot, "
                 "got %u",
                 nbytes);
        return -1;
    }
    uint32_t slot = seq & 1u;
    const uint8_t *send = c->region + c->send_off + (size_t)slot * c->slot_bytes;
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        uint64_t remote_off =
            c->recv_off + ((uint64_t)c->rank * ROCE_SLOTS + slot) * c->slot_bytes;
        uint64_t flag_off = c->flag_off + ((uint64_t)c->rank * ROCE_SLOTS + slot) *
                                              (uint64_t)c->n_hca * ROCE_FLAG_STRIDE;
        if (post_striped(c, p, send, remote_off, nbytes / 16, flag_off, seq) != 0) {
            return -1;
        }
    }
    c->ops_posted += 1;
    return drain_all(c);
}

// First pack of shard j of a total-pack message (the kernel uses the same split).
static inline uint32_t shard_start(const roce_ctx_t *c, uint32_t total, int j) {
    return (uint32_t)(((uint64_t)j * total) / (uint64_t)c->world);
}

// Post chunk `chunk` of two-shot phase `phase` (0 reduce-scatter: peer p's shard
// of our staged input; 1 all-gather: our reduced shard) to every peer.
static int post_pipe_chunk(roce_ctx_t *c, uint32_t seq, int phase, uint32_t chunk,
                           uint32_t total, uint32_t chunk_packs) {
    const uint8_t *send = c->region + c->pipe_send_off;
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        int shard = phase == 0 ? p : c->rank;
        uint32_t first = shard_start(c, total, shard);
        uint32_t end = shard_start(c, total, shard + 1);
        uint64_t lo = (uint64_t)first + (uint64_t)chunk * chunk_packs;
        uint64_t hi = lo + chunk_packs;
        if (lo > end) {
            lo = end;
        }
        if (hi > end) {
            hi = end;
        }
        uint64_t remote_off = c->pipe_recv_off +
                              ((uint64_t)c->rank * 2 + (uint64_t)phase) * c->shard_cap +
                              (lo - first) * 16;
        uint64_t flag_off =
            c->pipe_flag_off +
            ((((uint64_t)phase * c->world + c->rank) * ROCE_MAX_CHUNKS + chunk) *
             (uint64_t)c->n_hca) *
                ROCE_FLAG_STRIDE;
        if (post_striped(c, p, send + lo * 16, remote_off, (uint32_t)(hi - lo), flag_off,
                         seq) != 0) {
            return -1;
        }
    }
    return 0;
}

static uint32_t outstanding_writes(const roce_ctx_t *c) {
    uint32_t pending = 0;
    for (int h = 0; h < c->n_hca; h++) {
        for (int p = 0; p < c->world; p++) {
            pending += c->hca[h].outstanding[p];
        }
    }
    return pending;
}

// Run two-shot op seq: post each reduce-scatter chunk once the kernel staged it
// and each all-gather chunk once the kernel reduced it, then release pipe_send.
static int run_pipe_op(roce_ctx_t *c, uint32_t seq, uint32_t nbytes, uint32_t chunk_packs,
                       uint32_t chunks) {
    volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
    uint32_t total = nbytes / 16;
    uint32_t widest = (total + (uint32_t)c->world - 1) / (uint32_t)c->world;
    if (c->pipe_bytes == 0 || nbytes == 0 || nbytes % 16 != 0 || nbytes > c->pipe_bytes ||
        chunks == 0 || chunks > ROCE_MAX_CHUNKS || chunk_packs == 0 ||
        (uint64_t)chunks * chunk_packs < widest) {
        snprintf(c->err, sizeof(c->err),
                 "invalid two-shot op %u: %u bytes in %u chunks of %u packs (capacity %zu)",
                 seq, nbytes, chunks, chunk_packs, c->pipe_bytes);
        return -1;
    }
    // Ready words, flags and pipe_done compare equal to seq: exact until the
    // sequence wraps (see the header), refused after.
    if (c->seq_wrapped) {
        snprintf(c->err, sizeof(c->err),
                 "two-shot op %u after the 32-bit sequence wrapped; recreate the transport",
                 seq);
        return -1;
    }
    uint32_t rs = 0;
    uint32_t ag = 0;
    uint64_t idle = 0;
    while (rs < chunks || ag < chunks) {
        int posted = 0;
        if (rs < chunks && __atomic_load_n(&ctrl[CTRL_READY_RS + rs], __ATOMIC_ACQUIRE) == seq) {
            if (post_pipe_chunk(c, seq, 0, rs, total, chunk_packs) != 0) {
                return -1;
            }
            rs++;
            posted = 1;
        }
        if (ag < chunks && __atomic_load_n(&ctrl[CTRL_READY_AG + ag], __ATOMIC_ACQUIRE) == seq) {
            if (post_pipe_chunk(c, seq, 1, ag, total, chunk_packs) != 0) {
                return -1;
            }
            ag++;
            posted = 1;
        }
        if (posted) {
            idle = 0;
            continue;
        }
        if (++idle % 64 == 0) {
            if (drain_all(c) != 0) {
                return -1;
            }
            // A kernel whose peer wait timed out publishes no further chunks.
            if (ctrl[CTRL_ERROR] != 0) {
                snprintf(c->err, sizeof(c->err),
                         "two-shot op %u abandoned: kernel timed out at seq %u", seq,
                         ctrl[CTRL_ERROR]);
                return -1;
            }
            if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
                snprintf(c->err, sizeof(c->err), "RoCE proxy stopped inside two-shot op %u",
                         seq);
                return -1;
            }
        }
    }
    c->ops_posted += 1;
    // Release pipe_send: wait until every posted write, our all-gather included,
    // completed, so the NIC no longer reads the shard the next op will overwrite.
    for (;;) {
        if (drain_all(c) != 0) {
            return -1;
        }
        if (outstanding_writes(c) == 0) {
            break;
        }
        if (ctrl[CTRL_ERROR] != 0) {
            snprintf(c->err, sizeof(c->err),
                     "two-shot op %u abandoned: kernel timed out at seq %u", seq,
                     ctrl[CTRL_ERROR]);
            return -1;
        }
        if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
            snprintf(c->err, sizeof(c->err), "RoCE proxy stopped releasing two-shot op %u",
                     seq);
            return -1;
        }
    }
    c->last_pipe_seq = seq;
    __atomic_store_n(&ctrl[CTRL_PIPE_DONE], seq, __ATOMIC_RELEASE);
    return 0;
}

static void *proxy_main(void *arg) {
    roce_ctx_t *c = (roce_ctx_t *)arg;
    volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
#if defined(ROCE_IDLE_PARK) && ROCE_IDLE_PARK
    atomic_store(&c->park.tid, (int32_t)syscall(SYS_gettid));
    pthread_setname_np(pthread_self(), "roce-proxy");
#endif
    // Spin while ops are flowing.  After ROCE_IDLE_SPINS polls without a
    // doorbell, request a short nanosleep between polls (the OS decides the
    // actual delay) so an idle runtime does not hold a core next to the
    // serving process.  The missed-doorbell catch-up below keeps the protocol
    // correct however long the thread is away.
    uint64_t idle = 0;
    const struct timespec nap = {0, 20000};
    while (atomic_load_explicit(&c->running, memory_order_relaxed)) {
        uint32_t seq = __atomic_load_n(&ctrl[CTRL_SEQ], __ATOMIC_ACQUIRE);
        if (seq == c->last_seq) {
            idle++;
            if (idle % 64 == 0) {
                if (drain_all(c) != 0) {
                    atomic_store(&c->failed, 1);
                    return NULL;
                }
#if defined(ROCE_IDLE_PARK) && ROCE_IDLE_PARK
                // The engine said idle and no doorbell for after_ms: block until
                // it wakes us (never with writes in flight; retried in 64 polls).
                if (roce_park_due(&c->park, idle) && outstanding_writes(c) == 0) {
                    if (roce_park_block(&c->park, &ctrl[CTRL_SEQ], c->last_seq, &c->running,
                                        park_tick, c) < 0) {
                        atomic_store(&c->failed, 1);
                        return NULL;
                    }
                    idle = 0;
                    continue;
                }
#endif
            }
            if (idle >= ROCE_IDLE_SPINS) {
                nanosleep(&nap, NULL);
            }
            continue;
        }
        idle = 0;
#if defined(ROCE_IDLE_PARK) && ROCE_IDLE_PARK
        roce_park_on_doorbell(&c->park);
#endif
        // The doorbell holds only the newest sequence.  Our kernel for op N
        // completes on the peers' payloads alone, so op N+1 can ring before
        // this thread has seen op N (it slept, or the scheduler moved it).
        // Peers cannot get further than one op ahead of us, so at most
        // ROCE_SLOTS doorbells are pending and every send slot is intact:
        // post each missed sequence in order using its per-slot byte count.
        uint32_t pending = seq - c->last_seq;
        if (pending > ROCE_SLOTS) {
            snprintf(c->err, sizeof(c->err),
                     "doorbell skipped %u ops (last %u, now %u)", pending, c->last_seq, seq);
            atomic_store(&c->failed, 1);
            return NULL;
        }
        for (uint32_t s = c->last_seq + 1; pending > 0; s++, pending--) {
            uint32_t slot = s & 1u;
            if (s == 0) {
                c->seq_wrapped = 1;
            }
            uint32_t nbytes = ctrl[CTRL_SLOT_NBYTES + slot];
            uint32_t kind = ctrl[CTRL_SLOT_KIND + slot];
            int rc;
            if (kind == KIND_ONESHOT) {
                rc = post_op(c, s, nbytes);
            } else if (kind == KIND_TWOSHOT) {
                rc = run_pipe_op(c, s, nbytes, ctrl[CTRL_SLOT_CHUNK_PACKS + slot],
                                 ctrl[CTRL_SLOT_CHUNKS + slot]);
            } else {
                snprintf(c->err, sizeof(c->err), "unknown op kind %u at seq %u", kind, s);
                rc = -1;
            }
            if (rc != 0) {
                atomic_store(&c->failed, 1);
                return NULL;
            }
            c->last_seq = s;
        }
    }
    return NULL;
}

int roce_start(roce_ctx_t *c) {
    if (atomic_load(&c->running)) {
        return 0;
    }
    if (!c->started) {
        // A restart continues from the last posted sequence so ops that rang
        // the doorbell while the thread was stopped are still posted.
        volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
        c->last_seq = ctrl[CTRL_SEQ];
        c->started = 1;
    }
    atomic_store(&c->failed, 0);
    atomic_store(&c->running, 1);
    int rc = pthread_create(&c->thread, NULL, proxy_main, c);
    if (rc != 0) {
        atomic_store(&c->running, 0);
        set_err(c, "pthread_create", rc);
        return -1;
    }
    return 0;
}

void roce_stop(roce_ctx_t *c) {
    if (atomic_exchange(&c->running, 0)) {
#if defined(ROCE_IDLE_PARK) && ROCE_IDLE_PARK
        roce_park_kick(&c->park);
#endif
        pthread_join(c->thread, NULL);
    }
}

int roce_failed(roce_ctx_t *c) { return atomic_load(&c->failed); }

const char *roce_error(roce_ctx_t *c) { return c->err; }

uint64_t roce_stat(roce_ctx_t *c, int which) {
    switch (which) {
    case 0:
        return c->ops_posted;
    case 1:
        return c->writes_completed;
    case 2:
        return c->last_seq;
    case 3:
        return c->last_pipe_seq;
    default:
        return 0;
    }
}

uint64_t roce_hca_stat(roce_ctx_t *c, int hca, int which) {
    if (hca < 0 || hca >= c->n_hca) {
        return -1;
    }
    switch (which) {
    case 0:
        return c->hca[hca].writes_completed;
    case 1:
        return c->hca[hca].bytes_posted;
    default:
        return 0;
    }
}

void roce_destroy(roce_ctx_t *c) {
    if (c == NULL) {
        return;
    }
    roce_stop(c);
    for (int h = 0; h < ROCE_MAX_HCAS; h++) {
        roce_hca_t *hca = &c->hca[h];
        for (int p = 0; p < ROCE_MAX_PEERS; p++) {
            if (hca->qp[p] != NULL) {
                ibv_destroy_qp(hca->qp[p]);
            }
        }
        if (hca->cq != NULL) {
            ibv_destroy_cq(hca->cq);
        }
        if (hca->mr != NULL) {
            ibv_dereg_mr(hca->mr);
        }
        if (hca->pd != NULL) {
            ibv_dealloc_pd(hca->pd);
        }
        if (hca->ctx != NULL) {
            ibv_close_device(hca->ctx);
        }
    }
    free(c);
}

#if defined(ROCE_IDLE_PARK) && ROCE_IDLE_PARK
// Idle parking API (adapter/roce_idle_park.py through _proxy.py).  Engine side; any
// thread.  after_ms == 0 keeps the proxy from ever parking (the default).
static int park_tick(void *arg) { return drain_all((roce_ctx_t *)arg); }

int roce_park_abi(void) { return ROCE_PARK_ABI; }

int roce_park_configure(roce_ctx_t *c, uint32_t after_ms, uint32_t poll_ms) {
    if (poll_ms == 0 || poll_ms > 1000 || after_ms > 3600000) {
        return -1;
    }
    roce_park_set(&c->park, after_ms, poll_ms);
    return 0;
}

void roce_park_allow(roce_ctx_t *c) { roce_park_allow_now(&c->park); }

void roce_park_wake(roce_ctx_t *c) { roce_park_wake_now(&c->park); }

uint64_t roce_park_stat(roce_ctx_t *c, int which) { return roce_park_stat_of(&c->park, which); }
#endif
