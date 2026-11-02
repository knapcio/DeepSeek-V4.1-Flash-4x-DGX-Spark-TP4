// Park / wake race stress on the REAL proxy loop: _roce_proxy.c (roce_twoshot) is compiled into
// this file, with or without -DROCE_IDLE_PARK=1, and W proxy threads run proxy_main over an
// in-process fake RDMA (a post copies the payload, then release-stores the flag, into the peer's
// region; signaled flags complete on the poster's CQ at once).  Each rank also has a driver thread
// that plays its engine and its GPU kernel:
//
//   burst b: all drivers meet at a barrier, each sleeps its own random 0..delay_us (so the ranks
//   wake and ring in a different order every burst; an early rank's kernel waits on peers whose
//   proxies may still be parked), calls roce_park_wake, runs the burst's one-shot ops exactly as
//   the kernel does (stage send[seq & 1], nbytes / kind per slot, release-store the doorbell, spin
//   on every peer's flag of every HCA, check every received byte), then roce_park_allow and an
//   idle gap of 0..idle_ms (parks happen in the gaps longer than after_ms).
//
// Modes:  nofault   every wake is called; late doorbells must be 0, every first op after a park
//                   must finish far below the 1 s backstop (a lost wake would take a full poll)
//         fault     rank 1 skips every third wake: its proxy must be found by the backstop
//                   (late doorbells > 0) and every op must still complete and check out
//         (no ROCE_IDLE_PARK) the same harness on the v2.6 proxy: the baseline
// A peer flag not seen within 10 s is a deadlock and fails the run.  At the end every proxy is
// parked and roce_stop must return promptly.
//
//   gcc -O2 -std=gnu11 -Wall -pthread [-DROCE_IDLE_PARK=1] -I<roce_twoshot dir>
//       park_stress.c -libverbs -lm -o park_stress
//   ./park_stress MODE BURSTS     prints one "STRESS ... PASS" line, exit 0
#include "_roce_proxy.c"

#include <math.h>

#define W 4
#define NHCA 2
#define SLOT (16 * 1024)
#define CQ_RING 8192
#define FLAG_TIMEOUT_NS 10000000000ull

#if defined(ROCE_IDLE_PARK) && ROCE_IDLE_PARK
#define PARK 1
#else
#define PARK 0
#endif

typedef struct {
    struct ibv_cq cq;
    struct ibv_wc ring[CQ_RING];
    uint32_t head, tail;
} fake_cq_t;

typedef struct {
    struct ibv_qp qp;
} fake_qp_t;

static int fake_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad) {
    (void)bad;
    fake_cq_t *cq = (fake_cq_t *)qp->send_cq;
    for (; wr != NULL; wr = wr->next) {
        uint8_t *dst = (uint8_t *)(uintptr_t)wr->wr.rdma.remote_addr;
        const uint8_t *src = (const uint8_t *)(uintptr_t)wr->sg_list[0].addr;
        uint32_t len = wr->sg_list[0].length;
        if (len == 4) {  // the seq flag after the stripe: visible only after the payload
            uint32_t v;
            memcpy(&v, src, 4);
            __atomic_store_n((uint32_t *)dst, v, __ATOMIC_RELEASE);
        } else {
            memcpy(dst, src, len);
        }
        if (wr->send_flags & IBV_SEND_SIGNALED) {
            if (cq->tail - cq->head >= CQ_RING) {
                fprintf(stderr, "fake CQ overflow\n");
                abort();
            }
            struct ibv_wc *wc = &cq->ring[cq->tail % CQ_RING];
            memset(wc, 0, sizeof(*wc));
            wc->wr_id = wr->wr_id;
            wc->status = IBV_WC_SUCCESS;
            cq->tail++;
        }
    }
    return 0;
}

static int fake_poll_cq(struct ibv_cq *ibcq, int n, struct ibv_wc *wc) {
    fake_cq_t *cq = (fake_cq_t *)ibcq;
    int got = 0;
    while (got < n && cq->head != cq->tail) {
        wc[got++] = cq->ring[cq->head % CQ_RING];
        cq->head++;
    }
    return got;
}

static struct ibv_context g_vctx;
static struct ibv_mr g_mr;
static fake_cq_t g_cq[W][NHCA];
static fake_qp_t g_qp[W][NHCA][W];
static roce_ctx_t *g_c[W];

static pthread_barrier_t g_barrier;
static int g_mode_fault;
static int g_bursts;
static atomic_int g_errors;
static atomic_ulong g_ops;
static atomic_ulong g_cross_parked;  // first ops rung while some peer's proxy was parked
static uint64_t g_first_op_ns[W][4096];  // per burst: this rank's first op (ring -> all flags)
#if PARK
static uint64_t g_parks_start[W][4096];  // this rank's park count before its wake / at its allow
static uint64_t g_parks_end[W][4096];
#endif

static uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

static uint32_t nbytes_of(uint32_t seq) { return 16u * (1u + mix(seq) % (SLOT / 16)); }

static uint32_t word_of(int rank, uint32_t seq, uint32_t i) {
    return mix(((uint32_t)rank << 28) ^ (seq * 2654435761u) ^ i);
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void setup_rank(int r, uint8_t **regions, uint64_t *layout) {
    roce_ctx_t *c = calloc(1, sizeof(*c));
    c->world = W;
    c->rank = r;
    c->n_hca = NHCA;
    c->region = regions[r];
    c->region_bytes = layout[4];
    c->slot_bytes = SLOT;
    c->recv_off = layout[0];
    c->flag_off = layout[1];
    c->send_off = layout[2];
    c->ctrl_off = layout[3];
    c->pipe_send_off = layout[7];
    c->pipe_recv_off = layout[8];
    c->pipe_flag_off = layout[9];
    c->shard_cap = layout[10];
    c->pipe_bytes = 0;
    for (int h = 0; h < NHCA; h++) {
        g_cq[r][h].cq.context = &g_vctx;
        c->hca[h].cq = &g_cq[r][h].cq;
        c->hca[h].mr = &g_mr;
        for (int p = 0; p < W; p++) {
            if (p == r) {
                continue;
            }
            g_qp[r][h][p].qp.context = &g_vctx;
            g_qp[r][h][p].qp.send_cq = &g_cq[r][h].cq;
            c->hca[h].qp[p] = &g_qp[r][h][p].qp;
        }
    }
    for (int p = 0; p < W; p++) {
        c->peer_addr[p] = (uint64_t)(uintptr_t)regions[p];
    }
    g_c[r] = c;
}

// One one-shot all-reduce as the kernel runs it (minus the reduction): stage, ring, wait, check.
static int run_op(int r, uint32_t seq) {
    roce_ctx_t *c = g_c[r];
    volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
    uint32_t slot = seq & 1u;
    uint32_t nbytes = nbytes_of(seq);
    uint32_t *send = (uint32_t *)(c->region + c->send_off + (size_t)slot * SLOT);
    for (uint32_t i = 0; i < nbytes / 4; i++) {
        send[i] = word_of(r, seq, i);
    }
    ctrl[CTRL_SLOT_NBYTES + slot] = nbytes;
    ctrl[CTRL_SLOT_KIND + slot] = KIND_ONESHOT;
    __atomic_store_n(&ctrl[CTRL_SEQ], seq, __ATOMIC_RELEASE);
    uint64_t t0 = now_ns();
    for (int p = 0; p < W; p++) {
        if (p == r) {
            continue;
        }
        for (int h = 0; h < NHCA; h++) {
            const uint32_t *flag =
                (const uint32_t *)(c->region + c->flag_off +
                                   (((uint64_t)p * ROCE_SLOTS + slot) * NHCA + h) * ROCE_FLAG_STRIDE);
            while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) != seq) {
                if (now_ns() - t0 > FLAG_TIMEOUT_NS || roce_failed(c)) {
                    fprintf(stderr, "rank %d seq %u: no flag from rank %d hca %d (deadlock?) %s\n", r,
                            seq, p, h, roce_failed(c) ? roce_error(c) : "");
                    return -1;
                }
            }
        }
        const uint32_t *recv =
            (const uint32_t *)(c->region + c->recv_off + ((uint64_t)p * ROCE_SLOTS + slot) * SLOT);
        for (uint32_t i = 0; i < nbytes / 4; i++) {
            if (recv[i] != word_of(p, seq, i)) {
                fprintf(stderr, "rank %d seq %u: byte mismatch from rank %d at word %u\n", r, seq, p, i);
                return -1;
            }
        }
    }
    atomic_fetch_add(&g_ops, 1);
    return 0;
}

static void *driver_main(void *arg) {
    int r = (int)(intptr_t)arg;
    uint32_t seq = 0;
    for (int b = 0; b < g_bursts; b++) {
        pthread_barrier_wait(&g_barrier);
        unsigned seed = (unsigned)(b * 7919 + r * 104729 + 17);
        uint32_t delay_us = (uint32_t)(rand_r(&seed) % 1500);
        uint32_t ops = 1 + mix((uint32_t)b) % 6;           // the same on every rank
        uint32_t idle_us = mix((uint32_t)b * 31u + 7u) % 6000;  // the same on every rank
        struct timespec d = {0, (long)delay_us * 1000};
        nanosleep(&d, NULL);
#if PARK
        if (b < 4096) {
            g_parks_start[r][b] = roce_park_stat(g_c[r], ROCE_PARK_STAT_PARKS);
        }
        int skip = g_mode_fault && r == 1 && b % 3 == 0;
        if (!skip) {
            roce_park_wake(g_c[r]);
        }
        for (int p = 0; p < W; p++) {
            if (p != r && roce_park_stat(g_c[p], ROCE_PARK_STAT_PARKED)) {
                atomic_fetch_add(&g_cross_parked, 1);
                break;
            }
        }
#endif
        for (uint32_t k = 0; k < ops; k++) {
            uint64_t t0 = now_ns();
            if (run_op(r, ++seq) != 0) {
                atomic_fetch_add(&g_errors, 1);
                return NULL;
            }
            if (k == 0 && b < 4096) {
                g_first_op_ns[r][b] = now_ns() - t0;
            }
        }
#if PARK
        if (b < 4096) {
            g_parks_end[r][b] = roce_park_stat(g_c[r], ROCE_PARK_STAT_PARKS);
        }
        roce_park_allow(g_c[r]);
#endif
        struct timespec gap = {0, (long)idle_us * 1000};
        nanosleep(&gap, NULL);
    }
    return NULL;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "nofault";
    g_bursts = argc > 2 ? atoi(argv[2]) : 1500;
    if (g_bursts > 4096) {
        g_bursts = 4096;
    }
    g_mode_fault = strcmp(mode, "fault") == 0;
    g_vctx.ops.post_send = fake_post_send;
    g_vctx.ops.poll_cq = fake_poll_cq;
    uint64_t layout[12];
    if (roce_layout(W, SLOT, 0, layout) != 0) {
        fprintf(stderr, "layout failed\n");
        return 2;
    }
    uint8_t *regions[W];
    for (int r = 0; r < W; r++) {
        regions[r] = aligned_alloc(4096, (layout[4] + 4095) / 4096 * 4096);
        memset(regions[r], 0, layout[4]);
    }
    for (int r = 0; r < W; r++) {
        setup_rank(r, regions, layout);
    }
    for (int r = 0; r < W; r++) {
        if (roce_start(g_c[r]) != 0) {
            fprintf(stderr, "roce_start failed\n");
            return 2;
        }
#if PARK
        // after 2 ms of engine idle; backstop 1 s (nofault: a lost wake would show) or 3 ms (fault)
        if (roce_park_configure(g_c[r], 2, g_mode_fault ? 3 : 1000) != 0) {
            fprintf(stderr, "configure failed\n");
            return 2;
        }
#endif
    }
    pthread_barrier_init(&g_barrier, NULL, W);
    pthread_t drv[W];
    uint64_t t0 = now_ns();
    for (int r = 0; r < W; r++) {
        pthread_create(&drv[r], NULL, driver_main, (void *)(intptr_t)r);
    }
    for (int r = 0; r < W; r++) {
        pthread_join(drv[r], NULL);
    }
    double wall = (now_ns() - t0) / 1e9;
    int fail = atomic_load(&g_errors) != 0;
    for (int r = 0; r < W; r++) {
        if (roce_failed(g_c[r])) {
            fprintf(stderr, "rank %d proxy failed: %s\n", r, roce_error(g_c[r]));
            fail = 1;
        }
    }
    // first-op latency of bursts whose proxy had parked before them (rank-local)
    static uint64_t lat[W * 4096];
    int nlat = 0;
    uint64_t parks[W] = {0}, wakes[W] = {0}, late[W] = {0}, timeouts[W] = {0}, maxwake[W] = {0};
#if PARK
    for (int r = 0; r < W; r++) {
        for (int b = 1; b < g_bursts && b < 4096; b++) {
            if (g_parks_start[r][b] > g_parks_end[r][b - 1]) {  // parked in the gap before b
                lat[nlat++] = g_first_op_ns[r][b];
            }
        }
    }
    // stop with every proxy parked: roce_stop must return promptly
    for (int r = 0; r < W; r++) {
        roce_park_allow(g_c[r]);
    }
    struct timespec settle = {0, 50 * 1000000};
    nanosleep(&settle, NULL);
    int all_parked = 1;
    for (int r = 0; r < W; r++) {
        all_parked &= (int)roce_park_stat(g_c[r], ROCE_PARK_STAT_PARKED);
    }
    uint64_t s0 = now_ns();
    for (int r = 0; r < W; r++) {
        roce_stop(g_c[r]);
    }
    double stop_ms = (now_ns() - s0) / 1e6;
    for (int r = 0; r < W; r++) {
        parks[r] = roce_park_stat(g_c[r], ROCE_PARK_STAT_PARKS);
        wakes[r] = roce_park_stat(g_c[r], ROCE_PARK_STAT_WAKES);
        late[r] = roce_park_stat(g_c[r], ROCE_PARK_STAT_LATE);
        timeouts[r] = roce_park_stat(g_c[r], ROCE_PARK_STAT_TIMEOUTS);
        maxwake[r] = roce_park_stat(g_c[r], ROCE_PARK_STAT_MAX_WAKE_NS);
        if (parks[r] == 0) {
            fprintf(stderr, "rank %d never parked\n", r);
            fail = 1;
        }
    }
    if (!all_parked || stop_ms > 500) {
        fprintf(stderr, "stop with parked proxies: all_parked=%d stop %.1f ms\n", all_parked, stop_ms);
        fail = 1;
    }
    if (atomic_load(&g_cross_parked) == 0) {
        fprintf(stderr, "no burst started while a peer proxy was parked (scenario not exercised)\n");
        fail = 1;
    }
    if (!g_mode_fault) {
        for (int r = 0; r < W; r++) {
            if (late[r] != 0) {
                fprintf(stderr, "rank %d: %llu late doorbells with every wake called\n", r,
                        (unsigned long long)late[r]);
                fail = 1;
            }
        }
    } else if (late[1] == 0) {
        fprintf(stderr, "fault mode: rank 1 skipped wakes but the backstop never found a doorbell\n");
        fail = 1;
    }
#else
    for (int r = 0; r < W; r++) {
        roce_stop(g_c[r]);
    }
    for (int r = 0; r < W; r++) {
        for (int b = 0; b < g_bursts && b < 4096; b++) {
            lat[nlat++] = g_first_op_ns[r][b];
        }
    }
#endif
    uint64_t p50 = 0, p99 = 0, mx = 0;
    if (nlat > 0) {
        qsort(lat, (size_t)nlat, sizeof(lat[0]), cmp_u64);
        p50 = lat[nlat / 2];
        p99 = lat[(int)floor(nlat * 0.99)];
        mx = lat[nlat - 1];
    }
    // nofault: a lost wake would make some first op wait for the 1 s backstop
    if (PARK && !g_mode_fault && mx > 300000000ull) {
        fprintf(stderr, "first op after a park took %.1f ms (lost wake?)\n", mx / 1e6);
        fail = 1;
    }
    printf("STRESS mode=%s park=%d ranks=%d bursts=%d ops=%lu wall=%.1fs cross_parked=%lu "
           "first_op_after_park_us p50=%.1f p99=%.1f max=%.1f (n=%d)",
           PARK ? mode : "baseline", PARK, W, g_bursts, (unsigned long)atomic_load(&g_ops), wall,
           (unsigned long)atomic_load(&g_cross_parked), p50 / 1e3, p99 / 1e3, mx / 1e3, nlat);
    printf(" parks=[%llu %llu %llu %llu] wakes=[%llu %llu %llu %llu] late=[%llu %llu %llu %llu] "
           "timeouts=[%llu %llu %llu %llu] max_wake_us=[%.0f %.0f %.0f %.0f] %s\n",
           (unsigned long long)parks[0], (unsigned long long)parks[1], (unsigned long long)parks[2],
           (unsigned long long)parks[3], (unsigned long long)wakes[0], (unsigned long long)wakes[1],
           (unsigned long long)wakes[2], (unsigned long long)wakes[3], (unsigned long long)late[0],
           (unsigned long long)late[1], (unsigned long long)late[2], (unsigned long long)late[3],
           (unsigned long long)timeouts[0], (unsigned long long)timeouts[1],
           (unsigned long long)timeouts[2], (unsigned long long)timeouts[3], maxwake[0] / 1e3,
           maxwake[1] / 1e3, maxwake[2] / 1e3, maxwake[3] / 1e3, fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
