// Idle parking for the RoCE proxy thread (ds41, DSV41_ROCE_IDLE_PARK=1).
//
// Included by _roce_proxy.c only when it is compiled with -DROCE_IDLE_PARK=1 (the
// adapter sets that before the proxy library is first built).  Without the define
// this file is not part of the build and the proxy is the v2.6 one, instruction for
// instruction (tests/test_roce_idle_park.py compares the .text sections).
//
// Today an idle proxy spins ROCE_IDLE_SPINS polls after its last doorbell and then
// polls once per nanosleep(20 us): about 10^4 wakeups a second per rank for as long
// as the engine is idle.  Here the engine tells the proxy when it is idle and when
// work is coming, and the proxy blocks on a futex in between:
//
//   engine thread                               proxy thread
//   on idle:   roce_park_allow  -> allow = 1     idle branch, every 64 polls:
//                                                  allow == 1 for after_ms with no
//                                                  doorbell -> roce_park_block
//   before a forward: roce_park_wake             roce_park_block:
//              allow = 0, gen += 1,                g = gen; allow == 0 -> woken
//              FUTEX_WAKE(gen)                     doorbell rang -> leave
//                                                  FUTEX_WAIT(gen, g, poll_ms)
//
// No lost wakeup: the proxy reads gen BEFORE it checks allow, the engine clears allow
// BEFORE it bumps gen.  If the bump precedes the proxy's read of gen, the proxy also
// sees allow == 0 and does not wait; if it follows, FUTEX_WAIT finds gen != g and
// returns at once (EAGAIN).
//
// The engine never relies on the proxy for safety: the GPU kernel spins on the
// peers' flags in pinned memory and the proxy's missed-doorbell catch-up posts every
// pending op in order however long the thread was away (at most ROCE_SLOTS can be
// pending, see proxy_main).  So a wake that never comes costs latency, never
// correctness, and the poll_ms backstop bounds that latency: a parked proxy re-checks
// the doorbell every poll_ms and leaves when it rang (counted as a late doorbell).
// The kernels' own peer timeout is about 20 s (DEFAULT_SPIN_LIMIT), three orders of
// magnitude above the backstop.
//
// Linux only (futex(2)); plain C, no CUDA.

#ifndef ROCE_IDLE_PARK_H
#define ROCE_IDLE_PARK_H

#include <errno.h>
#include <limits.h>
#include <linux/futex.h>
#include <stdatomic.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define ROCE_PARK_ABI 1

// roce_park_block results
#define ROCE_PARK_WOKEN 1     // the engine cleared allow (normal wake)
#define ROCE_PARK_DOORBELL 2  // a doorbell rang and no wake came during the park (late doorbell)
#define ROCE_PARK_STOPPED 3   // roce_stop

// roce_park_stat indices
#define ROCE_PARK_STAT_PARKS 0          // times the proxy blocked
#define ROCE_PARK_STAT_WAKES 1          // parks ended by the engine's wake
#define ROCE_PARK_STAT_TIMEOUTS 2       // backstop polls while parked
#define ROCE_PARK_STAT_LATE 3           // parks ended by a doorbell, not a wake
#define ROCE_PARK_STAT_PARKED_NS 4      // total time parked
#define ROCE_PARK_STAT_LAST_WAKE_NS 5   // engine wake -> proxy running, last park
#define ROCE_PARK_STAT_MAX_WAKE_NS 6    // the same, maximum
#define ROCE_PARK_STAT_LAST_LEAD_NS 7   // engine wake -> first doorbell after it
#define ROCE_PARK_STAT_TID 8            // the proxy thread's tid (0 before it ran)
#define ROCE_PARK_STAT_PARKED 9         // 1 while parked
#define ROCE_PARK_STAT_ALLOW 10         // 1 while the engine says idle
#define ROCE_PARK_STAT_AFTER_NS 11
#define ROCE_PARK_STAT_POLL_NS 12

typedef struct {
    // Three groups, 64 bytes of padding apart so the engine's writes, the proxy's
    // private clock and the statistics never share a cache line (the context comes
    // from calloc, so the padding, not _Alignas, does the separating).
    // shared: written by the engine thread, read by the proxy
    _Atomic uint32_t allow;
    _Atomic uint32_t gen;  // the futex word
    _Atomic uint64_t wake_ns;
    _Atomic uint64_t after_ns;  // 0 = never park (the build default until configured)
    _Atomic uint64_t poll_ns;
    char pad0[64];
    // proxy thread only
    uint64_t since_ns;
    int lead_pending;
    char pad1[64];
    // statistics: written by the proxy, read by anyone
    _Atomic uint64_t parks;
    _Atomic uint64_t wakes;
    _Atomic uint64_t timeouts;
    _Atomic uint64_t late;
    _Atomic uint64_t parked_ns;
    _Atomic uint64_t last_wake_ns;
    _Atomic uint64_t max_wake_ns;
    _Atomic uint64_t last_lead_ns;
    _Atomic uint32_t parked;
    _Atomic int32_t tid;
} roce_park_t;

static inline uint64_t roce_park_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// 0 = woken or value changed (or a signal); 1 = timed out.
static inline int roce_park_futex_wait(_Atomic uint32_t *word, uint32_t expected,
                                       uint64_t timeout_ns) {
    struct timespec ts = {(time_t)(timeout_ns / 1000000000ull),
                          (long)(timeout_ns % 1000000000ull)};
    long rc = syscall(SYS_futex, (uint32_t *)word, FUTEX_WAIT_PRIVATE, expected, &ts, NULL, 0);
    return rc == -1 && errno == ETIMEDOUT;
}

static inline void roce_park_futex_wake(_Atomic uint32_t *word) {
    syscall(SYS_futex, (uint32_t *)word, FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0);
}

// Engine side, any thread.  after_ms == 0 disables parking.
static inline void roce_park_set(roce_park_t *p, uint32_t after_ms, uint32_t poll_ms) {
    atomic_store(&p->poll_ns, (uint64_t)(poll_ms ? poll_ms : 1) * 1000000ull);
    atomic_store(&p->after_ns, (uint64_t)after_ms * 1000000ull);
}

static inline void roce_park_allow_now(roce_park_t *p) { atomic_store(&p->allow, 1); }

// Wake a parked proxy without touching allow (roce_stop: the loop sees running == 0).
static inline void roce_park_kick(roce_park_t *p) {
    atomic_fetch_add(&p->gen, 1);
    roce_park_futex_wake(&p->gen);
}

static inline void roce_park_wake_now(roce_park_t *p) {
    atomic_store(&p->allow, 0);
    atomic_store(&p->wake_ns, roce_park_now_ns());
    atomic_fetch_add(&p->gen, 1);
    roce_park_futex_wake(&p->gen);
}

// Proxy side.  Called from the idle branch every 64 polls; `idle` is the proxy's
// count of polls since its last doorbell (or park), so idle == 64 is the first call
// of a new quiet period and restarts the clock.  Nonzero when it is time to park.
// While allow == 0 (the engine is busy) this is one relaxed load and one store.
static inline int roce_park_due(roce_park_t *p, uint64_t idle) {
    uint32_t allow = atomic_load_explicit(&p->allow, memory_order_relaxed);
    if (idle == 64 || !allow) {
        p->since_ns = 0;
    }
    if (!allow) {
        return 0;
    }
    uint64_t after = atomic_load_explicit(&p->after_ns, memory_order_relaxed);
    if (after == 0) {
        return 0;
    }
    uint64_t now = roce_park_now_ns();
    if (p->since_ns == 0) {
        p->since_ns = now;
        return 0;
    }
    return now - p->since_ns >= after;
}

// Proxy side: block until the engine wakes us, a doorbell rings, or roce_stop.
// `tick` runs on every backstop timeout (the proxy drains its completion queues
// there); a nonzero return aborts with -1.
typedef int (*roce_park_tick_fn)(void *arg);

static int roce_park_block(roce_park_t *p, const volatile uint32_t *doorbell, uint32_t last_seq,
                           atomic_int *running, roce_park_tick_fn tick, void *arg) {
    uint64_t t0 = roce_park_now_ns();
    atomic_fetch_add(&p->parks, 1);
    atomic_store(&p->parked, 1);
    // Any wake bumps gen.  A doorbell is "late" only if no wake came while we were
    // parked: the engine may wake us, ring, finish a one-op forward (it needs only the
    // peers' payloads) and mark idle again before this thread runs; that is a wake.
    uint32_t g0 = atomic_load(&p->gen);
    int why;
    for (;;) {
        uint32_t g = atomic_load(&p->gen);
        if (!atomic_load(&p->allow)) {
            why = ROCE_PARK_WOKEN;
            break;
        }
        if (__atomic_load_n(doorbell, __ATOMIC_ACQUIRE) != last_seq) {
            why = atomic_load(&p->gen) != g0 ? ROCE_PARK_WOKEN : ROCE_PARK_DOORBELL;
            break;
        }
        if (!atomic_load_explicit(running, memory_order_relaxed)) {
            why = ROCE_PARK_STOPPED;
            break;
        }
        if (roce_park_futex_wait(&p->gen, g, atomic_load(&p->poll_ns))) {
            atomic_fetch_add(&p->timeouts, 1);
            if (tick != NULL && tick(arg) != 0) {
                atomic_store(&p->parked, 0);
                return -1;
            }
        }
    }
    uint64_t t1 = roce_park_now_ns();
    atomic_store(&p->parked, 0);
    atomic_fetch_add(&p->parked_ns, t1 - t0);
    p->since_ns = 0;
    if (why == ROCE_PARK_WOKEN) {
        atomic_fetch_add(&p->wakes, 1);
        uint64_t w = atomic_load(&p->wake_ns);
        uint64_t lat = (w != 0 && t1 > w) ? t1 - w : 0;
        atomic_store(&p->last_wake_ns, lat);
        if (lat > atomic_load(&p->max_wake_ns)) {
            atomic_store(&p->max_wake_ns, lat);
        }
        p->lead_pending = 1;
    } else if (why == ROCE_PARK_DOORBELL) {
        atomic_fetch_add(&p->late, 1);
    }
    return why;
}

// Proxy side, on a doorbell: the first one after a wake records the wake's lead
// (engine wake -> first collective).  One predictable branch otherwise.
static inline void roce_park_on_doorbell(roce_park_t *p) {
    if (__builtin_expect(p->lead_pending, 0)) {
        p->lead_pending = 0;
        uint64_t w = atomic_load(&p->wake_ns);
        uint64_t now = roce_park_now_ns();
        atomic_store(&p->last_lead_ns, (w != 0 && now > w) ? now - w : 0);
    }
}

static inline uint64_t roce_park_stat_of(roce_park_t *p, int which) {
    switch (which) {
    case ROCE_PARK_STAT_PARKS:
        return atomic_load(&p->parks);
    case ROCE_PARK_STAT_WAKES:
        return atomic_load(&p->wakes);
    case ROCE_PARK_STAT_TIMEOUTS:
        return atomic_load(&p->timeouts);
    case ROCE_PARK_STAT_LATE:
        return atomic_load(&p->late);
    case ROCE_PARK_STAT_PARKED_NS:
        return atomic_load(&p->parked_ns);
    case ROCE_PARK_STAT_LAST_WAKE_NS:
        return atomic_load(&p->last_wake_ns);
    case ROCE_PARK_STAT_MAX_WAKE_NS:
        return atomic_load(&p->max_wake_ns);
    case ROCE_PARK_STAT_LAST_LEAD_NS:
        return atomic_load(&p->last_lead_ns);
    case ROCE_PARK_STAT_TID:
        return (uint64_t)atomic_load(&p->tid);
    case ROCE_PARK_STAT_PARKED:
        return atomic_load(&p->parked);
    case ROCE_PARK_STAT_ALLOW:
        return atomic_load(&p->allow);
    case ROCE_PARK_STAT_AFTER_NS:
        return atomic_load(&p->after_ns);
    case ROCE_PARK_STAT_POLL_NS:
        return atomic_load(&p->poll_ns);
    default:
        return 0;
    }
}

#endif  // ROCE_IDLE_PARK_H
