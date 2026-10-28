// Unit tests of the idle-park state machine (runtime/b12x/b12x/comm/roce_twoshot/_roce_idle_park.h)
// on its own: the due clock, every way a park ends, its counters, and a lost-wakeup race of
// many thousand allow/park/wake rounds.  Linux (futex), CPU only.
//
//   gcc -O2 -std=gnu11 -Wall -Wextra -Werror -pthread -I<roce_twoshot dir> park_unit.c -o park_unit
//   ./park_unit [race_rounds]        prints "UNIT <name> PASS" per test and "UNIT PASS <n> tests"
#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "_roce_idle_park.h"

static int g_tests = 0;
static int g_failed = 0;

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "  CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
            g_failed++;                                                                    \
            return;                                                                        \
        }                                                                                  \
    } while (0)

static void sleep_us(long us) {
    struct timespec ts = {us / 1000000, (us % 1000000) * 1000};
    nanosleep(&ts, NULL);
}

static roce_park_t *fresh(uint32_t after_ms, uint32_t poll_ms) {
    roce_park_t *p = calloc(1, sizeof(*p));
    roce_park_set(p, after_ms, poll_ms);
    return p;
}

// ---------------------------------------------------------------------------------------------
static void t_due_disabled(void) {
    roce_park_t *p = fresh(0, 20);
    roce_park_allow_now(p);
    for (uint64_t idle = 64; idle < 64 * 100; idle += 64) {
        CHECK(!roce_park_due(p, idle));
        sleep_us(50);
    }
    free(p);
}

static void t_due_needs_allow(void) {
    roce_park_t *p = fresh(1, 20);
    for (uint64_t idle = 64; idle < 64 * 40; idle += 64) {
        CHECK(!roce_park_due(p, idle));
        CHECK(p->since_ns == 0);
        sleep_us(200);
    }
    free(p);
}

static void t_due_after_quiet_period(void) {
    roce_park_t *p = fresh(5, 20);
    roce_park_allow_now(p);
    CHECK(!roce_park_due(p, 64));  // starts the clock
    CHECK(p->since_ns != 0);
    CHECK(!roce_park_due(p, 128));
    sleep_us(6000);
    CHECK(roce_park_due(p, 192));
    free(p);
}

static void t_due_restarts_after_doorbell(void) {
    roce_park_t *p = fresh(5, 20);
    roce_park_allow_now(p);
    CHECK(!roce_park_due(p, 64));
    sleep_us(6000);
    // a doorbell reset the proxy's idle count: idle == 64 again restarts the clock
    CHECK(!roce_park_due(p, 64));
    CHECK(!roce_park_due(p, 128));
    sleep_us(6000);
    CHECK(roce_park_due(p, 192));
    free(p);
}

static void t_due_cleared_by_wake(void) {
    roce_park_t *p = fresh(5, 20);
    roce_park_allow_now(p);
    CHECK(!roce_park_due(p, 64));
    sleep_us(6000);
    roce_park_wake_now(p);
    CHECK(!roce_park_due(p, 128));
    CHECK(p->since_ns == 0);
    roce_park_allow_now(p);
    CHECK(!roce_park_due(p, 192));  // the clock starts over
    free(p);
}

// ---------------------------------------------------------------------------------------------
typedef struct {
    roce_park_t *p;
    volatile uint32_t doorbell;
    uint32_t last_seq;
    atomic_int running;
    int result;
    uint64_t took_ns;
    int tick_calls;
    int tick_fail;
    atomic_int started;
} blocker_t;

static int tick(void *arg) {
    blocker_t *b = arg;
    b->tick_calls++;
    return b->tick_fail ? -1 : 0;
}

static void *blocker_main(void *arg) {
    blocker_t *b = arg;
    atomic_store(&b->started, 1);
    uint64_t t0 = roce_park_now_ns();
    b->result = roce_park_block(b->p, &b->doorbell, b->last_seq, &b->running, tick, b);
    b->took_ns = roce_park_now_ns() - t0;
    return NULL;
}

static void blocker_init(blocker_t *b, roce_park_t *p) {
    memset(b, 0, sizeof(*b));
    b->p = p;
    b->doorbell = 7;
    b->last_seq = 7;
    atomic_store(&b->running, 1);
}

static void wait_parked(roce_park_t *p) {
    for (int i = 0; i < 100000 && !atomic_load(&p->parked); i++) {
        sleep_us(10);
    }
}

static void t_block_woken(void) {
    roce_park_t *p = fresh(1, 1000);
    blocker_t b;
    blocker_init(&b, p);
    roce_park_allow_now(p);
    pthread_t th;
    pthread_create(&th, NULL, blocker_main, &b);
    wait_parked(p);
    sleep_us(10000);
    CHECK(atomic_load(&p->parked) == 1);
    roce_park_wake_now(p);
    pthread_join(th, NULL);
    CHECK(b.result == ROCE_PARK_WOKEN);
    CHECK(atomic_load(&p->parked) == 0);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_PARKS) == 1);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_WAKES) == 1);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_LATE) == 0);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_TIMEOUTS) == 0);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_PARKED_NS) >= 9000000ull);
    uint64_t w = roce_park_stat_of(p, ROCE_PARK_STAT_LAST_WAKE_NS);
    CHECK(w > 0 && w < 200000000ull);  // engine wake -> proxy running, well below the backstop
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_MAX_WAKE_NS) == w);
    CHECK(b.took_ns < 900000000ull);   // not the 1 s backstop
    CHECK(p->lead_pending == 1);
    sleep_us(1000);
    roce_park_on_doorbell(p);
    CHECK(p->lead_pending == 0);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_LAST_LEAD_NS) >= 1000000ull);
    free(p);
}

static void t_block_already_woken(void) {
    roce_park_t *p = fresh(1, 1000);
    blocker_t b;
    blocker_init(&b, p);
    roce_park_allow_now(p);
    roce_park_wake_now(p);  // the engine woke between the due check and the block
    blocker_main(&b);
    CHECK(b.result == ROCE_PARK_WOKEN);
    CHECK(b.took_ns < 50000000ull);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_TIMEOUTS) == 0);
    free(p);
}

static void t_block_late_doorbell_backstop(void) {
    roce_park_t *p = fresh(1, 3);
    blocker_t b;
    blocker_init(&b, p);
    roce_park_allow_now(p);
    pthread_t th;
    pthread_create(&th, NULL, blocker_main, &b);
    wait_parked(p);
    sleep_us(10000);
    __atomic_store_n(&b.doorbell, 8, __ATOMIC_RELEASE);  // a doorbell with no wake
    pthread_join(th, NULL);
    CHECK(b.result == ROCE_PARK_DOORBELL);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_LATE) == 1);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_WAKES) == 0);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_TIMEOUTS) >= 1);
    CHECK(b.tick_calls == (int)roce_park_stat_of(p, ROCE_PARK_STAT_TIMEOUTS));
    CHECK(p->lead_pending == 0);
    free(p);
}

static void t_block_doorbell_after_wake_is_a_wake(void) {
    roce_park_t *p = fresh(1, 1000);
    blocker_t b;
    blocker_init(&b, p);
    roce_park_allow_now(p);
    roce_park_wake_now(p);
    b.doorbell = 8;  // and the engine rang right after it
    blocker_main(&b);
    CHECK(b.result == ROCE_PARK_WOKEN);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_LATE) == 0);
    free(p);
}

static void t_block_wake_ring_reallow_is_a_wake(void) {
    // the engine wakes, rings, finishes a one-op forward and marks idle again before the parked
    // proxy runs: the proxy finds allow == 1 and a new doorbell; that is a wake, not a late doorbell
    roce_park_t *p = fresh(1, 1000);
    blocker_t b;
    blocker_init(&b, p);
    roce_park_allow_now(p);
    pthread_t th;
    pthread_create(&th, NULL, blocker_main, &b);
    wait_parked(p);
    atomic_store(&p->allow, 0);
    atomic_fetch_add(&p->gen, 1);  // the wake's gen bump, futex wake held back
    __atomic_store_n(&b.doorbell, 8, __ATOMIC_RELEASE);
    roce_park_allow_now(p);
    roce_park_futex_wake(&p->gen);
    pthread_join(th, NULL);
    CHECK(b.result == ROCE_PARK_WOKEN);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_LATE) == 0);
    CHECK(b.took_ns < 900000000ull);
    free(p);
}

static void t_block_stop(void) {
    roce_park_t *p = fresh(1, 1000);
    blocker_t b;
    blocker_init(&b, p);
    roce_park_allow_now(p);
    pthread_t th;
    pthread_create(&th, NULL, blocker_main, &b);
    wait_parked(p);
    uint64_t t0 = roce_park_now_ns();
    atomic_store(&b.running, 0);
    roce_park_kick(p);
    pthread_join(th, NULL);
    CHECK(b.result == ROCE_PARK_STOPPED);
    CHECK(roce_park_now_ns() - t0 < 500000000ull);
    CHECK(atomic_load(&p->allow) == 1);  // kick leaves the idle mark alone
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_WAKES) == 0);
    free(p);
}

static void t_block_tick_error(void) {
    roce_park_t *p = fresh(1, 2);
    blocker_t b;
    blocker_init(&b, p);
    b.tick_fail = 1;
    roce_park_allow_now(p);
    blocker_main(&b);
    CHECK(b.result == -1);
    CHECK(atomic_load(&p->parked) == 0);
    free(p);
}

static void t_stats_api(void) {
    roce_park_t *p = fresh(1500, 25);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_AFTER_NS) == 1500000000ull);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_POLL_NS) == 25000000ull);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_ALLOW) == 0);
    roce_park_allow_now(p);
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_ALLOW) == 1);
    CHECK(roce_park_stat_of(p, 99) == 0);
    roce_park_set(p, 10, 0);  // poll 0 is clamped to 1 ms
    CHECK(roce_park_stat_of(p, ROCE_PARK_STAT_POLL_NS) == 1000000ull);
    free(p);
}

// ---------------------------------------------------------------------------------------------
// Lost-wakeup race: the engine marks idle, the proxy blocks (backstop 5 s), the engine wakes after
// a random 0..~60 us, so the wake lands before, during and after the proxy's way into FUTEX_WAIT.
// A lost wakeup would hold a round for the 5 s backstop; every round must end well before.
typedef struct {
    roce_park_t *p;
    int rounds;
    atomic_int go;
    atomic_int done;
    uint64_t worst_ns;
    int bad;
    volatile uint32_t doorbell;
    atomic_int running;
} race_t;

static void *race_proxy(void *arg) {
    race_t *r = arg;
    for (int i = 1; i <= r->rounds; i++) {
        while (atomic_load(&r->go) < i) {
        }
        int why = roce_park_block(r->p, &r->doorbell, 0, &r->running, NULL, NULL);
        if (why != ROCE_PARK_WOKEN) {
            r->bad++;
        }
        atomic_store(&r->done, i);
    }
    return NULL;
}

static void t_race_lost_wakeup(int rounds) {
    race_t r;
    memset(&r, 0, sizeof(r));
    r.p = fresh(1, 5000);
    r.rounds = rounds;
    atomic_store(&r.running, 1);
    pthread_t th;
    pthread_create(&th, NULL, race_proxy, &r);
    unsigned seed = 12345;
    for (int i = 1; i <= rounds; i++) {
        roce_park_allow_now(r.p);
        atomic_store(&r.go, i);
        int spin = rand_r(&seed) % 4000;
        for (volatile int k = 0; k < spin; k++) {
        }
        uint64_t t0 = roce_park_now_ns();
        roce_park_wake_now(r.p);
        while (atomic_load(&r.done) < i) {
        }
        uint64_t dt = roce_park_now_ns() - t0;
        if (dt > r.worst_ns) {
            r.worst_ns = dt;
        }
    }
    pthread_join(th, NULL);
    printf("  race: %d rounds, worst wake -> proxy back %.1f us, parks %llu wakes %llu\n", rounds,
           r.worst_ns / 1e3, (unsigned long long)roce_park_stat_of(r.p, ROCE_PARK_STAT_PARKS),
           (unsigned long long)roce_park_stat_of(r.p, ROCE_PARK_STAT_WAKES));
    CHECK(r.bad == 0);
    CHECK(r.worst_ns < 1000000000ull);  // far below the 5 s backstop: no wakeup was lost
    CHECK(roce_park_stat_of(r.p, ROCE_PARK_STAT_TIMEOUTS) == 0);
    CHECK(roce_park_stat_of(r.p, ROCE_PARK_STAT_WAKES) == (uint64_t)rounds);
    free(r.p);
}

// ---------------------------------------------------------------------------------------------
#define RUN(fn, ...)                                                                       \
    do {                                                                                   \
        int before = g_failed;                                                             \
        g_tests++;                                                                         \
        fn(__VA_ARGS__);                                                                   \
        printf("UNIT %s %s\n", #fn, g_failed == before ? "PASS" : "FAIL");                \
    } while (0)

int main(int argc, char **argv) {
    int rounds = argc > 1 ? atoi(argv[1]) : 20000;
    RUN(t_due_disabled);
    RUN(t_due_needs_allow);
    RUN(t_due_after_quiet_period);
    RUN(t_due_restarts_after_doorbell);
    RUN(t_due_cleared_by_wake);
    RUN(t_block_woken);
    RUN(t_block_already_woken);
    RUN(t_block_late_doorbell_backstop);
    RUN(t_block_doorbell_after_wake_is_a_wake);
    RUN(t_block_wake_ring_reallow_is_a_wake);
    RUN(t_block_stop);
    RUN(t_block_tick_error);
    RUN(t_stats_api);
    RUN(t_race_lost_wakeup, rounds);
    if (g_failed) {
        printf("UNIT FAIL %d of %d tests\n", g_failed, g_tests);
        return 1;
    }
    printf("UNIT PASS %d tests\n", g_tests);
    return 0;
}
