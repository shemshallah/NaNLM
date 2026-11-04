#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "nanlm_vec.h"

static int failures = 0;

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);             \
            fprintf(stderr, __VA_ARGS__);                                    \
            fprintf(stderr, "\n");                                           \
            failures++;                                                      \
        }                                                                    \
    } while (0)

static uint32_t bswap32(uint32_t x) {
    return ((x & 0x000000FFu) << 24) |
           ((x & 0x0000FF00u) << 8) |
           ((x & 0x00FF0000u) >> 8) |
           ((x & 0xFF000000u) >> 24);
}

static void vec_copy(nanlm_vec *dst, const nanlm_vec *src) {
    memcpy(dst->w, src->w, src->words * sizeof(nan16_word));
}

static void flip_bit(nanlm_vec *v, uint32_t bit) {
    uint32_t wi = bit / 32u;
    uint32_t bi = bit % 32u;
    uint32_t p = nan16_payload(v->w[wi]);
    v->w[wi] = nan16_make(p ^ (1u << bi), NAN16_N_DATA);
}

static int same_vec(const nanlm_vec *a, const nanlm_vec *b) {
    return a->words == b->words &&
           memcmp(a->w, b->w, a->words * sizeof(nan16_word)) == 0;
}

int main(void) {
    enum { WORDS = 8 };
    uint64_t seed = 0x123456789ABCDEF0ull;
    nanlm_vec *a = nanlm_vec_new(WORDS);
    nanlm_vec *b = nanlm_vec_new(WORDS);
    nanlm_vec *c = nanlm_vec_new(WORDS);
    nanlm_vec *d = nanlm_vec_new(WORDS);
    nanlm_vec *e = nanlm_vec_new(WORDS);
    size_t i;
    unsigned n;

    CHECK(a && b && c && d && e, "allocation");
    if (!a || !b || !c || !d || !e) return 1;

    nanlm_vec_random(a, &seed);
    nanlm_vec_random(b, &seed);
    nanlm_vec_random(c, &seed);

    CHECK(nanlm_vec_is_data(a), "random a is not neutral data");
    CHECK(nanlm_vec_is_data(b), "random b is not neutral data");
    CHECK(nanlm_vec_is_data(c), "random c is not neutral data");

    /* bool against a scalar reference, for every truth table. */
    for (n = 0; n < 16; n++) {
        int rc = nanlm_vec_bool(d, a, b, n);
        CHECK(rc == 0, "vec_bool n=%u rc=%d", n, rc);
        for (i = 0; i < WORDS; i++) {
            uint32_t want = nan16_bool(n, nan16_payload(a->w[i]),
                                      nan16_payload(b->w[i]));
            CHECK(nan16_payload(d->w[i]) == want, "vec_bool n=%u i=%zu", n, i);
            CHECK(nan16_n(d->w[i]) == NAN16_N_DATA,
                  "vec_bool n=%u i=%zu not neutral", n, i);
        }
    }

    /* bind is XOR (n=6). */
    CHECK(nanlm_vec_bind(d, a, b) == 0, "bind rc");
    for (i = 0; i < WORDS; i++) {
        uint32_t want = nan16_payload(a->w[i]) ^ nan16_payload(b->w[i]);
        CHECK(nan16_payload(d->w[i]) == want, "bind i=%zu", i);
    }

    /* majority of three against a scalar reference. */
    CHECK(nanlm_vec_bundle3(d, a, b, c) == 0, "bundle3 rc");
    for (i = 0; i < WORDS; i++) {
        uint32_t x = nan16_payload(a->w[i]);
        uint32_t y = nan16_payload(b->w[i]);
        uint32_t z = nan16_payload(c->w[i]);
        uint32_t want = (x & y) | (x & z) | (y & z);
        CHECK(nan16_payload(d->w[i]) == want, "bundle3 i=%zu", i);
    }

    /* Hamming distance and the XOR cancellation property. */
    CHECK(nanlm_vec_hamming(a, a) == 0, "hamming self");
    CHECK(nanlm_vec_hamming(a, b) > 0, "hamming distinct");
    CHECK(nanlm_vec_bind(d, a, c) == 0, "bind a,c");
    CHECK(nanlm_vec_bind(e, b, c) == 0, "bind b,c");
    CHECK(nanlm_vec_hamming(d, e) == nanlm_vec_hamming(a, b),
          "XOR cancellation");

    /* BSWAPM reverses payload bytes, preserves neutral nibbles, is involutive. */
    CHECK(nanlm_vec_bswap(d, a) == 0, "bswap rc");
    for (i = 0; i < WORDS; i++) {
        CHECK(nan16_n(d->w[i]) == NAN16_N_DATA, "bswap i=%zu not neutral", i);
        CHECK(nan16_payload(d->w[i]) ==
                  bswap32(nan16_payload(a->w[i])),
              "bswap i=%zu payload", i);
    }
    CHECK(nanlm_vec_bswap(e, d) == 0, "bswap twice rc");
    CHECK(same_vec(e, a), "bswap twice differs");

    /* Popcount of the vector equals the sum of word popcounts. */
    {
        uint64_t sum = 0;
        for (i = 0; i < WORDS; i++) {
            sum += nan16_popcount32(nan16_payload(a->w[i]));
        }
        CHECK(nanlm_vec_popcount(a) == sum, "vec popcount");
    }

    /* A tiny nearest-prototype classifier: 4 classes, 128 bits, noise 2 bits. */
    {
        enum { D = 128, CWORDS = D / 32, CLASSES = 4, SAMPLES = 6 };
        nanlm_vec *proto[CLASSES] = {0};
        nanlm_vec *sample[CLASSES][SAMPLES] = {{0}};
        nanlm_vec *q = nanlm_vec_new(CWORDS);
        int ci, si, ok = 1;
        uint64_t s = 0xDEADBEEFCAFEF00Dull;

        CHECK(q != NULL, "classifier q allocation");
        for (ci = 0; ci < CLASSES; ci++) {
            proto[ci] = nanlm_vec_new(CWORDS);
            CHECK(proto[ci] != NULL, "proto allocation");
            if (!proto[ci]) { ok = 0; break; }
            nanlm_vec_random(proto[ci], &s);
            for (si = 0; si < SAMPLES; si++) {
                sample[ci][si] = nanlm_vec_new(CWORDS);
                CHECK(sample[ci][si] != NULL, "sample allocation");
                if (!sample[ci][si]) { ok = 0; break; }
                vec_copy(sample[ci][si], proto[ci]);
                flip_bit(sample[ci][si], (uint32_t)(s % D));
                flip_bit(sample[ci][si], (uint32_t)((s >> 8) % D));
            }
        }

        if (ok) {
            for (ci = 0; ci < CLASSES; ci++) {
                for (si = 0; si < SAMPLES; si++) {
                    uint64_t best = UINT64_MAX;
                    int best_c = -1;
                    for (int k = 0; k < CLASSES; k++) {
                        uint64_t h = nanlm_vec_hamming(sample[ci][si], proto[k]);
                        CHECK(h != UINT64_MAX, "classifier hamming");
                        if (h < best) { best = h; best_c = k; }
                    }
                    CHECK(best_c == ci, "classifier ci=%d si=%d got=%d", ci, si,
                          best_c);
                }
            }
        }

        if (q) nanlm_vec_free(q);
        for (ci = 0; ci < CLASSES; ci++) {
            if (proto[ci]) nanlm_vec_free(proto[ci]);
            for (si = 0; si < SAMPLES; si++) {
                if (sample[ci][si]) nanlm_vec_free(sample[ci][si]);
            }
        }
    }

    nanlm_vec_free(a);
    nanlm_vec_free(b);
    nanlm_vec_free(c);
    nanlm_vec_free(d);
    nanlm_vec_free(e);

    if (failures) {
        fprintf(stderr, "test_vec: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_vec: PASS\n");
    return 0;
}
