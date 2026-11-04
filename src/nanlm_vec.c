#include "nanlm_vec.h"

#include <stdlib.h>

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

nanlm_vec *nanlm_vec_new(size_t words) {
    nanlm_vec *v;
    if (words == 0 || words > (SIZE_MAX / sizeof(nan16_word))) return NULL;
    v = (nanlm_vec *)calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->w = (nan16_word *)calloc(words, sizeof(nan16_word));
    if (!v->w) {
        free(v);
        return NULL;
    }
    v->words = words;
    return v;
}

void nanlm_vec_free(nanlm_vec *v) {
    if (!v) return;
    free(v->w);
    free(v);
}

int nanlm_vec_valid(const nanlm_vec *v) {
    size_t i;
    if (!v || !v->w || v->words == 0) return 0;
    for (i = 0; i < v->words; i++) {
        if (!nan16_gate(v->w[i])) return 0;
    }
    return 1;
}

int nanlm_vec_is_data(const nanlm_vec *v) {
    size_t i;
    if (!v || !v->w || v->words == 0) return 0;
    for (i = 0; i < v->words; i++) {
        if (!nan16_gate(v->w[i])) return 0;
        if (nan16_n(v->w[i]) != NAN16_N_DATA) return 0;
    }
    return 1;
}

void nanlm_vec_zero(nanlm_vec *v) {
    size_t i;
    if (!v || !v->w) return;
    for (i = 0; i < v->words; i++) {
        v->w[i] = nan16_make(0, NAN16_N_DATA);
    }
}

void nanlm_vec_fill(nanlm_vec *v, nan16_word word) {
    size_t i;
    if (!v || !v->w) return;
    for (i = 0; i < v->words; i++) v->w[i] = word;
}

void nanlm_vec_random(nanlm_vec *v, uint64_t *state) {
    size_t i;
    if (!v || !v->w || !state) return;
    for (i = 0; i < v->words; i++) {
        uint32_t payload = (uint32_t)splitmix64(state);
        v->w[i] = nan16_make(payload, NAN16_N_DATA);
    }
}

static int same_shape(const nanlm_vec *a, const nanlm_vec *b) {
    return a && b && a->w && b->w && a->words == b->words;
}

int nanlm_vec_bool(nanlm_vec *dst, const nanlm_vec *a, const nanlm_vec *b,
                   unsigned n) {
    size_t i;
    if (!dst || !dst->w || !same_shape(a, b) || dst->words != a->words) return -1;
    if (!nanlm_vec_valid(a) || !nanlm_vec_valid(b)) return -1;
    for (i = 0; i < dst->words; i++) {
        uint32_t x = nan16_payload(a->w[i]);
        uint32_t y = nan16_payload(b->w[i]);
        dst->w[i] = nan16_make(nan16_bool(n, x, y), NAN16_N_DATA);
    }
    return 0;
}

int nanlm_vec_bind(nanlm_vec *dst, const nanlm_vec *a, const nanlm_vec *b) {
    return nanlm_vec_bool(dst, a, b, 6u);
}

int nanlm_vec_bundle3(nanlm_vec *dst, const nanlm_vec *a, const nanlm_vec *b,
                      const nanlm_vec *c) {
    size_t i;
    if (!dst || !dst->w || !same_shape(a, b) || !same_shape(a, c) ||
        dst->words != a->words) {
        return -1;
    }
    if (!nanlm_vec_valid(a) || !nanlm_vec_valid(b) || !nanlm_vec_valid(c)) {
        return -1;
    }
    for (i = 0; i < dst->words; i++) {
        uint32_t x = nan16_payload(a->w[i]);
        uint32_t y = nan16_payload(b->w[i]);
        uint32_t z = nan16_payload(c->w[i]);
        uint32_t m = (x & y) | (x & z) | (y & z);
        dst->w[i] = nan16_make(m, NAN16_N_DATA);
    }
    return 0;
}

int nanlm_vec_bswap(nanlm_vec *dst, const nanlm_vec *a) {
    size_t i;
    if (!dst || !dst->w || !a || !a->w || dst->words != a->words) return -1;
    if (!nanlm_vec_is_data(a)) return -1;
    for (i = 0; i < dst->words; i++) {
        nan16_word w = nan16_bswap(a->w[i]);
        if (!nan16_gate(w) || nan16_n(w) != NAN16_N_DATA) return -1;
        dst->w[i] = w;
    }
    return 0;
}

uint64_t nanlm_vec_hamming(const nanlm_vec *a, const nanlm_vec *b) {
    size_t i;
    uint64_t sum = 0;
    if (!same_shape(a, b) || !nanlm_vec_is_data(a) || !nanlm_vec_is_data(b)) {
        return UINT64_MAX;
    }
    for (i = 0; i < a->words; i++) {
        uint32_t x = nan16_payload(a->w[i]) ^ nan16_payload(b->w[i]);
        sum += nan16_popcount32(x);
    }
    return sum;
}

uint64_t nanlm_vec_popcount(const nanlm_vec *a) {
    size_t i;
    uint64_t sum = 0;
    if (!a || !a->w || !nanlm_vec_is_data(a)) return UINT64_MAX;
    for (i = 0; i < a->words; i++) {
        sum += nan16_popcount32(nan16_payload(a->w[i]));
    }
    return sum;
}

int nanlm_vec_equal(const nanlm_vec *a, const nanlm_vec *b) {
    size_t i;
    if (!same_shape(a, b)) return -1;
    for (i = 0; i < a->words; i++) {
        if (a->w[i] != b->w[i]) return 0;
    }
    return 1;
}
