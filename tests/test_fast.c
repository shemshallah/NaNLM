#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nanlm_fast.h"

static uint32_t scalar_hamming(const uint32_t *a, const uint32_t *b, size_t n) {
    uint32_t d = 0;
    size_t i;
    for (i = 0; i < n; i++) {
        uint32_t x = a[i] ^ b[i];
        x = x - ((x >> 1) & 0x55555555u);
        x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
        x = (x + (x >> 4)) & 0x0F0F0F0Fu;
        d += (x * 0x01010101u) >> 24;
    }
    return d;
}

int main(void) {
    enum { N = 1024 };
    uint32_t a[N], b[N];
    uint32_t seed = 0x12345678u;
    size_t i;
    for (i = 0; i < N; i++) {
        seed = seed * 1664525u + 1013904223u;
        a[i] = seed;
        seed = seed * 1664525u + 1013904223u;
        b[i] = seed;
    }
    {
        uint32_t got = nanlm_fast_hamming32(a, b, N);
        uint32_t want = scalar_hamming(a, b, N);
        if (got != want) {
            fprintf(stderr, "fast hamming mismatch: got=%u want=%u\n", got, want);
            return 1;
        }
        printf("fast_hamming: PASS (%u bits over %d words)\n", got, N);
    }
    {
        uint32_t h1 = nanlm_fast_fnv32(a, N, 2166136261u);
        uint32_t h2 = 2166136261u;
        for (i = 0; i < N; i++) {
            h2 ^= a[i];
            h2 *= 16777619u;
        }
        if (h1 != h2) {
            fprintf(stderr, "fast fnv mismatch\n");
            return 1;
        }
        printf("fast_fnv: PASS (%08X)\n", h1);
    }
    return 0;
}
