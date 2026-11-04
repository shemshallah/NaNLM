#include "nanlm_fast.h"

#include "nan16_word.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

uint32_t nanlm_fast_hamming32(const uint32_t *a, const uint32_t *b, size_t n) {
    size_t i = 0;
    uint32_t d = 0;
    if (!a || !b) return 0;
#if defined(__ARM_NEON)
    for (; i + 4 <= n; i += 4) {
        uint32x4_t x = veorq_u32(vld1q_u32(a + i), vld1q_u32(b + i));
        uint8x16_t bytes = vreinterpretq_u8_u32(x);
        uint8x16_t counts = vcntq_u8(bytes);
        d += (uint32_t)vaddvq_u32(vpaddlq_u16(vpaddlq_u8(counts)));
    }
#endif
    for (; i < n; i++) d += nan16_popcount32(a[i] ^ b[i]);
    return d;
}

uint32_t nanlm_fast_fnv32(const uint32_t *a, size_t n, uint32_t seed) {
    uint32_t h = seed ? seed : 2166136261u;
    size_t i;
    if (!a) return h;
    for (i = 0; i < n; i++) {
        h ^= a[i];
        h *= 16777619u;
    }
    return h;
}
