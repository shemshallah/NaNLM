/* nan16_word.h - exact NaN16 word model (header-only C99).
 *
 * This is the word model of NaN/0S, not a new format:
 *
 *   bytes 0..1  FAR head, big-endian
 *   bytes 2..5  32-bit payload, little-endian
 *   bytes 6..7  NEAR head, little-endian
 *
 * Legal heads: 0x7FF1, 0xFFF1, 0x7FF8, 0xFFF8.
 * class nibble n = (near_class << 2) | far_class.
 * For BOOL, bit k of n is f(a,b) with k = 2a + b.
 *
 * The canonical in-memory value is the little-endian u64 of those bytes.
 * On a big-endian host, convert bytes explicitly at the I/O boundary.
 *
 * Reference: nan0s/aarch64/android/core/nan16_int.h (nan_mkword, nan_near,
 * nan_far, nan_hcls, nan_hvalid) and spec/NAN16_SPEC.md.
 */
#ifndef NAN16_WORD_H
#define NAN16_WORD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t nan16_word;

#define NAN16_HEAD_S_POS 0x7FF1u
#define NAN16_HEAD_S_NEG 0xFFF1u
#define NAN16_HEAD_Q_POS 0x7FF8u
#define NAN16_HEAD_Q_NEG 0xFFF8u

/* Canonical data nibble: both heads +q, neutral under BSWAPM. */
#define NAN16_N_DATA 10u
/* Canonical control nibble: every non-BOOL op always runs. */
#define NAN16_N_ALWAYS 15u

static inline uint16_t nan16_head_of(unsigned cls) {
    switch (cls & 3u) {
    case 0: return NAN16_HEAD_S_POS;
    case 1: return NAN16_HEAD_S_NEG;
    case 2: return NAN16_HEAD_Q_POS;
    default: return NAN16_HEAD_Q_NEG;
    }
}

static inline int nan16_head_valid(uint32_t h) {
    return (h & 0x7FF0u) == 0x7FF0u &&
           ((h & 0xFu) == 1u || (h & 0xFu) == 8u);
}

/* class(h) = 2*Q + S, as in the NaN16 core. */
static inline uint32_t nan16_head_class(uint32_t h) {
    return ((h >> 2) & 2u) | (h >> 15);
}

/* FAR head as read big-endian from bytes 0..1. */
static inline uint32_t nan16_far(nan16_word w) {
    return ((uint32_t)(w & 0xFFu) << 8) | ((uint32_t)(w >> 8) & 0xFFu);
}

/* NEAR head as read little-endian from bytes 6..7. */
static inline uint32_t nan16_near(nan16_word w) {
    return (uint32_t)(w >> 48);
}

static inline int nan16_gate(nan16_word w) {
    return nan16_head_valid(nan16_near(w)) && nan16_head_valid(nan16_far(w));
}

/* Only meaningful when nan16_gate(w) is true. */
static inline uint32_t nan16_n(nan16_word w) {
    return (nan16_head_class(nan16_near(w)) << 2) |
           nan16_head_class(nan16_far(w));
}

static inline uint32_t nan16_payload(nan16_word w) {
    return (uint32_t)(w >> 16);
}

/* Build a canonical word. n is the 4-bit class nibble. */
static inline nan16_word nan16_make(uint32_t payload, unsigned n) {
    uint32_t far = nan16_head_of(n & 3u);
    uint32_t near = nan16_head_of((n >> 2) & 3u);
    return (uint64_t)(far >> 8) |
           ((uint64_t)(far & 0xFFu) << 8) |
           ((uint64_t)payload << 16) |
           ((uint64_t)near << 48);
}

/* BSWAPM: reverse all eight bytes. Swaps the faces and reverses the payload. */
static inline uint64_t nan16_bswap64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(x);
#else
    return ((x & 0x00000000000000FFull) << 56) |
           ((x & 0x000000000000FF00ull) << 40) |
           ((x & 0x0000000000FF0000ull) << 24) |
           ((x & 0x00000000FF000000ull) << 8) |
           ((x & 0x000000FF00000000ull) >> 8) |
           ((x & 0x0000FF0000000000ull) >> 24) |
           ((x & 0x00FF000000000000ull) >> 40) |
           ((x & 0xFF00000000000000ull) >> 56);
#endif
}

static inline nan16_word nan16_bswap(nan16_word w) {
    return nan16_bswap64(w);
}

/* rot2: the nibble transform induced by swapping the faces. */
static inline unsigned nan16_rot2(unsigned n) {
    return ((n & 3u) << 2) | ((n >> 2) & 3u);
}

static inline int nan16_is_neutral(unsigned n) {
    return n == 0u || n == 5u || n == 10u || n == 15u;
}

/* BOOL.n over 32 lanes. This is the exact rule in nan16.c:
 *   bit0 ~(A|B), bit1 ~A&B, bit2 A&~B, bit3 A&B
 * Branchless: each mask is all-ones or zero.
 */
static inline uint32_t nan16_bool(unsigned n, uint32_t a, uint32_t b) {
    uint32_t m0 = 0u - ((n >> 0) & 1u);
    uint32_t m1 = 0u - ((n >> 1) & 1u);
    uint32_t m2 = 0u - ((n >> 2) & 1u);
    uint32_t m3 = 0u - ((n >> 3) & 1u);
    return (m0 & ~(a | b)) |
           (m1 & (~a & b)) |
           (m2 & (a & ~b)) |
           (m3 & (a & b));
}

static inline uint32_t nan16_popcount32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}

/* Human-readable truth-table name. Useful for probes and disassembly. */
static inline const char *nan16_function_name(unsigned n) {
    switch (n & 15u) {
    case 0:  return "0";
    case 1:  return "NOR";
    case 2:  return "~A&B";
    case 3:  return "NOT A";
    case 4:  return "A&~B";
    case 5:  return "NOT B";
    case 6:  return "XOR";
    case 7:  return "NAND";
    case 8:  return "AND";
    case 9:  return "XNOR";
    case 10: return "B";
    case 11: return "A=>B";
    case 12: return "A";
    case 13: return "A|~B";
    case 14: return "OR";
    default: return "1";
    }
}

#ifdef __cplusplus
}
#endif

#endif /* NAN16_WORD_H */
