#include <stdio.h>
#include <stdint.h>

#include "nan16_word.h"

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

int main(void) {
    unsigned n;
    uint32_t payload = 0x12345678u;

    for (n = 0; n < 16; n++) {
        nan16_word w = nan16_make(payload, n);
        CHECK(nan16_gate(w), "n=%u gate failed", n);
        CHECK(nan16_payload(w) == payload, "n=%u payload=%08x", n,
              nan16_payload(w));
        CHECK(nan16_n(w) == n, "n=%u read back %u", n, nan16_n(w));
        CHECK(nan16_head_valid(nan16_near(w)), "n=%u near head invalid", n);
        CHECK(nan16_head_valid(nan16_far(w)), "n=%u far head invalid", n);
    }

    /* The lane identity: the opcode is literally the output. */
    for (n = 0; n < 16; n++) {
        uint32_t got = nan16_bool(n, 0xCCCCCCCCu, 0xAAAAAAAAu);
        uint32_t want = (uint32_t)n * 0x11111111u;
        CHECK(got == want, "lane n=%u got=%08x want=%08x", n, got, want);
    }

    /* The mirror law: BSWAPM swaps the faces and maps n to rot2(n). */
    {
        static const uint32_t as[] = {0x00000000u, 0xFFFFFFFFu, 0x12345678u,
                                      0xAAAAAAAAu, 0xCCCCCCCCu};
        static const uint32_t bs[] = {0x00000000u, 0xFFFFFFFFu, 0x9ABCDEF0u,
                                      0x55555555u, 0x33333333u};
        size_t i;
        for (n = 0; n < 16; n++) {
            for (i = 0; i < sizeof(as) / sizeof(as[0]); i++) {
                uint32_t lhs = nan16_bool(nan16_rot2(n), as[i], bs[i]);
                uint32_t rhs = nan16_bool(n, ~as[i], bs[i]);
                CHECK(lhs == rhs, "mirror n=%u i=%zu lhs=%08x rhs=%08x", n, i,
                      lhs, rhs);
            }
        }
    }

    /* BSWAPM is an involution and preserves neutral data nibbles. */
    for (n = 0; n < 16; n++) {
        nan16_word w = nan16_make(payload, n);
        nan16_word b = nan16_bswap(w);
        CHECK(nan16_gate(b), "bswap n=%u gate failed", n);
        CHECK(nan16_n(b) == nan16_rot2(n), "bswap n=%u -> %u want %u", n,
              nan16_n(b), nan16_rot2(n));
        CHECK(nan16_bswap(b) == w, "bswap twice n=%u", n);
    }

    {
        nan16_word w = nan16_make(payload, NAN16_N_DATA);
        nan16_word b = nan16_bswap(w);
        CHECK(nan16_n(b) == NAN16_N_DATA, "neutral bswap changed n");
        CHECK(nan16_payload(b) == bswap32(payload),
              "neutral bswap payload=%08x want=%08x", nan16_payload(b),
              bswap32(payload));
    }

    /* Invalid heads must fail the gate. */
    {
        nan16_word w = nan16_make(payload, NAN16_N_DATA);
        nan16_word bad_far = w ^ 0x0000000000000001ull;
        nan16_word bad_near = w ^ 0x0001000000000000ull;
        CHECK(!nan16_gate(bad_far), "bad far head passed the gate");
        CHECK(!nan16_gate(bad_near), "bad near head passed the gate");
    }

    /* Popcount reference values. */
    CHECK(nan16_popcount32(0u) == 0u, "popcount 0");
    CHECK(nan16_popcount32(0xFFFFFFFFu) == 32u, "popcount all");
    CHECK(nan16_popcount32(0x80000001u) == 2u, "popcount 0x80000001");
    CHECK(nan16_popcount32(0xAAAAAAAAu) == 16u, "popcount 0xAAAAAAAA");
    CHECK(nan16_popcount32(0x55555555u) == 16u, "popcount 0x55555555");

    /* The four neutral nibbles are fixed by rot2. */
    CHECK(nan16_is_neutral(0u), "n=0 neutral");
    CHECK(nan16_is_neutral(5u), "n=5 neutral");
    CHECK(nan16_is_neutral(10u), "n=10 neutral");
    CHECK(nan16_is_neutral(15u), "n=15 neutral");
    CHECK(nan16_rot2(0u) == 0u, "rot2(0)");
    CHECK(nan16_rot2(5u) == 5u, "rot2(5)");
    CHECK(nan16_rot2(10u) == 10u, "rot2(10)");
    CHECK(nan16_rot2(15u) == 15u, "rot2(15)");
    CHECK(nan16_rot2(6u) == 9u, "rot2(6)");
    CHECK(nan16_rot2(9u) == 6u, "rot2(9)");

    if (failures) {
        fprintf(stderr, "test_word: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_word: PASS\n");
    return 0;
}
