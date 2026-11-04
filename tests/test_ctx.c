#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "nanlm_ctx.h"

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

int main(void) {
    enum { DIM = 256, CLASSES = 4, QUERIES = 4 };
    nanlm_ctx_model *m = nanlm_ctx_model_new(DIM, CLASSES, QUERIES);
    nanlm_ctx_model *m2 = nanlm_ctx_model_new(DIM, CLASSES, QUERIES);
    nanlm_vec *t0 = nanlm_vec_new(DIM / 32);
    nanlm_vec *t1 = nanlm_vec_new(DIM / 32);
    nanlm_vec *t2 = nanlm_vec_new(DIM / 32);
    nanlm_vec *q = nanlm_vec_new(DIM / 32);
    uint32_t i, c, correct = 0;

    CHECK(m && m2 && t0 && t1 && t2 && q, "allocation");
    if (!m || !m2 || !t0 || !t1 || !t2 || !q) return 1;

    CHECK(nanlm_ctx_build(m, 0xABCDEF0123456789ull) == 0, "build m");
    CHECK(nanlm_ctx_build(m2, 0xABCDEF0123456789ull) == 0, "build m2");

    for (i = 0; i < CLASSES * NANLM_CTX_LEN; i++) {
        CHECK(nanlm_vec_is_data(m->codebook[i]), "codebook %u data", i);
    }
    for (i = 0; i < NANLM_CTX_LEN; i++) {
        CHECK(nanlm_vec_is_data(m->positions[i]), "position %u data", i);
    }
    for (c = 0; c < CLASSES; c++) {
        CHECK(nanlm_vec_is_data(m->prototypes[c]), "prototype %u data", c);
        CHECK(nanlm_vec_equal(m->prototypes[c], m2->prototypes[c]) == 1,
              "prototype %u deterministic", c);
    }
    for (i = 0; i < QUERIES; i++) {
        CHECK(nanlm_vec_equal(m->queries[i], m2->queries[i]) == 1,
              "query %u deterministic", i);
        CHECK(m->hashes[i] == m2->hashes[i], "hash %u deterministic", i);
        CHECK(m->expected[i] == m2->expected[i], "label %u deterministic", i);
    }

    /* Manual bind+bundle must equal nanlm_ctx_encode for every query. */
    for (i = 0; i < QUERIES; i++) {
        const uint8_t *ids = m->query_ids + (size_t)i * NANLM_CTX_LEN;
        CHECK(nanlm_ctx_encode(m, ids, q) == 0, "encode %u", i);
        CHECK(nanlm_vec_bind(t0, m->codebook[ids[0]], m->positions[0]) == 0,
              "bind 0");
        CHECK(nanlm_vec_bind(t1, m->codebook[ids[1]], m->positions[1]) == 0,
              "bind 1");
        CHECK(nanlm_vec_bind(t2, m->codebook[ids[2]], m->positions[2]) == 0,
              "bind 2");
        CHECK(nanlm_vec_bundle3(q, t0, t1, t2) == 0, "bundle");
        CHECK(nanlm_vec_equal(q, m->queries[i]) == 1, "query %u encode", i);
        CHECK(nanlm_ctx_hash(m->queries[i]) == m->hashes[i],
              "query %u hash", i);
    }

    for (i = 0; i < QUERIES; i++) {
        uint32_t got = 0;
        CHECK(nanlm_ctx_predict(m, m->queries[i], &got) == 0, "predict %u", i);
        if (got == m->expected[i]) correct++;
    }
    CHECK(correct == QUERIES, "C accuracy %u/%u", correct, QUERIES);

    CHECK(nanlm_ctx_emit_nas(m, "build/test_ctx.nas") == 0, "emit nas");
    CHECK(nanlm_ctx_write_expected(m, "build/test_ctx.expected") == 0,
          "emit expected");
    {
        FILE *f = fopen("build/test_ctx.nas", "r");
        char line[256];
        int saw_vw = 0, saw_ctx = 0, saw_codebook = 0, saw_positions = 0;
        int saw_proto = 0, saw_ids = 0, saw_qbuf = 0;
        CHECK(f != NULL, "open emitted nas");
        if (f) {
            while (fgets(line, sizeof(line), f)) {
                if (strstr(line, ".equ VW 8")) saw_vw = 1;
                if (strstr(line, ".equ CTX 3")) saw_ctx = 1;
                if (strstr(line, "codebook:")) saw_codebook = 1;
                if (strstr(line, "positions:")) saw_positions = 1;
                if (strstr(line, "prototypes:")) saw_proto = 1;
                if (strstr(line, "query_ids:")) saw_ids = 1;
                if (strstr(line, "qbuf:")) saw_qbuf = 1;
            }
            fclose(f);
        }
        CHECK(saw_vw, "emitted VW");
        CHECK(saw_ctx, "emitted CTX");
        CHECK(saw_codebook, "emitted codebook");
        CHECK(saw_positions, "emitted positions");
        CHECK(saw_proto, "emitted prototypes");
        CHECK(saw_ids, "emitted query ids");
        CHECK(saw_qbuf, "emitted qbuf");
    }

    /* Negative gate: zeroing positions must change at least one query hash. */
    {
        uint32_t before = m2->hashes[0];
        uint8_t ids[NANLM_CTX_LEN];
        memcpy(ids, m2->query_ids, NANLM_CTX_LEN);
        nanlm_ctx_zero_positions(m2);
        CHECK(nanlm_ctx_encode(m2, ids, q) == 0, "zero-position encode");
        CHECK(nanlm_ctx_hash(q) != before, "zero positions did not change hash");
    }

    /* Negative gate: swapping 0/1 must change at least one prediction. */
    {
        int changed = 0;
        CHECK(nanlm_ctx_swap_prototypes(m2, 0, 1) == 0, "swap prototypes");
        for (i = 0; i < QUERIES; i++) {
            uint32_t got = 0;
            nanlm_ctx_predict(m2, m2->queries[i], &got);
            if (got != m2->expected[i]) changed++;
        }
        CHECK(changed > 0, "swapped prototypes changed no prediction");
    }

    nanlm_vec_free(t0);
    nanlm_vec_free(t1);
    nanlm_vec_free(t2);
    nanlm_vec_free(q);
    nanlm_ctx_model_free(m);
    nanlm_ctx_model_free(m2);

    if (failures) {
        fprintf(stderr, "test_ctx: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_ctx: PASS\n");
    return 0;
}
