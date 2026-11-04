#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "nanlm_model.h"

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
    enum { DIM = 256, CLASSES = 4, QUERIES = 8, SAMPLES = 32, NOISE = 8 };
    nanlm_model *m = nanlm_model_new(DIM, CLASSES, QUERIES);
    nanlm_model *m2 = nanlm_model_new(DIM, CLASSES, QUERIES);
    uint32_t i, c;

    CHECK(m && m2, "model allocation");
    if (!m || !m2) return 1;

    CHECK(m->words == DIM / 32, "words");
    CHECK(m->classes == CLASSES, "classes");
    CHECK(m->nqueries == QUERIES, "queries");

    CHECK(nanlm_model_train_toy(m, 0x123456789ABCDEF0ull, SAMPLES, NOISE) == 0,
          "train m");
    CHECK(nanlm_model_train_toy(m2, 0x123456789ABCDEF0ull, SAMPLES, NOISE) == 0,
          "train m2");

    for (c = 0; c < CLASSES; c++) {
        CHECK(nanlm_vec_is_data(m->prototypes[c]), "prototype %u data", c);
        CHECK(nanlm_vec_equal(m->prototypes[c], m2->prototypes[c]) == 1,
              "prototype %u deterministic", c);
    }

    for (i = 0; i < QUERIES; i++) {
        CHECK(nanlm_vec_equal(m->queries[i], m2->queries[i]) == 1,
              "query %u deterministic", i);
        CHECK(m->expected[i] == m2->expected[i], "label %u deterministic", i);
    }

    {
        int sequential = 0;
        int seen[CLASSES] = {0};
        for (i = 0; i < QUERIES; i++) {
            if (m->expected[i] == (uint8_t)(i % CLASSES)) sequential++;
            if (m->expected[i] < CLASSES) seen[m->expected[i]]++;
        }
        CHECK(sequential != QUERIES, "query labels are still sequential");
        for (c = 0; c < CLASSES; c++) {
            CHECK(seen[c] > 0, "class %u has no query", c);
        }
    }

    for (i = 0; i < QUERIES; i++) {
        uint32_t got = 0;
        CHECK(nanlm_vec_is_data(m->queries[i]), "query %u data", i);
        CHECK(nanlm_model_predict(m, m->queries[i], &got) == 0, "predict %u", i);
        CHECK(got == m->expected[i], "query %u got=%u want=%u", i, got,
              m->expected[i]);
    }

    CHECK(nanlm_model_emit_nas(m, "build/test_model.nas") == 0, "emit nas");
    CHECK(nanlm_model_write_expected(m, "build/test_model.expected") == 0,
          "emit expected");
    CHECK(nanlm_model_write_manifest(m, "build/test_model.manifest", 1, SAMPLES,
                                     NOISE) == 0,
          "emit manifest");

    {
        FILE *f = fopen("build/test_model.nas", "r");
        char line[256];
        int saw_equ = 0, saw_proto = 0, saw_query = 0, saw_org = 0;
        CHECK(f != NULL, "open emitted nas");
        if (f) {
            while (fgets(line, sizeof(line), f)) {
                if (strstr(line, ".equ NVEC 4")) saw_equ = 1;
                if (strstr(line, "prototypes:")) saw_proto = 1;
                if (strstr(line, "queries:")) saw_query = 1;
                if (strstr(line, ".org 8192")) saw_org = 1;
            }
            fclose(f);
        }
        CHECK(saw_equ, "emitted NVEC");
        CHECK(saw_proto, "emitted prototypes");
        CHECK(saw_query, "emitted queries");
        CHECK(saw_org, "emitted data org");
    }

    nanlm_model_free(m);
    nanlm_model_free(m2);

    if (failures) {
        fprintf(stderr, "test_model: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_model: PASS\n");
    return 0;
}
