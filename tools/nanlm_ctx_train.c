#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanlm_ctx.h"

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--out FILE] [--expected FILE] [--manifest FILE]\n"
            "          [--dim BITS] [--classes N] [--queries N] [--seed HEX]\n"
            "          [--zero-positions] [--swap01]\n",
            argv0);
}

static int parse_u32(const char *s, uint32_t *out) {
    char *end = NULL;
    unsigned long v;
    if (!s || !*s) return -1;
    v = strtoul(s, &end, 0);
    if (!end || *end != '\0' || v > 0xFFFFFFFFul) return -1;
    *out = (uint32_t)v;
    return 0;
}

static int parse_u64(const char *s, uint64_t *out) {
    char *end = NULL;
    unsigned long long v;
    if (!s || !*s) return -1;
    v = strtoull(s, &end, 0);
    if (!end || *end != '\0') return -1;
    *out = (uint64_t)v;
    return 0;
}

int main(int argc, char **argv) {
    const char *out = "models/nanlm_ctx.nas";
    const char *expected_path = "models/nanlm_ctx.expected";
    const char *manifest_path = "models/nanlm_ctx.manifest";
    uint32_t dim = 2048;
    uint32_t classes = 16;
    uint32_t nqueries = 16;
    uint64_t seed = 0x4E414E4C4D324100ull;
    int zero_positions = 0;
    int swap01 = 0;
    nanlm_ctx_model *m;
    uint32_t i;
    uint32_t correct = 0;

    for (i = 1; i < (uint32_t)argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < (uint32_t)argc) ? argv[i + 1] : NULL;
        if (strcmp(a, "--out") == 0 && v) { out = v; i++; }
        else if (strcmp(a, "--expected") == 0 && v) { expected_path = v; i++; }
        else if (strcmp(a, "--manifest") == 0 && v) { manifest_path = v; i++; }
        else if (strcmp(a, "--dim") == 0 && v) { if (parse_u32(v, &dim)) return 2; i++; }
        else if (strcmp(a, "--classes") == 0 && v) { if (parse_u32(v, &classes)) return 2; i++; }
        else if (strcmp(a, "--queries") == 0 && v) { if (parse_u32(v, &nqueries)) return 2; i++; }
        else if (strcmp(a, "--seed") == 0 && v) { if (parse_u64(v, &seed)) return 2; i++; }
        else if (strcmp(a, "--zero-positions") == 0) { zero_positions = 1; }
        else if (strcmp(a, "--swap01") == 0) { swap01 = 1; }
        else {
            usage(argv[0]);
            return 2;
        }
    }

    if (dim == 0 || (dim % 32u) != 0 || classes == 0 || nqueries == 0) {
        fprintf(stderr, "invalid model dimensions\n");
        return 2;
    }
    if (dim > 65504u || nqueries > 65535u) {
        fprintf(stderr, "model too large for the current MOVI immediates\n");
        return 2;
    }

    m = nanlm_ctx_model_new(dim, classes, nqueries);
    if (!m) {
        fprintf(stderr, "model allocation failed\n");
        return 1;
    }
    if (nanlm_ctx_build(m, seed) != 0) {
        fprintf(stderr, "build failed\n");
        nanlm_ctx_model_free(m);
        return 1;
    }
    for (i = 0; i < m->nqueries; i++) {
        uint32_t got = 0;
        if (nanlm_ctx_predict(m, m->queries[i], &got) != 0) {
            fprintf(stderr, "prediction failed at query %u\n", i);
            nanlm_ctx_model_free(m);
            return 1;
        }
        if (got == m->expected[i]) correct++;
    }
    if (correct != m->nqueries) {
        fprintf(stderr, "C classifier: %u/%u correct, expected all\n", correct,
                m->nqueries);
        nanlm_ctx_model_free(m);
        return 1;
    }

    if (zero_positions) {
        nanlm_ctx_zero_positions(m);
        printf("negative gate: zeroed position vectors\n");
    }
    if (swap01) {
        if (nanlm_ctx_swap_prototypes(m, 0, 1) != 0) {
            fprintf(stderr, "could not swap prototypes\n");
            nanlm_ctx_model_free(m);
            return 1;
        }
        printf("negative gate: swapped prototypes 0 and 1\n");
    }

    if (nanlm_ctx_emit_nas(m, out) != 0) {
        fprintf(stderr, "could not write %s\n", out);
        nanlm_ctx_model_free(m);
        return 1;
    }
    if (nanlm_ctx_write_expected(m, expected_path) != 0) {
        fprintf(stderr, "could not write %s\n", expected_path);
        nanlm_ctx_model_free(m);
        return 1;
    }
    if (nanlm_ctx_write_manifest(m, manifest_path, seed) != 0) {
        fprintf(stderr, "could not write %s\n", manifest_path);
        nanlm_ctx_model_free(m);
        return 1;
    }

    printf("nanlm-ctx: dim=%u classes=%u queries=%u context=%u\n", m->dim,
           m->classes, m->nqueries, NANLM_CTX_LEN);
    printf("c_accuracy=%u/%u\n", correct, m->nqueries);
    printf("wrote %s, %s, %s\n", out, expected_path, manifest_path);

    nanlm_ctx_model_free(m);
    return 0;
}
