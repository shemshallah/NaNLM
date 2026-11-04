#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanlm_lm.h"

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--corpus FILE] [--out FILE] [--expected FILE]\n"
            "          [--manifest FILE] [--dim BITS] [--queries N]\n"
            "          [--train-frac F] [--seed HEX]\n"
            "          [--zero-positions] [--swap01] [--swap-predicted]\n",
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

static unsigned char *read_file(const char *path, size_t *out_len) {
    FILE *f;
    long sz;
    unsigned char *buf;
    f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    sz = ftell(f);
    if (sz <= 0) { fclose(f); return NULL; }
    rewind(f);
    buf = (unsigned char *)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

int main(int argc, char **argv) {
    const char *corpus = "data/alice_excerpt.txt";
    const char *out = "models/nanlm_lm.nas";
    const char *expected_path = "models/nanlm_lm.expected";
    const char *manifest_path = "models/nanlm_lm.manifest";
    uint32_t dim = 1024;
    uint32_t nqueries = 32;
    double train_frac = 0.85;
    uint64_t seed = 0x4E414E4C4D4C4D01ull;
    int zero_positions = 0;
    int swap01 = 0;
    int swap_predicted = 0;
    unsigned char *data;
    size_t len = 0;
    nanlm_lm_model *m;
    uint32_t i;
    int rc = 1;

    for (i = 1; i < (uint32_t)argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < (uint32_t)argc) ? argv[i + 1] : NULL;
        if (strcmp(a, "--corpus") == 0 && v) { corpus = v; i++; }
        else if (strcmp(a, "--out") == 0 && v) { out = v; i++; }
        else if (strcmp(a, "--expected") == 0 && v) { expected_path = v; i++; }
        else if (strcmp(a, "--manifest") == 0 && v) { manifest_path = v; i++; }
        else if (strcmp(a, "--dim") == 0 && v) { if (parse_u32(v, &dim)) return 2; i++; }
        else if (strcmp(a, "--queries") == 0 && v) { if (parse_u32(v, &nqueries)) return 2; i++; }
        else if (strcmp(a, "--train-frac") == 0 && v) { train_frac = strtod(v, NULL); i++; }
        else if (strcmp(a, "--seed") == 0 && v) { if (parse_u64(v, &seed)) return 2; i++; }
        else if (strcmp(a, "--zero-positions") == 0) { zero_positions = 1; }
        else if (strcmp(a, "--swap01") == 0) { swap01 = 1; }
        else if (strcmp(a, "--swap-predicted") == 0) { swap_predicted = 1; }
        else { usage(argv[0]); return 2; }
    }

    if (dim == 0 || (dim % 32u) != 0 || dim > 65504u || nqueries == 0 ||
        nqueries > 65535u || train_frac <= 0.0 || train_frac >= 1.0) {
        fprintf(stderr, "invalid model parameters\n");
        return 2;
    }

    data = read_file(corpus, &len);
    if (!data) {
        fprintf(stderr, "cannot read corpus %s\n", corpus);
        return 1;
    }
    m = nanlm_lm_model_new(dim, 256, nqueries);
    if (!m) {
        fprintf(stderr, "model allocation failed\n");
        free(data);
        return 1;
    }
    if (nanlm_lm_train(m, data, len, train_frac, seed) != 0) {
        fprintf(stderr, "training failed\n");
        goto done;
    }
    if (zero_positions) {
        nanlm_lm_zero_positions(m);
        printf("negative gate: zeroed position vectors\n");
    }
    if (swap01) {
        if (nanlm_lm_swap_prototypes(m, 0, 1) != 0) {
            fprintf(stderr, "could not swap prototypes\n");
            goto done;
        }
        printf("negative gate: swapped prototypes 0 and 1\n");
    }
    if (swap_predicted) {
        if (nanlm_lm_swap_predicted(m) != 0) {
            fprintf(stderr, "could not swap predicted prototypes\n");
            goto done;
        }
        printf("negative gate: swapped predicted prototypes\n");
    }
    if (nanlm_lm_emit_nas(m, out) != 0) {
        fprintf(stderr, "cannot write %s\n", out);
        goto done;
    }
    if (nanlm_lm_write_expected(m, expected_path) != 0) {
        fprintf(stderr, "cannot write %s\n", expected_path);
        goto done;
    }
    if (nanlm_lm_write_manifest(m, manifest_path, seed, train_frac) != 0) {
        fprintf(stderr, "cannot write %s\n", manifest_path);
        goto done;
    }

    printf("nanlm-lm: dim=%u vocab=%u queries=%u context=%u\n", m->dim,
           m->vocab, m->nqueries, NANLM_LM_CTX);
    printf("train_bytes=%u val_bytes=%u train_frac=%.3f\n", m->train_bytes,
           m->val_bytes, train_frac);
    printf("hdc_top1=%.2f%% ngram_top1=%.2f%%\n", m->hdc_top1,
           m->ngram_top1);
    printf("ngram_bpc=%.4f (HDC probability calibration not implemented)\n",
           m->ngram_bpc);
    printf("wrote %s, %s, %s\n", out, expected_path, manifest_path);
    rc = 0;

done:
    nanlm_lm_model_free(m);
    free(data);
    return rc;
}
