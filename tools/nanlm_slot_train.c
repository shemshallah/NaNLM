#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanlm_slot.h"

static void usage(const char *a) {
    fprintf(stderr,
            "usage: %s [--corpus FILE] [--out FILE] [--expected FILE]\n"
            "          [--manifest FILE] [--queries N] [--train-frac F]\n"
            "          [--seed HEX] [--eval-log FILE]\n",
            a);
}

static unsigned char *read_file(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb");
    long sz;
    unsigned char *b;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    sz = ftell(f);
    if (sz <= 0) { fclose(f); return NULL; }
    rewind(f);
    b = (unsigned char *)malloc((size_t)sz);
    if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) {
        free(b);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

int main(int argc, char **argv) {
    const char *corpus = "data/alice_excerpt.txt";
    const char *out = "models/nanlm_slot.nas";
    const char *expected = "models/nanlm_slot.expected";
    const char *manifest = "models/nanlm_slot.manifest";
    const char *eval_log = NULL;
    uint32_t nqueries = 32;
    double frac = 0.85;
    uint64_t seed = 0x4E414E4C534C4F54ull;
    unsigned char *data;
    size_t len = 0;
    nanlm_slot_model *m;
    int i, rc = 1;

    for (i = 1; i < argc; i++) {
        const char *a = argv[i], *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!strcmp(a, "--corpus") && v) { corpus = v; i++; }
        else if (!strcmp(a, "--out") && v) { out = v; i++; }
        else if (!strcmp(a, "--expected") && v) { expected = v; i++; }
        else if (!strcmp(a, "--manifest") && v) { manifest = v; i++; }
        else if (!strcmp(a, "--queries") && v) { nqueries = (uint32_t)strtoul(v, NULL, 0); i++; }
        else if (!strcmp(a, "--train-frac") && v) { frac = strtod(v, NULL); i++; }
        else if (!strcmp(a, "--seed") && v) { seed = strtoull(v, NULL, 0); i++; }
        else if (!strcmp(a, "--eval-log") && v) { eval_log = v; i++; }
        else { usage(argv[0]); return 2; }
    }
    data = read_file(corpus, &len);
    if (!data) { fprintf(stderr, "cannot read %s\n", corpus); return 1; }
    m = nanlm_slot_build(data, len, frac, nqueries);
    if (!m) { fprintf(stderr, "build failed\n"); free(data); return 1; }
    printf("nanlm-slot: keys=%u bits=%u queries=%u\n", m->nkeys,
           NANLM_SLOT_BITS, m->nqueries);
    printf("eval_n=%u hdc_top1=%.2f%% hdc_bpc=%.4f\n", m->eval_n,
           m->hdc_top1, m->hdc_bpc);
    printf("ngram_top1=%.2f%% ngram_bpc=%.4f\n", m->ngram_top1,
           m->ngram_bpc);
    if (nanlm_slot_eval_stream(m, data, len, (size_t)((double)len * frac),
                               eval_log) != 0) {
        if (eval_log) { fprintf(stderr, "stream eval failed\n"); goto done; }
    }
    if (m->stream_n) {
        printf("stream_n=%llu stream_top1=%.2f%% stream_bpc=%.4f "
               "stream_ngram_top1=%.2f%% stream_ngram_bpc=%.4f\n",
               (unsigned long long)m->stream_n, m->stream_top1,
               m->stream_bpc, m->stream_ngram_top1, m->stream_ngram_bpc);
    }
    if (nanlm_slot_emit_nas(m, out) != 0) goto done;
    if (nanlm_slot_write_expected(m, expected) != 0) goto done;
    if (nanlm_slot_write_manifest(m, manifest, seed, frac) != 0) goto done;
    printf("wrote %s, %s, %s\n", out, expected, manifest);
    rc = 0;
done:
    nanlm_slot_free(m);
    free(data);
    return rc;
}
