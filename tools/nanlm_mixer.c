#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanlm_mixer.h"

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
        free(b); fclose(f); return NULL;
    }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

int main(int argc, char **argv) {
    const char *corpus = "data/alice_excerpt.txt";
    const char *log = NULL;
    uint32_t table_bits = 16;
    double frac = 0.85;
    int doc = 1, online = 0, top1_limit = -1;
    unsigned char *data;
    size_t len = 0;
    nanlm_mixer_report report;
    int i;
    for (i = 1; i < argc; i++) {
        const char *a = argv[i], *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!strcmp(a, "--corpus") && v) { corpus = v; i++; }
        else if (!strcmp(a, "--log") && v) { log = v; i++; }
        else if (!strcmp(a, "--table-bits") && v) { table_bits = (uint32_t)strtoul(v, NULL, 0); i++; }
        else if (!strcmp(a, "--train-frac") && v) { frac = strtod(v, NULL); i++; }
        else if (!strcmp(a, "--doc") && v) { doc = atoi(v); i++; }
        else if (!strcmp(a, "--online") && v) { online = atoi(v); i++; }
        else if (!strcmp(a, "--top1-limit") && v) { top1_limit = atoi(v); i++; }
        else {
            fprintf(stderr, "usage: %s [--corpus FILE] [--log FILE] [--table-bits N] [--doc 0|1] [--online 0|1] [--top1-limit N] [--train-frac F]\n", argv[0]);
            return 2;
        }
    }
    data = read_file(corpus, &len);
    if (!data) return 1;
    if (nanlm_mixer_train_eval(data, len, frac, table_bits, doc, online,
                               top1_limit, log, &report) != 0) {
        free(data);
        return 1;
    }
    printf("mixer: corpus=%s bytes=%zu eval=%llu\n", corpus, len,
           (unsigned long long)report.eval_bytes);
    printf("mixer: bpc=%.4f top1=%.2f%% (n=%llu) ngram_bpc=%.4f ngram_top1=%.2f%%\n",
           report.bpc, report.top1,
           (unsigned long long)report.top1_n,
           report.ngram_bpc, report.ngram_top1);
    printf("mixer: model_bytes=%zu hutter_total_bpc=%.4f\n",
           report.model_bytes, report.hutter_total_bpc);
    free(data);
    return 0;
}
