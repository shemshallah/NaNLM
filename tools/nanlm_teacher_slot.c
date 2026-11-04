#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanlm_slot.h"
#include "nanlm_teacher.h"

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

static int cmp_count(const void *a, const void *b) {
    const nanlm_slot_count *x = (const nanlm_slot_count *)a;
    const nanlm_slot_count *y = (const nanlm_slot_count *)b;
    if (x->key != y->key) return (x->key > y->key) - (x->key < y->key);
    return (int)x->next - (int)y->next;
}

int main(int argc, char **argv) {
    const char *teacher_path = NULL;
    const char *corpus = "data/alice_excerpt.txt";
    const char *out = "models/nanlm_teacher_slot.nas";
    const char *expected = "models/nanlm_teacher_slot.expected";
    const char *manifest = "models/nanlm_teacher_slot.manifest";
    uint32_t nqueries = 32;
    double frac = 0.85;
    nanlm_teacher_rec *tr = NULL;
    size_t tn = 0, i;
    nanlm_slot_count *cnt = NULL;
    size_t cn = 0, w = 0;
    unsigned char *data = NULL;
    size_t len = 0;
    nanlm_slot_model *m = NULL;
    int rc = 1;

    for (i = 1; i < (size_t)argc; i++) {
        const char *a = argv[i], *v = (i + 1 < (size_t)argc) ? argv[i + 1] : NULL;
        if (!strcmp(a, "--teacher") && v) { teacher_path = v; i++; }
        else if (!strcmp(a, "--corpus") && v) { corpus = v; i++; }
        else if (!strcmp(a, "--out") && v) { out = v; i++; }
        else if (!strcmp(a, "--expected") && v) { expected = v; i++; }
        else if (!strcmp(a, "--manifest") && v) { manifest = v; i++; }
        else if (!strcmp(a, "--queries") && v) { nqueries = (uint32_t)strtoul(v, NULL, 0); i++; }
        else if (!strcmp(a, "--train-frac") && v) { frac = strtod(v, NULL); i++; }
        else { fprintf(stderr, "usage: %s --teacher FILE [--corpus FILE] ...\n", argv[0]); return 2; }
    }
    if (!teacher_path) return 2;
    if (nanlm_teacher_read(teacher_path, &tr, &tn) != 0) {
        fprintf(stderr, "cannot read teacher %s\n", teacher_path);
        return 1;
    }
    cnt = (nanlm_slot_count *)calloc(tn ? tn : 1, sizeof(*cnt));
    if (!cnt) goto done;
    for (i = 0; i < tn; i++) {
        cnt[cn].key = (uint32_t)(tr[i].context & 0xFFFFFFu);
        cnt[cn].next = tr[i].next;
        cnt[cn].weight = tr[i].weight;
        cn++;
    }
    qsort(cnt, cn, sizeof(*cnt), cmp_count);
    for (i = 0; i < cn;) {
        uint32_t key = cnt[i].key, next = cnt[i].next, sum = 0;
        while (i < cn && cnt[i].key == key && cnt[i].next == next) {
            sum += cnt[i].weight;
            i++;
        }
        cnt[w].key = key;
        cnt[w].next = (uint8_t)next;
        cnt[w].weight = sum;
        w++;
    }
    data = read_file(corpus, &len);
    if (!data) goto done;
    m = nanlm_slot_build_counts(cnt, w, data, len, frac, nqueries);
    if (!m) goto done;
    printf("teacher-slot: teacher_records=%zu contexts=%zu keys=%u\n", tn, w,
           m->nkeys);
    printf("eval_n=%u hdc_top1=%.2f%% hdc_bpc=%.4f\n", m->eval_n,
           m->hdc_top1, m->hdc_bpc);
    printf("ngram_top1=%.2f%% ngram_bpc=%.4f\n", m->ngram_top1,
           m->ngram_bpc);
    if (nanlm_slot_emit_nas(m, out) != 0) goto done;
    if (nanlm_slot_write_expected(m, expected) != 0) goto done;
    if (nanlm_slot_write_manifest(m, manifest, 0, frac) != 0) goto done;
    rc = 0;
done:
    nanlm_slot_free(m);
    free(data);
    free(cnt);
    free(tr);
    return rc;
}
