/* nanlm_slot8_train.c - build the 8-byte / 16-slot student from teacher shards.
 *
 * Teacher records are (8-byte context, next byte, weight). We keep the
 * highest-mass contexts, then compile them into simplex-coded 8-byte + ctype
 * prototypes. The result is emitted as a ordinary .nas image for nanlm_vm.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanlm_slot8.h"
#include "nanlm_teacher.h"

typedef struct {
    nanlm_slot_key key;
    uint64_t total;
} group_t;

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

static int cmp_group_desc(const void *a, const void *b) {
    const group_t *x = (const group_t *)a;
    const group_t *y = (const group_t *)b;
    if (x->total != y->total) return (x->total < y->total) - (x->total > y->total);
    return (x->key > y->key) - (x->key < y->key);
}

static int cmp_key(const void *a, const void *b) {
    nanlm_slot_key x = *(const nanlm_slot_key *)a;
    nanlm_slot_key y = *(const nanlm_slot_key *)b;
    return (x > y) - (x < y);
}

static int key_selected(const nanlm_slot_key *keys, size_t n,
                        nanlm_slot_key key) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (keys[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    return lo < n && keys[lo] == key;
}

int main(int argc, char **argv) {
    const char *teacher_path = NULL;
    const char *corpus = "data/alice_excerpt.txt";
    const char *out = "models/nanlm_slot8.nas";
    const char *expected = "models/nanlm_slot8.expected";
    const char *manifest = "models/nanlm_slot8.manifest";
    uint32_t nqueries = 32;
    uint32_t max_keys = 4096;
    double frac = 0.85;
    nanlm_teacher_rec *tr = NULL;
    size_t tn = 0, i;
    nanlm_slot_count *all = NULL;
    size_t n = 0, w = 0;
    group_t *groups = NULL;
    size_t ngroups = 0;
    nanlm_slot_key *selected = NULL;
    size_t nselected = 0;
    nanlm_slot_count *kept = NULL;
    size_t nkept = 0;
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
        else if (!strcmp(a, "--max-keys") && v) { max_keys = (uint32_t)strtoul(v, NULL, 0); i++; }
        else if (!strcmp(a, "--train-frac") && v) { frac = strtod(v, NULL); i++; }
        else {
            fprintf(stderr, "usage: %s --teacher FILE [--corpus FILE] [--max-keys N] ...\n",
                    argv[0]);
            return 2;
        }
    }
    if (!teacher_path) return 2;
    if (nanlm_teacher_read(teacher_path, &tr, &tn) != 0) {
        fprintf(stderr, "cannot read teacher %s\n", teacher_path);
        return 1;
    }
    all = (nanlm_slot_count *)calloc(tn ? tn : 1, sizeof(*all));
    if (!all) goto done;
    for (i = 0; i < tn; i++) {
        all[n].key = tr[i].context;
        all[n].next = tr[i].next;
        all[n].weight = tr[i].weight;
        n++;
    }
    qsort(all, n, sizeof(*all), cmp_count);
    for (i = 0; i < n;) {
        nanlm_slot_key key = all[i].key;
        uint8_t next = all[i].next;
        uint32_t sum = 0;
        while (i < n && all[i].key == key && all[i].next == next) {
            sum += all[i].weight;
            i++;
        }
        all[w].key = key;
        all[w].next = next;
        all[w].weight = (uint16_t)(sum > 65535u ? 65535u : sum);
        w++;
    }
    n = w;

    groups = (group_t *)calloc(n ? n : 1, sizeof(*groups));
    if (!groups) goto done;
    for (i = 0; i < n;) {
        nanlm_slot_key key = all[i].key;
        uint64_t total = 0;
        while (i < n && all[i].key == key) {
            total += all[i].weight;
            i++;
        }
        groups[ngroups].key = key;
        groups[ngroups].total = total;
        ngroups++;
    }
    qsort(groups, ngroups, sizeof(*groups), cmp_group_desc);
    nselected = ngroups;
    if (max_keys != 0 && nselected > max_keys) nselected = max_keys;
    selected = (nanlm_slot_key *)calloc(nselected ? nselected : 1,
                                        sizeof(*selected));
    if (!selected) goto done;
    for (i = 0; i < nselected; i++) selected[i] = groups[i].key;
    qsort(selected, nselected, sizeof(*selected), cmp_key);

    kept = (nanlm_slot_count *)calloc(n ? n : 1, sizeof(*kept));
    if (!kept) goto done;
    for (i = 0; i < n; i++) {
        if (key_selected(selected, nselected, all[i].key)) {
            kept[nkept++] = all[i];
        }
    }

    data = read_file(corpus, &len);
    if (!data) goto done;
    m = nanlm_slot_build_counts(kept, nkept, data, len, frac, nqueries);
    if (!m) goto done;
    printf("slot8: teacher_records=%zu contexts=%zu selected=%zu keys=%u\n",
           tn, ngroups, nselected, m->nkeys);
    printf("eval_n=%u top1=%.2f%% bpc=%.4f (order-3 baseline %.2f%% / %.4f)\n",
           m->eval_n, m->hdc_top1, m->hdc_bpc, m->ngram_top1, m->ngram_bpc);
    if (nanlm_slot_emit_nas(m, out) != 0) goto done;
    if (nanlm_slot_write_expected(m, expected) != 0) goto done;
    if (nanlm_slot_write_manifest(m, manifest, 0, frac) != 0) goto done;
    rc = 0;
done:
    nanlm_slot_free(m);
    free(data);
    free(kept);
    free(selected);
    free(groups);
    free(all);
    free(tr);
    return rc;
}
