#include "nanlm_teacher.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NTR_MAGIC 0x3152544Eu /* "NTR1" */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t nrec;
    uint32_t reserved;
} ntr_header;

static int cmp_packed(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int cmp_rec(const void *a, const void *b) {
    const nanlm_teacher_rec *x = (const nanlm_teacher_rec *)a;
    const nanlm_teacher_rec *y = (const nanlm_teacher_rec *)b;
    if (x->context != y->context)
        return (x->context > y->context) - (x->context < y->context);
    return (int)x->next - (int)y->next;
}

uint64_t nanlm_teacher_key(const uint8_t ctx[8]) {
    uint64_t k = 0;
    int i;
    for (i = 0; i < 8; i++) k = (k << 8) | ctx[i];
    return k;
}

int nanlm_teacher_build_ngram(const uint8_t *data, size_t start, size_t end,
                              uint32_t top_k, nanlm_teacher_rec **out,
                              size_t *out_n) {
    uint32_t *obs;
    size_t n = 0, i, cap, nrec = 0;
    nanlm_teacher_rec *rec;

    if (!data || !out || !out_n || end <= start + 9 || top_k == 0 ||
        top_k > 256) {
        return -1;
    }
    cap = end - start;
    obs = (uint32_t *)malloc(cap * sizeof(*obs));
    rec = (nanlm_teacher_rec *)calloc(cap * top_k, sizeof(*rec));
    if (!obs || !rec) { free(obs); free(rec); return -1; }
    for (i = start + 2; i + 1 < end; i++) {
        uint32_t key = ((uint32_t)data[i - 2] << 16) |
                       ((uint32_t)data[i - 1] << 8) | data[i];
        obs[n++] = (key << 8) | data[i + 1];
    }
    qsort(obs, n, sizeof(*obs), cmp_packed);

    for (i = start + 7; i + 1 < end; i++) {
        uint8_t ctx[8];
        uint32_t counts[256] = {0};
        uint32_t key = ((uint32_t)data[i - 2] << 16) |
                       ((uint32_t)data[i - 1] << 8) | data[i];
        size_t lo = 0, hi = n;
        uint32_t b, total = 0;
        double prob[256];
        uint8_t best[256];
        uint32_t nb = 0, sumw = 0;
        memcpy(ctx, data + i - 7, 8);
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (obs[mid] < (key << 8)) lo = mid + 1; else hi = mid;
        }
        while (lo < n && (obs[lo] >> 8) == key) {
            counts[obs[lo] & 0xFFu]++;
            total++;
            lo++;
        }
        for (b = 0; b < 256; b++) {
            prob[b] = ((double)counts[b] + 0.1) /
                      ((double)total + 0.1 * 256.0);
        }
        for (b = 0; b < 256; b++) {
            uint32_t j, bestj = nb;
            for (j = 0; j < nb; j++) {
                if (prob[b] > prob[best[j]]) { bestj = j; break; }
            }
            if (bestj == nb) {
                if (nb < top_k) best[nb++] = (uint8_t)b;
            } else {
                if (nb < top_k) nb++;
                memmove(best + bestj + 1, best + bestj, nb - bestj - 1);
                best[bestj] = (uint8_t)b;
            }
        }
        for (b = 0; b < nb; b++) sumw += (uint32_t)(prob[best[b]] * 1024.0);
        if (sumw == 0) sumw = 1;
        for (b = 0; b < nb; b++) {
            uint32_t w = (uint32_t)((prob[best[b]] * 1024.0) *
                                    (256.0 / (double)sumw));
            rec[nrec].context = nanlm_teacher_key(ctx);
            rec[nrec].next = best[b];
            rec[nrec].weight = (uint16_t)(w > 65535u ? 65535u : (w ? w : 1u));
            nrec++;
        }
    }
    free(obs);
    /* Sort and merge duplicate (context,next) records. */
    qsort(rec, nrec, sizeof(*rec), cmp_rec);
    {
        size_t w = 0;
        for (i = 0; i < nrec;) {
            uint64_t c = rec[i].context;
            uint8_t nx = rec[i].next;
            uint32_t s = 0;
            while (i < nrec && rec[i].context == c && rec[i].next == nx) {
                s += rec[i].weight;
                i++;
            }
            rec[w].context = c;
            rec[w].next = nx;
            rec[w].weight = (uint16_t)(s > 65535 ? 65535 : s);
            w++;
        }
        nrec = w;
    }
    *out = rec;
    *out_n = nrec;
    return 0;
}

int nanlm_teacher_write(const char *path, const nanlm_teacher_rec *rec,
                        size_t n) {
    FILE *f;
    ntr_header h;
    if (!path || (!rec && n)) return -1;
    f = fopen(path, "wb");
    if (!f) return -1;
    h.magic = NTR_MAGIC;
    h.version = 1;
    h.nrec = (uint32_t)n;
    h.reserved = 0;
    if (fwrite(&h, sizeof(h), 1, f) != 1 ||
        (n && fwrite(rec, sizeof(*rec), n, f) != n)) {
        fclose(f);
        return -1;
    }
    return fclose(f) == 0 ? 0 : -1;
}

int nanlm_teacher_read(const char *path, nanlm_teacher_rec **out,
                       size_t *out_n) {
    FILE *f;
    ntr_header h;
    nanlm_teacher_rec *rec;
    if (!path || !out || !out_n) return -1;
    f = fopen(path, "rb");
    if (!f) return -1;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != NTR_MAGIC ||
        h.version != 1) {
        fclose(f);
        return -1;
    }
    rec = (nanlm_teacher_rec *)calloc(h.nrec ? h.nrec : 1, sizeof(*rec));
    if (!rec) { fclose(f); return -1; }
    if (h.nrec && fread(rec, sizeof(*rec), h.nrec, f) != h.nrec) {
        free(rec);
        fclose(f);
        return -1;
    }
    fclose(f);
    *out = rec;
    *out_n = h.nrec;
    return 0;
}

int nanlm_teacher_merge(const char *out_path, const char **in_paths,
                        int n_in) {
    nanlm_teacher_rec *all = NULL;
    size_t n = 0, i;
    int k;
    if (!out_path || !in_paths || n_in <= 0) return -1;
    for (k = 0; k < n_in; k++) {
        nanlm_teacher_rec *r = NULL;
        size_t rn = 0;
        if (nanlm_teacher_read(in_paths[k], &r, &rn) != 0) {
            free(all);
            return -1;
        }
        {
            nanlm_teacher_rec *tmp = (nanlm_teacher_rec *)realloc(
                all, (n + rn) * sizeof(*all));
            if (!tmp) { free(r); free(all); return -1; }
            all = tmp;
        }
        memcpy(all + n, r, rn * sizeof(*r));
        n += rn;
        free(r);
    }
    qsort(all, n, sizeof(*all), cmp_rec);
    {
        size_t w = 0;
        for (i = 0; i < n;) {
            uint64_t c = all[i].context;
            uint8_t nx = all[i].next;
            uint32_t s = 0;
            while (i < n && all[i].context == c && all[i].next == nx) {
                s += all[i].weight;
                i++;
            }
            all[w].context = c;
            all[w].next = nx;
            all[w].weight = (uint16_t)(s > 65535 ? 65535 : s);
            w++;
        }
        n = w;
    }
    k = nanlm_teacher_write(out_path, all, n);
    free(all);
    return k;
}

int nanlm_teacher_scale(const char *out_path, const char *in_path,
                        uint32_t numerator, uint32_t denominator) {
    nanlm_teacher_rec *rec = NULL;
    size_t n = 0, i;
    int rc;
    if (!out_path || !in_path || denominator == 0) return -1;
    if (nanlm_teacher_read(in_path, &rec, &n) != 0) return -1;
    for (i = 0; i < n; i++) {
        uint64_t w = (uint64_t)rec[i].weight * numerator;
        w = (w + denominator / 2u) / denominator;
        if (w == 0 && rec[i].weight != 0) w = 1;
        rec[i].weight = (uint16_t)(w > 65535u ? 65535u : w);
    }
    rc = nanlm_teacher_write(out_path, rec, n);
    free(rec);
    return rc;
}

int nanlm_teacher_stats(const char *path, uint64_t *contexts,
                        uint64_t *records, uint64_t *weight) {
    nanlm_teacher_rec *rec = NULL;
    size_t n = 0, i;
    uint64_t c = 0, w = 0;
    if (!path) return -1;
    if (nanlm_teacher_read(path, &rec, &n) != 0) return -1;
    for (i = 0; i < n; i++) {
        w += rec[i].weight;
        if (i == 0 || rec[i].context != rec[i - 1].context) c++;
    }
    if (contexts) *contexts = c;
    if (records) *records = n;
    if (weight) *weight = w;
    free(rec);
    return 0;
}
