#include "nanlm_shard2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NSH2_MAGIC 0x3248534Eu /* "NSH2" little-endian */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t nrec;
    uint32_t reserved;
} nsh2_header;

static int cmp_packed(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int cmp_rec(const void *a, const void *b) {
    const nanlm_shard2_rec *x = (const nanlm_shard2_rec *)a;
    const nanlm_shard2_rec *y = (const nanlm_shard2_rec *)b;
    if (x->key != y->key) return (x->key > y->key) - (x->key < y->key);
    return (int)x->next - (int)y->next;
}

int nanlm_shard2_build(const uint8_t *data, size_t start, size_t end,
                       nanlm_shard2_rec **out_rec, size_t *out_n) {
    uint32_t *obs;
    size_t n = 0, cap, i, k = 0;
    nanlm_shard2_rec *rec;

    if (!data || !out_rec || !out_n || end <= start + 3) return -1;
    cap = end - start;
    obs = (uint32_t *)malloc(cap * sizeof(*obs));
    if (!obs) return -1;
    for (i = start + 2; i + 1 < end; i++) {
        uint32_t key = ((uint32_t)data[i - 2] << 16) |
                       ((uint32_t)data[i - 1] << 8) | data[i];
        obs[n++] = (key << 8) | data[i + 1];
    }
    if (n == 0) { free(obs); return -1; }
    qsort(obs, n, sizeof(*obs), cmp_packed);
    rec = (nanlm_shard2_rec *)calloc(n, sizeof(*rec));
    if (!rec) { free(obs); return -1; }
    for (i = 0; i < n;) {
        uint32_t key = obs[i] >> 8;
        uint32_t b = obs[i] & 0xFFu;
        size_t j = i;
        uint32_t cnt = 0;
        while (j < n && (obs[j] >> 8) == key && (obs[j] & 0xFFu) == b) {
            cnt++;
            j++;
        }
        rec[k].key = key;
        rec[k].next = (uint8_t)b;
        rec[k].count = (int32_t)(cnt > 0x7FFFFFFFu ? 0x7FFFFFFFu : cnt);
        k++;
        i = j;
    }
    free(obs);
    *out_rec = rec;
    *out_n = k;
    return 0;
}

int nanlm_shard2_write(const char *path, const nanlm_shard2_rec *rec, size_t n) {
    FILE *f;
    nsh2_header h;
    if (!path || (!rec && n)) return -1;
    f = fopen(path, "wb");
    if (!f) return -1;
    h.magic = NSH2_MAGIC;
    h.version = 2;
    h.nrec = (uint32_t)n;
    h.reserved = 0;
    if (fwrite(&h, sizeof(h), 1, f) != 1 ||
        (n && fwrite(rec, sizeof(*rec), n, f) != n)) {
        fclose(f);
        return -1;
    }
    return fclose(f) == 0 ? 0 : -1;
}

int nanlm_shard2_read(const char *path, nanlm_shard2_rec **out_rec,
                      size_t *out_n) {
    FILE *f;
    nsh2_header h;
    nanlm_shard2_rec *rec;
    if (!path || !out_rec || !out_n) return -1;
    f = fopen(path, "rb");
    if (!f) return -1;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != NSH2_MAGIC ||
        h.version != 2) {
        fclose(f);
        return -1;
    }
    rec = (nanlm_shard2_rec *)calloc(h.nrec ? h.nrec : 1, sizeof(*rec));
    if (!rec) { fclose(f); return -1; }
    if (h.nrec && fread(rec, sizeof(*rec), h.nrec, f) != h.nrec) {
        free(rec);
        fclose(f);
        return -1;
    }
    fclose(f);
    *out_rec = rec;
    *out_n = h.nrec;
    return 0;
}

int nanlm_shard2_merge(const char *out_path, const char **in_paths, int n_in) {
    nanlm_shard2_rec *all = NULL;
    size_t n = 0, i;
    int k;
    if (!out_path || !in_paths || n_in <= 0) return -1;
    for (k = 0; k < n_in; k++) {
        nanlm_shard2_rec *r = NULL;
        size_t rn = 0;
        if (nanlm_shard2_read(in_paths[k], &r, &rn) != 0) {
            free(all);
            return -1;
        }
        {
            nanlm_shard2_rec *tmp = (nanlm_shard2_rec *)realloc(
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
            uint32_t key = all[i].key;
            uint8_t next = all[i].next;
            uint32_t sum = 0;
            while (i < n && all[i].key == key && all[i].next == next) {
                /* Modular addition: the shard set is a group, not a
                 * saturating counter. */
                sum += (uint32_t)all[i].count;
                i++;
            }
            if (sum != 0) {
                all[w].key = key;
                all[w].next = next;
                all[w].count = (int32_t)sum;
                w++;
            }
        }
        n = w;
    }
    k = nanlm_shard2_write(out_path, all, n);
    free(all);
    return k;
}

int nanlm_shard2_negate(const char *out_path, const char *in_path) {
    nanlm_shard2_rec *rec = NULL;
    size_t n = 0, i;
    int rc;
    if (!out_path || !in_path) return -1;
    if (nanlm_shard2_read(in_path, &rec, &n) != 0) return -1;
    for (i = 0; i < n; i++) {
        rec[i].count = (int32_t)(0u - (uint32_t)rec[i].count);
    }
    rc = nanlm_shard2_write(out_path, rec, n);
    free(rec);
    return rc;
}

int nanlm_shard2_stats(const char *path, uint64_t *keys, int64_t *total,
                       uint64_t *records, int64_t *min_count,
                       int64_t *max_count) {
    nanlm_shard2_rec *rec = NULL;
    size_t n = 0, i;
    uint64_t k = 0;
    int64_t sum = 0;
    int64_t lo = 0, hi = 0;
    if (!path) return -1;
    if (nanlm_shard2_read(path, &rec, &n) != 0) return -1;
    for (i = 0; i < n; i++) {
        sum += rec[i].count;
        if (i == 0) { lo = hi = rec[i].count; }
        if (rec[i].count < lo) lo = rec[i].count;
        if (rec[i].count > hi) hi = rec[i].count;
        if (i == 0 || rec[i].key != rec[i - 1].key) k++;
    }
    if (keys) *keys = k;
    if (total) *total = sum;
    if (records) *records = n;
    if (min_count) *min_count = lo;
    if (max_count) *max_count = hi;
    free(rec);
    return 0;
}
