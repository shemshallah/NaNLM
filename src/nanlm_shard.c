#include "nanlm_shard.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NSH_MAGIC 0x3148534Eu /* "NSH1" little-endian */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t nrec;
    uint32_t reserved;
} nsh_header;

static int cmp_packed(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int cmp_rec(const void *a, const void *b) {
    const nanlm_shard_rec *x = (const nanlm_shard_rec *)a;
    const nanlm_shard_rec *y = (const nanlm_shard_rec *)b;
    if (x->key != y->key) return (x->key > y->key) - (x->key < y->key);
    return (int)x->next - (int)y->next;
}

int nanlm_shard_build(const uint8_t *data, size_t start, size_t end,
                      nanlm_shard_rec **out_rec, size_t *out_n) {
    uint32_t *obs;
    size_t n = 0, cap, i, k = 0;
    nanlm_shard_rec *rec;

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
    rec = (nanlm_shard_rec *)calloc(n, sizeof(*rec));
    if (!rec) { free(obs); return -1; }
    for (i = 0; i < n;) {
        uint32_t key = obs[i] >> 8;
        size_t j = i;
        while (j < n && (obs[j] >> 8) == key) {
            uint32_t b = obs[j] & 0xFFu;
            uint32_t cnt = 0;
            while (j < n && (obs[j] >> 8) == key &&
                   (obs[j] & 0xFFu) == b) {
                cnt++;
                j++;
            }
            rec[k].key = key;
            rec[k].next = (uint8_t)b;
            rec[k].count = (uint16_t)(cnt > 65535 ? 65535 : cnt);
            k++;
        }
        i = j;
    }
    free(obs);
    *out_rec = rec;
    *out_n = k;
    return 0;
}

int nanlm_shard_write(const char *path, const nanlm_shard_rec *rec, size_t n) {
    FILE *f;
    nsh_header h;
    if (!path || (!rec && n)) return -1;
    f = fopen(path, "wb");
    if (!f) return -1;
    h.magic = NSH_MAGIC;
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

int nanlm_shard_read(const char *path, nanlm_shard_rec **out_rec, size_t *out_n) {
    FILE *f;
    nsh_header h;
    nanlm_shard_rec *rec;
    if (!path || !out_rec || !out_n) return -1;
    f = fopen(path, "rb");
    if (!f) return -1;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != NSH_MAGIC ||
        h.version != 1) {
        fclose(f);
        return -1;
    }
    rec = (nanlm_shard_rec *)calloc(h.nrec ? h.nrec : 1, sizeof(*rec));
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

int nanlm_shard_merge(const char *out_path, const char **in_paths, int n_in) {
    nanlm_shard_rec *all = NULL;
    size_t n = 0, i;
    int k;
    if (!out_path || !in_paths || n_in <= 0) return -1;
    for (k = 0; k < n_in; k++) {
        nanlm_shard_rec *r = NULL;
        size_t rn = 0;
        if (nanlm_shard_read(in_paths[k], &r, &rn) != 0) {
            free(all);
            return -1;
        }
        {
            nanlm_shard_rec *tmp = (nanlm_shard_rec *)realloc(
                all, (n + rn) * sizeof(*all));
            if (!tmp) { free(r); free(all); return -1; }
            all = tmp;
        }
        memcpy(all + n, r, rn * sizeof(*r));
        n += rn;
        free(r);
    }
    qsort(all, n, sizeof(*all), cmp_rec);
    /* Compress runs, summing counts. */
    {
        size_t w = 0;
        for (i = 0; i < n;) {
            uint32_t key = all[i].key;
            uint8_t next = all[i].next;
            uint32_t sum = 0;
            while (i < n && all[i].key == key && all[i].next == next) {
                sum += all[i].count;
                i++;
            }
            all[w].key = key;
            all[w].next = next;
            all[w].count = (uint16_t)(sum > 65535 ? 65535 : sum);
            w++;
        }
        n = w;
    }
    k = nanlm_shard_write(out_path, all, n);
    free(all);
    return k;
}

int nanlm_shard_stats(const char *path, uint64_t *keys, uint64_t *total,
                      uint64_t *records) {
    nanlm_shard_rec *rec = NULL;
    size_t n = 0, i;
    uint64_t k = 0, sum = 0;
    if (!path) return -1;
    if (nanlm_shard_read(path, &rec, &n) != 0) return -1;
    for (i = 0; i < n; i++) {
        sum += rec[i].count;
        if (i == 0 || rec[i].key != rec[i - 1].key) k++;
    }
    if (keys) *keys = k;
    if (total) *total = sum;
    if (records) *records = n;
    free(rec);
    return 0;
}
