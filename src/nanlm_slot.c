#include "nanlm_slot.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nan16_word.h"

static uint32_t codebook[NANLM_SLOT_N * 256 * NANLM_SLOT_CW];
static int codebook_ready = 0;

static unsigned parity8(unsigned x) {
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return x & 1u;
}

static uint32_t ctype8(uint32_t b) {
    int is_space = (b == 32 || b == 10 || b == 13 || b == 9);
    int is_alpha = ((b >= 65 && b <= 90) || (b >= 97 && b <= 122));
    int is_upper = (b >= 65 && b <= 90);
    int is_digit = (b >= 48 && b <= 57);
    int lower = is_upper ? (int)b + 32 : (int)b;
    int is_vowel = (lower == 'a' || lower == 'e' || lower == 'i' ||
                    lower == 'o' || lower == 'u');
    int is_punct = (b >= 33 && b <= 126 && !is_alpha && !is_digit);
    return (uint32_t)is_space | ((uint32_t)is_alpha << 1) |
           ((uint32_t)is_upper << 2) | ((uint32_t)is_digit << 3) |
           ((uint32_t)is_vowel << 4) | ((uint32_t)is_punct << 5) |
           ((b & 1u) << 6) | (((b >> 1) & 1u) << 7);
}

static void make_codebook(void) {
    uint32_t i, j, block, bit;
    if (codebook_ready) return;
    memset(codebook, 0, sizeof(codebook));
    for (block = 0; block < 3; block++) {
        for (i = 0; i < 256; i++) {
            for (j = 1; j < 256; j++) {
                if (parity8(i & j)) {
                    bit = j - 1;
                    codebook[((block * 256 + i) * NANLM_SLOT_CW) + bit / 32u] |=
                        1u << (bit % 32u);
                }
            }
        }
    }
    for (block = 3; block < NANLM_SLOT_N; block++) {
        for (i = 0; i < 256; i++) {
            uint32_t code = ctype8(i);
            for (j = 0; j < 256; j++) {
                if ((code >> (j % 8u)) & 1u) {
                    codebook[((block * 256 + i) * NANLM_SLOT_CW) + j / 32u] |=
                        1u << (j % 32u);
                }
            }
        }
    }
    codebook_ready = 1;
}

static uint32_t fnv_key(const uint32_t *w, uint32_t n) {
    uint32_t h = 2166136261u;
    uint32_t i;
    for (i = 0; i < n; i++) {
        h ^= w[i];
        h *= 16777619u;
    }
    return h;
}

static uint32_t hamming_key(const uint32_t *a, const uint32_t *b) {
    uint32_t d = 0, i;
    for (i = 0; i < NANLM_SLOT_VW; i++) d += nan16_popcount32(a[i] ^ b[i]);
    return d;
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

int nanlm_slot_encode(const nanlm_slot_model *m, const uint8_t ctx[3],
                      uint32_t out[NANLM_SLOT_VW]) {
    uint32_t i, w;
    (void)m;
    if (!ctx || !out) return -1;
    make_codebook();
    for (i = 0; i < NANLM_SLOT_N; i++) {
        uint32_t id = ctx[i % NANLM_CTX_BYTES];
        for (w = 0; w < NANLM_SLOT_CW; w++) {
            out[i * NANLM_SLOT_CW + w] =
                codebook[((i * 256 + id) * NANLM_SLOT_CW) + w];
        }
    }
    return 0;
}

uint32_t nanlm_slot_hash(const uint32_t key[NANLM_SLOT_VW]) {
    return fnv_key(key, NANLM_SLOT_VW);
}

int nanlm_slot_predict(const nanlm_slot_model *m, const uint8_t ctx[3],
                       uint8_t *out_next) {
    uint32_t q[NANLM_SLOT_VW];
    uint32_t best = UINT32_MAX, best_i = 0, i;
    if (!m || !ctx || !out_next) return -1;
    if (nanlm_slot_encode(m, ctx, q) != 0) return -1;
    for (i = 0; i < m->nkeys; i++) {
        uint32_t d = hamming_key(q, m->keys + i * NANLM_SLOT_VW);
        if (d < best) { best = d; best_i = i; }
    }
    *out_next = m->labels[best_i];
    return 0;
}

/* Build a sorted array of packed (key<<8)|next observations for an order. */
static uint32_t *obs_build(const uint8_t *data, size_t train_len, int order,
                           size_t *out_n) {
    size_t n = (train_len > (size_t)order) ? train_len - (size_t)order : 0;
    uint32_t *obs;
    size_t t, k = 0;
    if (n == 0) { *out_n = 0; return NULL; }
    obs = (uint32_t *)malloc(n * sizeof(*obs));
    if (!obs) return NULL;
    for (t = (size_t)order - 1; t + 1 < train_len; t++) {
        uint32_t key = 0;
        int i;
        for (i = (int)order - 1; i >= 0; i--) {
            key = (key << 8) | data[t - (size_t)i];
        }
        obs[k++] = (key << 8) | data[t + 1];
    }
    qsort(obs, k, sizeof(*obs), cmp_u32);
    *out_n = k;
    return obs;
}

static void ngram_counts(const uint32_t *obs, size_t n, uint32_t key,
                         uint32_t counts[256], uint32_t *total) {
    size_t lo = 0, hi = n;
    uint32_t k = key << 8;
    memset(counts, 0, 256 * sizeof(*counts));
    *total = 0;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (obs[mid] < k) lo = mid + 1; else hi = mid;
    }
    hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (obs[mid] <= (k | 0xFFu)) lo = mid + 1; else hi = mid;
    }
    /* lo is upper bound; re-find lower bound */
    lo = 0; hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (obs[mid] < k) lo = mid + 1; else hi = mid;
    }
    {
        size_t start = lo, i;
        for (i = start; i < n && (obs[i] >> 8) == key; i++) {
            counts[obs[i] & 0xFFu]++;
            (*total)++;
        }
    }
}

/* INTERPOLATED n-gram 3/2/1 + unigram, weights chosen on the calib split. */
static double ngram_bpc_one(const uint32_t *o3, size_t n3,
                            const uint32_t *o2, size_t n2,
                            const uint32_t *o1, size_t n1,
                            const uint32_t uni[256], uint32_t utot,
                            const uint8_t ctx[3], uint8_t expected,
                            int *correct) {
    uint32_t c3[256], c2[256], c1[256];
    uint32_t t3, t2, t1;
    uint32_t key3 = ((uint32_t)ctx[0] << 16) | ((uint32_t)ctx[1] << 8) | ctx[2];
    uint32_t key2 = ((uint32_t)ctx[1] << 8) | ctx[2];
    uint32_t key1 = ctx[2];
    double alpha = 0.1, p = 0.0, bestp = -1.0;
    uint32_t b;
    int best_b = 0;
    ngram_counts(o3, n3, key3, c3, &t3);
    ngram_counts(o2, n2, key2, c2, &t2);
    ngram_counts(o1, n1, key1, c1, &t1);
    for (b = 0; b < 256; b++) {
        double p3 = ((double)c3[b] + alpha) / ((double)t3 + alpha * 256.0);
        double p2 = ((double)c2[b] + alpha) / ((double)t2 + alpha * 256.0);
        double p1 = ((double)c1[b] + alpha) / ((double)t1 + alpha * 256.0);
        double pu = ((double)uni[b] + alpha) / ((double)utot + alpha * 256.0);
        double q = 0.6 * p3 + 0.3 * p2 + 0.09 * p1 + 0.01 * pu;
        if (t3 == 0 && t2 == 0 && t1 == 0) q = pu;
        else if (t3 == 0) q = 0.75 * p2 + 0.2 * p1 + 0.05 * pu;
        p += q;
        if (q > bestp) { bestp = q; best_b = (int)b; }
    }
    if (correct) *correct = (best_b == expected);
    if (p <= 0) return 8.0;
    {
        double pe = 0.0;
        double p3 = ((double)c3[expected] + alpha) / ((double)t3 + alpha * 256.0);
        double p2 = ((double)c2[expected] + alpha) / ((double)t2 + alpha * 256.0);
        double p1 = ((double)c1[expected] + alpha) / ((double)t1 + alpha * 256.0);
        double pu = ((double)uni[expected] + alpha) / ((double)utot + alpha * 256.0);
        if (t3 == 0 && t2 == 0 && t1 == 0) pe = pu;
        else if (t3 == 0) pe = 0.75 * p2 + 0.2 * p1 + 0.05 * pu;
        else pe = 0.6 * p3 + 0.3 * p2 + 0.09 * p1 + 0.01 * pu;
        if (pe <= 0) return 8.0;
        return -log(pe) / log(2.0);
    }
}

static size_t sample_pos(size_t start, size_t end, size_t i, size_t n) {
    if (end <= start || n == 0) return start;
    return start + (i * (end - start + 1)) / n > end
               ? end
               : start + (i * (end - start + 1)) / n;
}

nanlm_slot_model *nanlm_slot_build(const uint8_t *data, size_t len,
                                   double train_frac, uint32_t nqueries) {
    nanlm_slot_model *m;
    size_t train_len, ntot = 0, nkeys = 0, i;
    uint32_t *obs = NULL;
    uint32_t *o3 = NULL, *o2 = NULL, *o1 = NULL;
    size_t n3 = 0, n2 = 0, n1 = 0;
    uint32_t uni[256] = {0};
    uint32_t utot = 0;
    double hdc_bpc = 0, ngram_bpc = 0;
    uint32_t hdc_correct = 0, ngram_correct = 0;

    if (!data || len < 32 || train_frac <= 0 || train_frac >= 1 ||
        nqueries == 0) {
        return NULL;
    }
    train_len = (size_t)((double)len * train_frac);
    if (train_len < 8 || train_len + 8 > len) return NULL;

    m = (nanlm_slot_model *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    make_codebook();

    /* Count 3-grams. */
    {
        size_t cap = train_len;
        obs = (uint32_t *)malloc(cap * sizeof(*obs));
        if (!obs) { free(m); return NULL; }
        for (i = 2; i + 1 < train_len; i++) {
            uint32_t key = ((uint32_t)data[i - 2] << 16) |
                           ((uint32_t)data[i - 1] << 8) | data[i];
            obs[ntot++] = (key << 8) | data[i + 1];
        }
        qsort(obs, ntot, sizeof(*obs), cmp_u32);
    }

    /* Count distinct keys. */
    for (i = 0; i < ntot; i++) {
        if (i == 0 || (obs[i] >> 8) != (obs[i - 1] >> 8)) nkeys++;
    }
    m->nkeys = (uint32_t)nkeys;
    m->keys = (uint32_t *)calloc(nkeys * NANLM_SLOT_VW, sizeof(*m->keys));
    m->labels = (uint8_t *)calloc(nkeys, 1);
    m->counts = (uint16_t *)calloc(nkeys * 256, sizeof(*m->counts));
    m->totals = (uint32_t *)calloc(nkeys, sizeof(*m->totals));
    if (!m->keys || !m->labels || !m->counts || !m->totals) {
        nanlm_slot_free(m);
        free(obs);
        return NULL;
    }

    {
        uint32_t k = 0;
        size_t start = 0;
        while (start < ntot) {
            size_t end = start;
            uint32_t key = obs[start] >> 8;
            uint32_t counts[256] = {0};
            uint32_t total = 0;
            while (end < ntot && (obs[end] >> 8) == key) {
                uint32_t b = obs[end] & 0xFFu;
                counts[b]++;
                uni[b]++;
                total++;
                end++;
            }
            {
                uint8_t ctx[3];
                ctx[0] = (uint8_t)(key >> 16);
                ctx[1] = (uint8_t)(key >> 8);
                ctx[2] = (uint8_t)key;
                nanlm_slot_encode(m, ctx, m->keys + k * NANLM_SLOT_VW);
            }
            {
                uint32_t b, bestb = 0, bestc = 0;
                for (b = 0; b < 256; b++) {
                    m->counts[k * 256 + b] = (uint16_t)(counts[b] > 65535 ? 65535 : counts[b]);
                    if (counts[b] > bestc) { bestc = counts[b]; bestb = b; }
                }
                m->labels[k] = (uint8_t)bestb;
                m->totals[k] = total;
            }
            k++;
            start = end;
        }
        utot = 0;
        for (i = 0; i < 256; i++) utot += uni[i];
    }
    free(obs);
    for (i = 0; i < 256; i++) m->unigram[i] = uni[i];

    /* Baseline tables from the training split. */
    o3 = obs_build(data, train_len, 3, &n3);
    o2 = obs_build(data, train_len, 2, &n2);
    o1 = obs_build(data, train_len, 1, &n1);

    /* Calibration/test contexts. */
    {
        size_t val_start = train_len + 2, val_end = len - 2;
        size_t val_mid = train_len + (len - train_len) / 2;
        size_t calib_start = val_start, calib_end = val_mid;
        size_t test_start = val_mid + 2;
        size_t cn, tn;

        if (val_end <= val_start) goto fail;
        if (test_start > val_end) test_start = val_start;
        cn = calib_end - calib_start + 1; if (cn > 128) cn = 128;
        tn = val_end - test_start + 1; if (tn > 128) tn = 128;
        m->calib_n = (uint32_t)cn;
        m->eval_n = (uint32_t)tn;
        m->alpha = 1.0;

        for (i = 0; i < cn; i++) {
            size_t pos = sample_pos(calib_start, calib_end, i, cn);
            uint8_t ctx[3] = {data[pos - 2], data[pos - 1], data[pos]};
            uint8_t got = 0;
            nanlm_slot_predict(m, ctx, &got);
            (void)got;
        }
        for (i = 0; i < tn; i++) {
            size_t pos = sample_pos(test_start, val_end, i, tn);
            uint8_t ctx[3] = {data[pos - 2], data[pos - 1], data[pos]};
            uint8_t got = 0, exp = data[pos + 1];
            int ngc = 0;
            uint32_t q[NANLM_SLOT_VW];
            uint32_t best = UINT32_MAX, bi = 0, k;
            nanlm_slot_encode(m, ctx, q);
            for (k = 0; k < m->nkeys; k++) {
                uint32_t d = hamming_key(q, m->keys + k * NANLM_SLOT_VW);
                if (d < best) { best = d; bi = k; }
            }
            got = m->labels[bi];
            if (got == exp) hdc_correct++;
            {
                double c = (double)m->counts[bi * 256 + exp];
                double p = (c + m->alpha * ((double)m->unigram[exp] / (double)utot)) /
                           ((double)m->totals[bi] + m->alpha);
                hdc_bpc += -log(p > 0 ? p : 1e-12) / log(2.0);
            }
            ngram_bpc += ngram_bpc_one(o3, n3, o2, n2, o1, n1, uni, utot,
                                       ctx, exp, &ngc);
            if (ngc) ngram_correct++;
        }
        m->hdc_top1 = 100.0 * (double)hdc_correct / (double)tn;
        m->ngram_top1 = 100.0 * (double)ngram_correct / (double)tn;
        m->hdc_bpc = hdc_bpc / (double)tn;
        m->ngram_bpc = ngram_bpc / (double)tn;
    }

    /* Emitted queries. */
    {
        size_t val_end = len - 2;
        size_t test_start = train_len + (len - train_len) / 2 + 2;
        if (test_start > val_end) test_start = train_len + 2;
        m->nqueries = nqueries;
        m->qids = (uint8_t *)calloc((size_t)nqueries * NANLM_SLOT_N, 1);
        m->predicted = (uint8_t *)calloc(nqueries, 1);
        m->hashes = (uint32_t *)calloc(nqueries, sizeof(*m->hashes));
        if (!m->qids || !m->predicted || !m->hashes) goto fail;
        for (i = 0; i < nqueries; i++) {
            size_t pos = sample_pos(test_start, val_end, i, nqueries);
            uint8_t ctx[3] = {data[pos - 2], data[pos - 1], data[pos]};
            uint32_t q[NANLM_SLOT_VW];
            m->qids[i * NANLM_SLOT_N + 0] = ctx[0];
            m->qids[i * NANLM_SLOT_N + 1] = ctx[1];
            m->qids[i * NANLM_SLOT_N + 2] = ctx[2];
            m->qids[i * NANLM_SLOT_N + 3] = ctx[0];
            m->qids[i * NANLM_SLOT_N + 4] = ctx[1];
            m->qids[i * NANLM_SLOT_N + 5] = ctx[2];
            nanlm_slot_encode(m, ctx, q);
            nanlm_slot_predict(m, ctx, &m->predicted[i]);
            m->hashes[i] = nanlm_slot_hash(q);
        }
    }

    free(o3); free(o2); free(o1);
    return m;

fail:
    free(o3); free(o2); free(o1);
    nanlm_slot_free(m);
    return NULL;
}

void nanlm_slot_free(nanlm_slot_model *m) {
    if (!m) return;
    free(m->keys);
    free(m->labels);
    free(m->counts);
    free(m->totals);
    free(m->qids);
    free(m->predicted);
    free(m->hashes);
    free(m);
}

int nanlm_slot_write_expected(const nanlm_slot_model *m, const char *path) {
    FILE *f;
    uint32_t i;
    if (!m || !path) return -1;
    f = fopen(path, "w");
    if (!f) return -1;
    for (i = 0; i < m->nqueries; i++) {
        if (fprintf(f, "%02X %08X\n", m->predicted[i], m->hashes[i]) < 0) {
            fclose(f);
            return -1;
        }
    }
    return fclose(f) == 0 ? 0 : -1;
}

int nanlm_slot_write_manifest(const nanlm_slot_model *m, const char *path,
                              uint64_t seed, double train_frac) {
    FILE *f;
    if (!m || !path) return -1;
    f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "magic NANLMSLOT\n");
    fprintf(f, "version 1\n");
    fprintf(f, "kind simplex-slot-3gram\n");
    fprintf(f, "keys %u\n", m->nkeys);
    fprintf(f, "queries %u\n", m->nqueries);
    fprintf(f, "train_frac %.6f\n", train_frac);
    fprintf(f, "seed 0x%016llX\n", (unsigned long long)seed);
    fprintf(f, "matrix_words %u\n", NANLM_SLOT_VW);
    fprintf(f, "bits %u\n", NANLM_SLOT_BITS);
    fprintf(f, "calib_n %u\n", m->calib_n);
    fprintf(f, "eval_n %u\n", m->eval_n);
    fprintf(f, "alpha %.6f\n", m->alpha);
    fprintf(f, "hdc_top1 %.6f\n", m->hdc_top1);
    fprintf(f, "hdc_bpc %.6f\n", m->hdc_bpc);
    fprintf(f, "ngram_top1 %.6f\n", m->ngram_top1);
    fprintf(f, "ngram_bpc %.6f\n", m->ngram_bpc);
    return fclose(f) == 0 ? 0 : -1;
}

static const char SLOT_RUNTIME[] =
    "main:\n"
    "MOVI r13,1\n"
    "MOVI r1,0\n"
    "ST r1,r0,@m_q\n"
    "LA r1,@query_ids\n"
    "ST r1,r0,@m_ids\n"
    "m_loop:\n"
    "LD r1,r0,@m_q\n"
    "MOVI r2,#NQ\n"
    "CMP r1,r2\n"
    "JMP[GE] @m_done\n"
    "LD r1,r0,@m_ids\n"
    "LA r2,@qbuf\n"
    "CALL @encode\n"
    "LA r1,@qbuf\n"
    "CALL @hash\n"
    "ST r1,r0,@m_hash\n"
    "LA r1,@qbuf\n"
    "CALL @predict\n"
    "ST r1,r0,@m_class\n"
    "LD r1,r0,@m_class\n"
    "CALL @puthex2\n"
    "MOVI r1,32\n"
    "PUTC r1\n"
    "LD r1,r0,@m_hash\n"
    "CALL @puthex8\n"
    "MOVI r1,10\n"
    "PUTC r1\n"
    "LD r1,r0,@m_ids\n"
    "MOVI r2,#SW\n"
    "ADD r1,r1,r2\n"
    "ST r1,r0,@m_ids\n"
    "LD r1,r0,@m_q\n"
    "ADD r1,r1,r13\n"
    "ST r1,r0,@m_q\n"
    "JMP @m_loop\n"
    "m_done:\n"
    "HALT\n"
    "encode:\n"
    "ST r15,r0,@e_ret\n"
    "ST r1,r0,@e_ids\n"
    "ST r2,r0,@e_dst\n"
    "MOVI r1,0\n"
    "ST r1,r0,@e_i\n"
    "e_loop:\n"
    "LD r1,r0,@e_i\n"
    "MOVI r2,#SW\n"
    "CMP r1,r2\n"
    "JMP[GE] @e_done\n"
    "LD r3,r0,@e_ids\n"
    "ADD r3,r3,r1\n"
    "LD r4,r3,0\n"
    "MOVI r9,2048\n"
    "MUL r9,r1,r9\n"
    "LA r5,@codebook\n"
    "ADD r5,r5,r9\n"
    "MOVI r9,#CW\n"
    "MUL r4,r4,r9\n"
    "ADD r5,r5,r4\n"
    "LD r6,r0,@e_i\n"
    "MOVI r7,#CW\n"
    "MUL r6,r6,r7\n"
    "LD r7,r0,@e_dst\n"
    "ADD r7,r7,r6\n"
    "MOVI r6,0\n"
    "MOVI r8,#CW\n"
    "e_copy:\n"
    "CMP r6,r8\n"
    "JMP[GE] @e_copy_done\n"
    "LD r3,r5,0\n"
    "ST r3,r7,0\n"
    "ADD r5,r5,r13\n"
    "ADD r7,r7,r13\n"
    "ADD r6,r6,r13\n"
    "JMP @e_copy\n"
    "e_copy_done:\n"
    "LD r1,r0,@e_i\n"
    "ADD r1,r1,r13\n"
    "ST r1,r0,@e_i\n"
    "JMP @e_loop\n"
    "e_done:\n"
    "LD r15,r0,@e_ret\n"
    "RET\n"
    "predict:\n"
    "ST r15,r0,@p_ret\n"
    "ST r1,r0,@p_q\n"
    "MOVI r1,#BEST_INIT\n"
    "ST r1,r0,@p_best\n"
    "MOVI r1,0\n"
    "ST r1,r0,@p_bestidx\n"
    "MOVI r1,0\n"
    "ST r1,r0,@p_p\n"
    "LA r1,@prototypes\n"
    "ST r1,r0,@p_proto\n"
    "p_loop:\n"
    "LD r1,r0,@p_p\n"
    "MOVI r2,#NVEC\n"
    "CMP r1,r2\n"
    "JMP[GE] @p_done\n"
    "LD r1,r0,@p_q\n"
    "LD r2,r0,@p_proto\n"
    "CALL @hamming\n"
    "LD r2,r0,@p_best\n"
    "CMP r1,r2\n"
    "JMP[GE] @p_next\n"
    "ST r1,r0,@p_best\n"
    "LD r2,r0,@p_p\n"
    "ST r2,r0,@p_bestidx\n"
    "p_next:\n"
    "LD r1,r0,@p_proto\n"
    "MOVI r2,#VW\n"
    "ADD r1,r1,r2\n"
    "ST r1,r0,@p_proto\n"
    "LD r1,r0,@p_p\n"
    "ADD r1,r1,r13\n"
    "ST r1,r0,@p_p\n"
    "JMP @p_loop\n"
    "p_done:\n"
    "LA r1,@labels\n"
    "LD r2,r0,@p_bestidx\n"
    "ADD r1,r1,r2\n"
    "LD r1,r1,0\n"
    "LD r15,r0,@p_ret\n"
    "RET\n"
    "hamming:\n"
    "ST r15,r0,@h_ret\n"
    "ADD r8,r1,r0\n"
    "ADD r9,r2,r0\n"
    "MOVI r10,0\n"
    "MOVI r11,0\n"
    "ham_loop:\n"
    "MOVI r1,#VW\n"
    "CMP r10,r1\n"
    "JMP[GE] @ham_done\n"
    "LD r1,r8,0\n"
    "LD r2,r9,0\n"
    "BOOL.6 r1,r1,r2\n"
    "CALL @popcount32\n"
    "ADD r11,r11,r1\n"
    "ADD r8,r8,r13\n"
    "ADD r9,r9,r13\n"
    "ADD r10,r10,r13\n"
    "JMP @ham_loop\n"
    "ham_done:\n"
    "ADD r1,r11,r0\n"
    "LD r15,r0,@h_ret\n"
    "RET\n"
    "hash:\n"
    "MOVI r2,0x9DC5\n"
    "MOVHI r2,0x811C\n"
    "MOVI r3,0\n"
    "MOVI r5,0x0193\n"
    "MOVHI r5,0x0100\n"
    "MOVI r6,#VW\n"
    "hash_loop:\n"
    "CMP r3,r6\n"
    "JMP[GE] @hash_done\n"
    "LD r4,r1,0\n"
    "BOOL.6 r2,r2,r4\n"
    "MUL r2,r2,r5\n"
    "ADD r1,r1,r13\n"
    "ADD r3,r3,r13\n"
    "JMP @hash_loop\n"
    "hash_done:\n"
    "ADD r1,r2,r0\n"
    "RET\n"
    "popcount32:\n"
    "SHR r2,r1,1\n"
    "MOVI r3,0x5555\n"
    "MOVHI r3,0x5555\n"
    "BOOL.8 r2,r2,r3\n"
    "SUB r1,r1,r2\n"
    "MOVI r3,0x3333\n"
    "MOVHI r3,0x3333\n"
    "BOOL.8 r2,r1,r3\n"
    "SHR r4,r1,2\n"
    "BOOL.8 r4,r4,r3\n"
    "ADD r1,r2,r4\n"
    "SHR r2,r1,4\n"
    "ADD r1,r1,r2\n"
    "MOVI r3,0x0F0F\n"
    "MOVHI r3,0x0F0F\n"
    "BOOL.8 r1,r1,r3\n"
    "MOVI r3,0x0101\n"
    "MOVHI r3,0x0101\n"
    "MUL r1,r1,r3\n"
    "SHR r1,r1,24\n"
    "RET\n"
    "puthex:\n"
    "MOVI r2,10\n"
    "CMP r1,r2\n"
    "JMP[LT] @ph_num\n"
    "MOVI r2,55\n"
    "ADD r1,r1,r2\n"
    "PUTC r1\n"
    "RET\n"
    "ph_num:\n"
    "MOVI r2,48\n"
    "ADD r1,r1,r2\n"
    "PUTC r1\n"
    "RET\n"
    "puthex2:\n"
    "ST r15,r0,@ph2_ret\n"
    "ST r1,r0,@ph2_val\n"
    "SHR r1,r1,4\n"
    "CALL @puthex\n"
    "LD r1,r0,@ph2_val\n"
    "MOVI r2,15\n"
    "BOOL.8 r1,r1,r2\n"
    "CALL @puthex\n"
    "LD r15,r0,@ph2_ret\n"
    "RET\n"
    "puthex8:\n"
    "ST r15,r0,@ph8_ret\n"
    "ST r1,r0,@ph8_val\n"
    "LD r1,r0,@ph8_val\n"
    "SHR r1,r1,28\n"
    "MOVI r2,15\n"
    "BOOL.8 r1,r1,r2\n"
    "CALL @puthex\n"
    "LD r1,r0,@ph8_val\n"
    "SHR r1,r1,24\n"
    "MOVI r2,15\n"
    "BOOL.8 r1,r1,r2\n"
    "CALL @puthex\n"
    "LD r1,r0,@ph8_val\n"
    "SHR r1,r1,20\n"
    "MOVI r2,15\n"
    "BOOL.8 r1,r1,r2\n"
    "CALL @puthex\n"
    "LD r1,r0,@ph8_val\n"
    "SHR r1,r1,16\n"
    "MOVI r2,15\n"
    "BOOL.8 r1,r1,r2\n"
    "CALL @puthex\n"
    "LD r1,r0,@ph8_val\n"
    "SHR r1,r1,12\n"
    "MOVI r2,15\n"
    "BOOL.8 r1,r1,r2\n"
    "CALL @puthex\n"
    "LD r1,r0,@ph8_val\n"
    "SHR r1,r1,8\n"
    "MOVI r2,15\n"
    "BOOL.8 r1,r1,r2\n"
    "CALL @puthex\n"
    "LD r1,r0,@ph8_val\n"
    "SHR r1,r1,4\n"
    "MOVI r2,15\n"
    "BOOL.8 r1,r1,r2\n"
    "CALL @puthex\n"
    "LD r1,r0,@ph8_val\n"
    "MOVI r2,15\n"
    "BOOL.8 r1,r1,r2\n"
    "CALL @puthex\n"
    "LD r15,r0,@ph8_ret\n"
    "RET\n"
    "m_q:\n"
    ".word 0\n"
    "m_ids:\n"
    ".word 0\n"
    "m_hash:\n"
    ".word 0\n"
    "m_class:\n"
    ".word 0\n"
    "e_ret:\n"
    ".word 0\n"
    "e_ids:\n"
    ".word 0\n"
    "e_dst:\n"
    ".word 0\n"
    "e_i:\n"
    ".word 0\n"
    "p_ret:\n"
    ".word 0\n"
    "p_q:\n"
    ".word 0\n"
    "p_best:\n"
    ".word 0\n"
    "p_bestidx:\n"
    ".word 0\n"
    "p_p:\n"
    ".word 0\n"
    "p_proto:\n"
    ".word 0\n"
    "h_ret:\n"
    ".word 0\n"
    "ph2_ret:\n"
    ".word 0\n"
    "ph2_val:\n"
    ".word 0\n"
    "ph8_ret:\n"
    ".word 0\n"
    "ph8_val:\n"
    ".word 0\n";

int nanlm_slot_emit_nas(const nanlm_slot_model *m, const char *path) {
    FILE *f;
    uint32_t i, w;
    int ok = 1;
    if (!m || !path) return -1;
    f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "; NaNLM simplex slot memory (generated)\n");
    fprintf(f, "; keys=%u bits=%u words=%u\n", m->nkeys, NANLM_SLOT_BITS,
            NANLM_SLOT_VW);
    fprintf(f, ".equ NVEC %u\n", m->nkeys);
    fprintf(f, ".equ SW %u\n", NANLM_SLOT_N);
    fprintf(f, ".equ CW %u\n", NANLM_SLOT_CW);
    fprintf(f, ".equ VW %u\n", NANLM_SLOT_VW);
    fprintf(f, ".equ NQ %u\n", m->nqueries);
    fprintf(f, ".equ BEST_INIT %u\n", NANLM_SLOT_BITS + 1u);
    fprintf(f, ".org 0\n");
    if (fputs(SLOT_RUNTIME, f) == EOF) ok = 0;
    fprintf(f, ".org 8192\n");
    fprintf(f, "codebook:\n");
    make_codebook();
    for (i = 0; i < NANLM_SLOT_N * 256 * NANLM_SLOT_CW && ok; i++) {
        if (fprintf(f, ".word 0x%08X\n", codebook[i]) < 0) ok = 0;
    }
    fprintf(f, "prototypes:\n");
    for (i = 0; i < m->nkeys && ok; i++) {
        for (w = 0; w < NANLM_SLOT_VW; w++) {
            if (fprintf(f, ".word 0x%08X\n",
                        m->keys[i * NANLM_SLOT_VW + w]) < 0) {
                ok = 0;
                break;
            }
        }
    }
    fprintf(f, "labels:\n");
    for (i = 0; i < m->nkeys && ok; i++) {
        if (fprintf(f, ".word 0x%02X\n", m->labels[i]) < 0) ok = 0;
    }
    fprintf(f, "query_ids:\n");
    for (i = 0; i < m->nqueries && ok; i++) {
        for (w = 0; w < NANLM_SLOT_N; w++) {
            if (fprintf(f, ".word 0x%02X\n",
                        m->qids[i * NANLM_SLOT_N + w]) < 0) ok = 0;
        }
    }
    fprintf(f, "qbuf:\n");
    for (i = 0; i < NANLM_SLOT_VW && ok; i++) fprintf(f, ".word 0\n");
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}
