#include "nanlm_lm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanlm_runtime.h"

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

typedef struct {
    uint32_t key;
    uint8_t next;
} Pair;

static int pair_cmp(const void *pa, const void *pb) {
    const Pair *a = (const Pair *)pa;
    const Pair *b = (const Pair *)pb;
    if (a->key < b->key) return -1;
    if (a->key > b->key) return 1;
    return (int)a->next - (int)b->next;
}

static size_t lower_key(const Pair *p, size_t n, uint32_t key) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (p[mid].key < key) lo = mid + 1; else hi = mid;
    }
    return lo;
}

static size_t upper_key(const Pair *p, size_t n, uint32_t key) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (p[mid].key <= key) lo = mid + 1; else hi = mid;
    }
    return lo;
}

nanlm_lm_model *nanlm_lm_model_new(uint32_t dim, uint32_t vocab,
                                   uint32_t nqueries) {
    return nanlm_lm_model_new_clusters(dim, vocab, nqueries,
                                       NANLM_LM_CLUSTERS);
}

nanlm_lm_model *nanlm_lm_model_new_clusters(uint32_t dim, uint32_t vocab,
                                            uint32_t nqueries,
                                            uint32_t clusters) {
    nanlm_lm_model *m;
    uint32_t i, shift = 0;

    if (dim == 0 || (dim % 32u) != 0 || dim > 65504u || vocab == 0 ||
        vocab > 256u || nqueries == 0 || nqueries > 65535u ||
        clusters == 0 || clusters > NANLM_LM_MAX_CLUSTERS ||
        (clusters & (clusters - 1u)) != 0) {
        return NULL;
    }
    while ((1u << shift) < clusters) shift++;
    m = (nanlm_lm_model *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->dim = dim;
    m->words = dim / 32u;
    m->vocab = vocab;
    m->clusters = clusters;
    m->proto_shift = shift;
    m->nqueries = nqueries;
    m->codebook = (nanlm_vec **)calloc(vocab, sizeof(*m->codebook));
    m->positions = (nanlm_vec **)calloc(NANLM_LM_CTX, sizeof(*m->positions));
    m->prototypes = (nanlm_vec **)calloc(vocab * m->clusters,
                                         sizeof(*m->prototypes));
    m->queries = (nanlm_vec **)calloc(nqueries, sizeof(*m->queries));
    m->query_ids = (uint8_t *)calloc((size_t)nqueries * NANLM_LM_CTX, 1);
    m->expected = (uint8_t *)calloc(nqueries, 1);
    m->predicted = (uint8_t *)calloc(nqueries, 1);
    m->hashes = (uint32_t *)calloc(nqueries, sizeof(*m->hashes));
    for (i = 0; i < NANLM_LM_CTX; i++) m->tmp[i] = NULL;
    if (!m->codebook || !m->positions || !m->prototypes || !m->queries ||
        !m->query_ids || !m->expected || !m->predicted || !m->hashes) {
        nanlm_lm_model_free(m);
        return NULL;
    }
    for (i = 0; i < vocab; i++) {
        m->codebook[i] = nanlm_vec_new(m->words);
        if (!m->codebook[i]) {
            nanlm_lm_model_free(m);
            return NULL;
        }
    }
    for (i = 0; i < vocab * m->clusters; i++) {
        m->prototypes[i] = nanlm_vec_new(m->words);
        if (!m->prototypes[i]) {
            nanlm_lm_model_free(m);
            return NULL;
        }
        nanlm_vec_zero(m->prototypes[i]);
    }
    for (i = 0; i < NANLM_LM_CTX; i++) {
        m->positions[i] = nanlm_vec_new(m->words);
        m->tmp[i] = nanlm_vec_new(m->words);
        if (!m->positions[i] || !m->tmp[i]) {
            nanlm_lm_model_free(m);
            return NULL;
        }
    }
    for (i = 0; i < nqueries; i++) {
        m->queries[i] = nanlm_vec_new(m->words);
        if (!m->queries[i]) {
            nanlm_lm_model_free(m);
            return NULL;
        }
    }
    return m;
}

void nanlm_lm_model_free(nanlm_lm_model *m) {
    uint32_t i;
    if (!m) return;
    if (m->codebook) {
        for (i = 0; i < m->vocab; i++) nanlm_vec_free(m->codebook[i]);
        free(m->codebook);
    }
    if (m->positions) {
        for (i = 0; i < NANLM_LM_CTX; i++) nanlm_vec_free(m->positions[i]);
        free(m->positions);
    }
    if (m->prototypes) {
        for (i = 0; i < m->vocab * m->clusters; i++) {
            nanlm_vec_free(m->prototypes[i]);
        }
        free(m->prototypes);
    }
    if (m->queries) {
        for (i = 0; i < m->nqueries; i++) nanlm_vec_free(m->queries[i]);
        free(m->queries);
    }
    for (i = 0; i < NANLM_LM_CTX; i++) nanlm_vec_free(m->tmp[i]);
    free(m->query_ids);
    free(m->expected);
    free(m->predicted);
    free(m->hashes);
    free(m);
}

int nanlm_lm_encode(const nanlm_lm_model *m, const uint8_t ids[3],
                    nanlm_vec *out) {
    uint32_t i;
    if (!m || !ids || !out || out->words != m->words) return -1;
    for (i = 0; i < NANLM_LM_CTX; i++) {
        if (ids[i] >= m->vocab) return -1;
    }
    for (i = 0; i < NANLM_LM_CTX; i++) {
        if (nanlm_vec_bind(m->tmp[i], m->codebook[ids[i]], m->positions[i]) !=
            0) {
            return -1;
        }
    }
    return nanlm_vec_bundle3(out, m->tmp[0], m->tmp[1], m->tmp[2]);
}

int nanlm_lm_predict(const nanlm_lm_model *m, const nanlm_vec *query,
                     uint32_t *out_byte) {
    uint32_t c, best_c = 0;
    uint64_t best = UINT64_MAX;
    if (!m || !query || !out_byte || query->words != m->words) return -1;
    for (c = 0; c < m->vocab * m->clusters; c++) {
        uint64_t d = nanlm_vec_hamming(query, m->prototypes[c]);
        if (d == UINT64_MAX) return -1;
        if (d < best) { best = d; best_c = c; }
    }
    *out_byte = best_c >> m->proto_shift;
    return 0;
}

uint32_t nanlm_lm_hash(const nanlm_vec *v) {
    uint32_t h = 2166136261u;
    size_t i;
    if (!v || !v->w || !nanlm_vec_is_data(v)) return 0;
    for (i = 0; i < v->words; i++) {
        h ^= nan16_payload(v->w[i]);
        h *= 16777619u;
    }
    return h;
}

static uint32_t flat_dist(const nanlm_vec *a, const nanlm_vec *b) {
    uint32_t d = 0;
    size_t w;
    for (w = 0; w < a->words; w++) {
        d += nan16_popcount32(nan16_payload(a->w[w]) ^
                              nan16_payload(b->w[w]));
    }
    return d;
}

static double bpc_from_dist(const uint32_t *dist, uint32_t nproto,
                            uint32_t clusters, uint32_t expected, double T) {
    uint32_t nclass = nproto / clusters;
    double *logsum;
    double maxs = -1e300, sum = 0.0, logp;
    uint32_t c, k;
    if (nclass == 0 || expected >= nclass || clusters == 0) return 8.0;
    logsum = (double *)calloc(nclass, sizeof(*logsum));
    if (!logsum) return 8.0;
    for (c = 0; c < nclass; c++) {
        double mx = -1e300, ssum = 0.0;
        for (k = 0; k < clusters; k++) {
            double s = -(double)dist[c * clusters + k] / T;
            if (s > mx) mx = s;
        }
        for (k = 0; k < clusters; k++) {
            ssum += exp(-(double)dist[c * clusters + k] / T - mx);
        }
        logsum[c] = mx + log(ssum);
        if (logsum[c] > maxs) maxs = logsum[c];
    }
    for (c = 0; c < nclass; c++) sum += exp(logsum[c] - maxs);
    logp = (logsum[expected] - maxs - log(sum)) / log(2.0);
    free(logsum);
    return -logp;
}

static size_t sample_pos(size_t start, size_t end, size_t i, size_t n) {
    size_t pos;
    if (end == 0 || start > end || n == 0) return start;
    pos = start + (i * (end - start + 1)) / n;
    if (pos > end) pos = end;
    return pos;
}

static void ngram_one(const Pair *pairs, size_t npairs, uint32_t key,
                      uint8_t expected, uint32_t vocab, uint32_t *correct,
                      double *bpc) {
    size_t lo = lower_key(pairs, npairs, key);
    size_t hi = upper_key(pairs, npairs, key);
    uint32_t counts[256] = {0};
    uint32_t total = (uint32_t)(hi - lo);
    uint32_t best_c = 0, best_count = 0;
    uint32_t c;
    double alpha = 0.1, p;
    for (size_t j = lo; j < hi; j++) counts[pairs[j].next]++;
    for (c = 0; c < vocab; c++) {
        if (counts[c] > best_count) {
            best_count = counts[c];
            best_c = c;
        }
    }
    if (total > 0 && best_c == expected) (*correct)++;
    p = ((double)counts[expected] + alpha) /
        ((double)total + alpha * (double)vocab);
    *bpc += -log(p) / log(2.0);
}

static int nanlm_lm_build_kmeans(nanlm_lm_model *m, const uint8_t *data,
                                 size_t train_len) {
    size_t ntrain = (train_len >= 3) ? train_len - 3 : 0;
    size_t idx = 0, t;
    nan16_word *ctx = NULL;
    uint8_t *labels = NULL;
    uint32_t *cnt = NULL, *offs = NULL, *idxlist = NULL, *fill = NULL;
    nanlm_vec *q = NULL;
    uint32_t c, i, k, w, b, it;

    if (ntrain == 0) return -1;
    ctx = (nan16_word *)calloc(ntrain * m->words, sizeof(*ctx));
    labels = (uint8_t *)calloc(ntrain, 1);
    q = nanlm_vec_new(m->words);
    if (!ctx || !labels || !q) {
        free(ctx);
        free(labels);
        nanlm_vec_free(q);
        return -1;
    }
    for (t = 2; t + 1 < train_len; t++) {
        uint8_t ids[3];
        uint32_t cls = data[t + 1];
        ids[0] = data[t - 2];
        ids[1] = data[t - 1];
        ids[2] = data[t];
        if (cls >= m->vocab) continue;
        if (nanlm_lm_encode(m, ids, q) != 0) {
            free(ctx);
            free(labels);
            nanlm_vec_free(q);
            return -1;
        }
        memcpy(ctx + idx * m->words, q->w, m->words * sizeof(*q->w));
        labels[idx] = (uint8_t)cls;
        idx++;
    }
    ntrain = idx;
    if (ntrain < m->clusters) {
        free(ctx);
        free(labels);
        nanlm_vec_free(q);
        return -1;
    }

    cnt = (uint32_t *)calloc(m->vocab, sizeof(*cnt));
    offs = (uint32_t *)calloc(m->vocab + 1, sizeof(*offs));
    idxlist = (uint32_t *)calloc(ntrain, sizeof(*idxlist));
    fill = (uint32_t *)calloc(m->vocab, sizeof(*fill));
    if (!cnt || !offs || !idxlist || !fill) {
        free(ctx);
        free(labels);
        nanlm_vec_free(q);
        free(cnt);
        free(offs);
        free(idxlist);
        free(fill);
        return -1;
    }
    for (i = 0; i < ntrain; i++) cnt[labels[i]]++;
    for (c = 0; c < m->vocab; c++) offs[c + 1] = offs[c] + cnt[c];
    memcpy(fill, offs, m->vocab * sizeof(*fill));
    for (i = 0; i < ntrain; i++) idxlist[fill[labels[i]]++] = i;

    for (c = 0; c < m->vocab; c++) {
        uint32_t n = offs[c + 1] - offs[c];
        uint32_t K = m->clusters;
        uint32_t *cent = NULL, *ones = NULL, *tot = NULL;
        if (n == 0) continue;
        cent = (uint32_t *)calloc((size_t)K * m->words, sizeof(*cent));
        ones = (uint32_t *)calloc((size_t)K * m->dim, sizeof(*ones));
        tot = (uint32_t *)calloc(K, sizeof(*tot));
        if (!cent || !ones || !tot) {
            free(cent);
            free(ones);
            free(tot);
            free(ctx);
            free(labels);
            nanlm_vec_free(q);
            free(cnt);
            free(offs);
            free(idxlist);
            free(fill);
            return -1;
        }
        for (k = 0; k < K; k++) {
            uint32_t src = offs[c] + (uint32_t)(((uint64_t)k * n) / K);
            if (src >= offs[c + 1]) src = offs[c];
            for (w = 0; w < m->words; w++) {
                cent[(size_t)k * m->words + w] =
                    nan16_payload(ctx[(size_t)src * m->words + w]);
            }
        }
        for (it = 0; it < 4; it++) {
            memset(ones, 0, (size_t)K * m->dim * sizeof(*ones));
            memset(tot, 0, K * sizeof(*tot));
            for (i = offs[c]; i < offs[c + 1]; i++) {
                uint32_t bestk = 0;
                uint32_t bestd = UINT32_MAX;
                uint32_t row = idxlist[i];
                for (k = 0; k < K; k++) {
                    uint32_t d = 0;
                    for (w = 0; w < m->words; w++) {
                        uint32_t x = nan16_payload(ctx[(size_t)row * m->words + w]) ^
                                     cent[(size_t)k * m->words + w];
                        d += nan16_popcount32(x);
                    }
                    if (d < bestd) { bestd = d; bestk = k; }
                }
                tot[bestk]++;
                for (w = 0; w < m->words; w++) {
                    uint32_t x = nan16_payload(ctx[(size_t)row * m->words + w]);
                    for (b = 0; b < 32u; b++) {
                        if ((x >> b) & 1u) {
                            ones[(size_t)bestk * m->dim + w * 32u + b]++;
                        }
                    }
                }
            }
            for (k = 0; k < K; k++) {
                if (tot[k] == 0) continue;
                for (w = 0; w < m->words; w++) {
                    uint32_t p = 0;
                    for (b = 0; b < 32u; b++) {
                        size_t z = (size_t)k * m->dim + w * 32u + b;
                        if ((uint64_t)ones[z] * 2u > (uint64_t)tot[k]) {
                            p |= (1u << b);
                        }
                    }
                    cent[(size_t)k * m->words + w] = p;
                }
            }
        }
        for (k = 0; k < K; k++) {
            for (w = 0; w < m->words; w++) {
                m->prototypes[(size_t)c * K + k]->w[w] =
                    nan16_make(cent[(size_t)k * m->words + w], NAN16_N_DATA);
            }
        }
        free(cent);
        free(ones);
        free(tot);
    }

    free(ctx);
    free(labels);
    nanlm_vec_free(q);
    free(cnt);
    free(offs);
    free(idxlist);
    free(fill);
    return 0;
}

int nanlm_lm_train(nanlm_lm_model *m, const uint8_t *data, size_t len,
                   double train_frac, uint64_t seed) {
    uint64_t state = seed;
    size_t train_len, npos, t, qi;
    uint32_t c, w, b, i;
    uint32_t *ones = NULL;
    uint32_t *total = NULL;
    uint32_t *seen = NULL;
    uint64_t reservoir = seed ^ 0xD1B54A32D192ED03ull;
    nanlm_vec *q = NULL;
    Pair *pairs = NULL;
    size_t npairs = 0;
    uint32_t *calib_dist = NULL;
    uint8_t *calib_exp = NULL;
    uint32_t *dist = NULL;
    double T = 32.0;
    double hdc_bpc = 0.0, ngram_bpc = 0.0;
    uint32_t hdc_correct = 0, ngram_correct = 0;

    if (!m || !data || len < 32 || train_frac <= 0.0 || train_frac >= 1.0) {
        return -1;
    }
    train_len = (size_t)((double)len * train_frac);
    if (train_len < 8 || train_len + 8 > len) return -1;
    m->train_bytes = (uint32_t)train_len;
    m->val_bytes = (uint32_t)(len - train_len);

    for (i = 0; i < m->vocab; i++) nanlm_vec_random(m->codebook[i], &state);
    for (i = 0; i < NANLM_LM_CTX; i++) {
        nanlm_vec_random(m->positions[i], &state);
    }

    if (m->memory_mode == 2) {
        if (nanlm_lm_build_kmeans(m, data, train_len) != 0) return -1;
    } else {
    if (m->memory_mode == 1) {
        seen = (uint32_t *)calloc(m->vocab, sizeof(*seen));
        if (!seen) return -1;
    } else {
        ones = (uint32_t *)calloc((size_t)m->vocab * m->clusters * m->dim,
                                  sizeof(*ones));
        total = (uint32_t *)calloc((size_t)m->vocab * m->clusters,
                                   sizeof(*total));
    }
    q = nanlm_vec_new(m->words);
    if ((m->memory_mode != 1 && (!ones || !total)) ||
        (m->memory_mode == 1 && !seen) || !q) {
        free(ones);
        free(total);
        free(seen);
        nanlm_vec_free(q);
        return -1;
    }

    for (t = 2; t + 1 < train_len; t++) {
        uint8_t ids[3];
        uint32_t cls = data[t + 1];
        ids[0] = data[t - 2];
        ids[1] = data[t - 1];
        ids[2] = data[t];
        if (cls >= m->vocab) continue;
        if (nanlm_lm_encode(m, ids, q) != 0) {
            free(ones);
            free(total);
            free(seen);
            nanlm_vec_free(q);
            return -1;
        }
        if (m->memory_mode == 1) {
            uint32_t slot;
            if (seen[cls] < m->clusters) {
                slot = seen[cls];
            } else {
                uint32_t j = (uint32_t)(splitmix64(&reservoir) % seen[cls]);
                slot = (j < m->clusters) ? j : UINT32_MAX;
            }
            if (slot != UINT32_MAX) {
                nanlm_vec *dst = m->prototypes[cls * m->clusters + slot];
                memcpy(dst->w, q->w, m->words * sizeof(*q->w));
            }
            seen[cls]++;
        } else {
            uint32_t pidx = cls * m->clusters +
                            (nanlm_lm_hash(q) & (m->clusters - 1u));
            total[pidx]++;
            for (w = 0; w < m->words; w++) {
                uint32_t p = nan16_payload(q->w[w]);
                for (b = 0; b < 32u; b++) {
                    if ((p >> b) & 1u) {
                        ones[(size_t)pidx * m->dim + w * 32u + b]++;
                    }
                }
            }
        }
    }

    if (m->memory_mode != 1) {
        for (c = 0; c < m->vocab * m->clusters; c++) {
            for (w = 0; w < m->words; w++) {
                uint32_t p = 0;
                for (b = 0; b < 32u; b++) {
                    size_t idx = (size_t)c * m->dim + w * 32u + b;
                    if ((uint64_t)ones[idx] * 2u > (uint64_t)total[c]) {
                        p |= (1u << b);
                    }
                }
                m->prototypes[c]->w[w] = nan16_make(p, NAN16_N_DATA);
            }
        }
    }

    }
    /* Order-3 byte n-gram baseline, built once from the training split. */
    npos = (train_len >= 3) ? train_len - 3 : 0;
    if (npos == 0) goto fail;
    pairs = (Pair *)calloc(npos, sizeof(*pairs));
    if (!pairs) goto fail;
    {
        size_t k = 0;
        for (t = 2; t + 1 < train_len; t++) {
            pairs[k].key = ((uint32_t)data[t - 2] << 16) |
                            ((uint32_t)data[t - 1] << 8) | data[t];
            pairs[k].next = data[t + 1];
            k++;
        }
        npairs = k;
        qsort(pairs, npairs, sizeof(*pairs), pair_cmp);
    }

    /* Calibration/test split inside the held-out validation region.
     * Calibration chooses the HDC temperature; test reports the metric. */
    {
        uint32_t nproto = m->vocab * m->clusters;
        size_t val_start = train_len + 2;
        size_t val_end = len - 2;
        size_t val_mid = train_len + (len - train_len) / 2;
        size_t calib_start = val_start;
        size_t calib_end = val_mid;
        size_t test_start = val_mid + 2;
        size_t calib_n, test_n, i;
        double grid[] = {4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0,
                         1024.0};
        size_t gi;

        if (val_end <= val_start || calib_end <= calib_start) goto fail;
        if (test_start > val_end) test_start = val_start;
        calib_n = calib_end - calib_start + 1;
        test_n = val_end - test_start + 1;
        if (calib_n > 128) calib_n = 128;
        if (test_n > 128) test_n = 128;
        if (calib_n == 0 || test_n == 0) goto fail;

        calib_dist = (uint32_t *)calloc((size_t)calib_n * nproto,
                                        sizeof(*calib_dist));
        calib_exp = (uint8_t *)calloc(calib_n, 1);
        dist = (uint32_t *)calloc(nproto, sizeof(*dist));
        if (!calib_dist || !calib_exp || !dist) goto fail;

        for (i = 0; i < calib_n; i++) {
            size_t pos = sample_pos(calib_start, calib_end, i, calib_n);
            uint8_t ids[3];
            ids[0] = data[pos - 2];
            ids[1] = data[pos - 1];
            ids[2] = data[pos];
            if (nanlm_lm_encode(m, ids, m->queries[0]) != 0) goto fail;
            calib_exp[i] = data[pos + 1];
            for (c = 0; c < nproto; c++) {
                dist[c] = flat_dist(m->queries[0], m->prototypes[c]);
            }
            memcpy(calib_dist + i * nproto, dist, nproto * sizeof(*dist));
        }

        {
            double best = 1e300;
            for (gi = 0; gi < sizeof(grid) / sizeof(grid[0]); gi++) {
                double sum = 0.0;
                for (i = 0; i < calib_n; i++) {
                    sum += bpc_from_dist(calib_dist + i * nproto, nproto,
                                         m->clusters, calib_exp[i], grid[gi]);
                }
                if (sum < best) {
                    best = sum;
                    T = grid[gi];
                }
            }
        }
        m->temperature = T;

        for (i = 0; i < test_n; i++) {
            size_t pos = sample_pos(test_start, val_end, i, test_n);
            uint8_t ids[3];
            uint32_t bestp = 0, bestd = UINT32_MAX;
            ids[0] = data[pos - 2];
            ids[1] = data[pos - 1];
            ids[2] = data[pos];
            if (nanlm_lm_encode(m, ids, m->queries[0]) != 0) goto fail;
            for (c = 0; c < nproto; c++) {
                dist[c] = flat_dist(m->queries[0], m->prototypes[c]);
                if (dist[c] < bestd) { bestd = dist[c]; bestp = c; }
            }
            if ((bestp >> m->proto_shift) == data[pos + 1]) hdc_correct++;
            hdc_bpc += bpc_from_dist(dist, nproto, m->clusters,
                                     data[pos + 1], T);
            {
                uint32_t key = ((uint32_t)ids[0] << 16) |
                               ((uint32_t)ids[1] << 8) | ids[2];
                ngram_one(pairs, npairs, key, data[pos + 1], m->vocab,
                          &ngram_correct, &ngram_bpc);
            }
        }
        m->calib_n = (uint32_t)calib_n;
        m->eval_n = (uint32_t)test_n;
        m->hdc_top1 = 100.0 * (double)hdc_correct / (double)test_n;
        m->ngram_top1 = 100.0 * (double)ngram_correct / (double)test_n;
        m->hdc_bpc = hdc_bpc / (double)test_n;
        m->ngram_bpc = ngram_bpc / (double)test_n;
    }

    /* Runtime queries: a subset of the test region, emitted for the .nan
     * differential gate. The expected class is the C prediction. */
    {
        size_t val_start = train_len + 2;
        size_t val_end = len - 2;
        size_t val_mid = train_len + (len - train_len) / 2;
        size_t test_start = val_mid + 2;
        if (test_start > val_end) test_start = val_start;
        if (test_start > val_end) goto fail;
        for (qi = 0; qi < m->nqueries; qi++) {
            size_t pos = sample_pos(test_start, val_end, qi, m->nqueries);
            uint8_t ids[3];
            uint32_t got = 0;
            ids[0] = data[pos - 2];
            ids[1] = data[pos - 1];
            ids[2] = data[pos];
            memcpy(m->query_ids + qi * NANLM_LM_CTX, ids, NANLM_LM_CTX);
            m->expected[qi] = data[pos + 1];
            if (nanlm_lm_encode(m, ids, m->queries[qi]) != 0) goto fail;
            if (nanlm_lm_predict(m, m->queries[qi], &got) != 0) goto fail;
            m->predicted[qi] = (uint8_t)got;
            m->hashes[qi] = nanlm_lm_hash(m->queries[qi]);
        }
    }

    free(calib_dist);
    free(calib_exp);
    free(dist);
    free(pairs);
    free(ones);
    free(total);
    free(seen);
    nanlm_vec_free(q);
    return 0;

fail:
    free(calib_dist);
    free(calib_exp);
    free(dist);
    free(pairs);
    free(ones);
    free(total);
    free(seen);
    nanlm_vec_free(q);
    return -1;
}

void nanlm_lm_zero_positions(nanlm_lm_model *m) {
    uint32_t i;
    if (!m) return;
    for (i = 0; i < NANLM_LM_CTX; i++) nanlm_vec_zero(m->positions[i]);
}

int nanlm_lm_swap_prototypes(nanlm_lm_model *m, uint32_t a, uint32_t b) {
    nanlm_vec *tmp;
    if (!m || a >= m->vocab * m->clusters ||
        b >= m->vocab * m->clusters) return -1;
    tmp = m->prototypes[a];
    m->prototypes[a] = m->prototypes[b];
    m->prototypes[b] = tmp;
    return 0;
}

int nanlm_lm_swap_predicted(nanlm_lm_model *m) {
    uint32_t a, b = UINT32_MAX, i;
    if (!m || m->nqueries == 0 || m->vocab < 2) return -1;
    a = m->predicted[0];
    for (i = 1; i < m->nqueries; i++) {
        if (m->predicted[i] != a) { b = m->predicted[i]; break; }
    }
    if (b == UINT32_MAX) b = (a + 1u) % m->vocab;
    return nanlm_lm_swap_prototypes(m, a * m->clusters, b * m->clusters);
}

static int emit_vec(FILE *f, const nanlm_vec *v) {
    size_t i;
    for (i = 0; i < v->words; i++) {
        if (fprintf(f, ".word 0x%08X\n", nan16_payload(v->w[i])) < 0) return -1;
    }
    return 0;
}

int nanlm_lm_emit_nas(const nanlm_lm_model *m, const char *path) {
    FILE *f;
    uint32_t i, q;
    int ok = 1;
    if (!m || !path) return -1;
    f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "; NaNLM byte-level model (generated, do not edit)\n");
    fprintf(f, "; dim=%u words=%u vocab=%u queries=%u context=%u\n", m->dim,
            m->words, m->vocab, m->nqueries, NANLM_LM_CTX);
    fprintf(f, ".equ NVEC %u\n", m->vocab * m->clusters);
    fprintf(f, ".equ VW %u\n", m->words);
    fprintf(f, ".equ NQ %u\n", m->nqueries);
    fprintf(f, ".equ CTX %u\n", NANLM_LM_CTX);
    fprintf(f, ".equ PROTOSHIFT %u\n", m->proto_shift);
    fprintf(f, ".equ BEST_INIT %u\n", m->dim + 1u);
    fprintf(f, ".org 0\n");
    if (fputs(nanlm_runtime_source(), f) == EOF) ok = 0;
    fprintf(f, ".org 8192\n");
    fprintf(f, "codebook:\n");
    for (i = 0; i < m->vocab && ok; i++) {
        if (emit_vec(f, m->codebook[i]) != 0) ok = 0;
    }
    fprintf(f, "positions:\n");
    for (i = 0; i < NANLM_LM_CTX && ok; i++) {
        if (emit_vec(f, m->positions[i]) != 0) ok = 0;
    }
    fprintf(f, "prototypes:\n");
    for (i = 0; i < m->vocab * m->clusters && ok; i++) {
        if (emit_vec(f, m->prototypes[i]) != 0) ok = 0;
    }
    fprintf(f, "query_ids:\n");
    for (q = 0; q < m->nqueries && ok; q++) {
        for (i = 0; i < NANLM_LM_CTX; i++) {
            if (fprintf(f, ".word 0x%08X\n",
                        (unsigned)m->query_ids[(size_t)q * NANLM_LM_CTX + i]) <
                0) {
                ok = 0;
                break;
            }
        }
    }
    fprintf(f, "tmp0:\n");
    for (i = 0; i < m->words && ok; i++) fprintf(f, ".word 0\n");
    fprintf(f, "tmp1:\n");
    for (i = 0; i < m->words && ok; i++) fprintf(f, ".word 0\n");
    fprintf(f, "tmp2:\n");
    for (i = 0; i < m->words && ok; i++) fprintf(f, ".word 0\n");
    fprintf(f, "qbuf:\n");
    for (i = 0; i < m->words && ok; i++) fprintf(f, ".word 0\n");
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

int nanlm_lm_write_expected(const nanlm_lm_model *m, const char *path) {
    FILE *f;
    uint32_t q;
    if (!m || !path) return -1;
    f = fopen(path, "w");
    if (!f) return -1;
    for (q = 0; q < m->nqueries; q++) {
        if (fprintf(f, "%02X %08X\n", m->predicted[q] & 0xFFu, m->hashes[q]) <
            0) {
            fclose(f);
            return -1;
        }
    }
    return fclose(f) == 0 ? 0 : -1;
}

int nanlm_lm_write_manifest(const nanlm_lm_model *m, const char *path,
                            uint64_t seed, double train_frac) {
    FILE *f;
    if (!m || !path) return -1;
    f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "magic NANLMLM\n");
    fprintf(f, "version 1\n");
    fprintf(f, "dim %u\n", m->dim);
    fprintf(f, "words %u\n", m->words);
    fprintf(f, "vocab %u\n", m->vocab);
    fprintf(f, "clusters %u\n", m->clusters);
    fprintf(f, "proto_shift %u\n", m->proto_shift);
    fprintf(f, "context %u\n", NANLM_LM_CTX);
    fprintf(f, "queries %u\n", m->nqueries);
    fprintf(f, "train_bytes %u\n", m->train_bytes);
    fprintf(f, "val_bytes %u\n", m->val_bytes);
    fprintf(f, "train_frac %.6f\n", train_frac);
    fprintf(f, "seed 0x%016llX\n", (unsigned long long)seed);
    fprintf(f, "calib_n %u\n", m->calib_n);
    fprintf(f, "eval_n %u\n", m->eval_n);
    fprintf(f, "temperature %.6f\n", m->temperature);
    fprintf(f, "hdc_top1 %.6f\n", m->hdc_top1);
    fprintf(f, "ngram_top1 %.6f\n", m->ngram_top1);
    fprintf(f, "hdc_bpc %.6f\n", m->hdc_bpc);
    fprintf(f, "ngram_bpc %.6f\n", m->ngram_bpc);
    fprintf(f, "data_org 8192\n");
    fprintf(f, "codebook_words %u\n", m->vocab * m->words);
    fprintf(f, "position_words %u\n", NANLM_LM_CTX * m->words);
    fprintf(f, "prototype_words %u\n", m->vocab * m->clusters * m->words);
    fprintf(f, "query_id_words %u\n", m->nqueries * NANLM_LM_CTX);
    return fclose(f) == 0 ? 0 : -1;
}
