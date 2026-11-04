#include "nanlm_lm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanlm_runtime.h"

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

int nanlm_lm_train(nanlm_lm_model *m, const uint8_t *data, size_t len,
                   double train_frac, uint64_t seed) {
    uint64_t state = seed;
    size_t train_len, qstart, qend, npos, t, qi;
    uint32_t c, w, b, i;
    uint32_t *ones = NULL;
    uint32_t *total = NULL;
    uint32_t *seen = NULL;
    nanlm_vec *q = NULL;
    Pair *pairs = NULL;
    size_t npairs = 0;
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

    if (m->exemplar) {
        seen = (uint32_t *)calloc(m->vocab, sizeof(*seen));
        if (!seen) return -1;
    } else {
        ones = (uint32_t *)calloc((size_t)m->vocab * m->clusters * m->dim,
                                  sizeof(*ones));
        total = (uint32_t *)calloc((size_t)m->vocab * m->clusters,
                                   sizeof(*total));
    }
    q = nanlm_vec_new(m->words);
    if ((!m->exemplar && (!ones || !total)) || (m->exemplar && !seen) || !q) {
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
        if (m->exemplar) {
            uint32_t slot = seen[cls] % m->clusters;
            nanlm_vec *dst = m->prototypes[cls * m->clusters + slot];
            memcpy(dst->w, q->w, m->words * sizeof(*q->w));
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

    if (!m->exemplar) {
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

    /* Held-out queries, evenly spaced over the validation split. */
    qstart = train_len + 2;
    qend = len - 1; /* exclusive: a query at pos needs pos+1 < len */
    if (qend <= qstart) {
        free(ones);
        free(total);
        free(seen);
        nanlm_vec_free(q);
        return -1;
    }
    for (qi = 0; qi < m->nqueries; qi++) {
        size_t pos = qstart + (qi * (qend - qstart)) / m->nqueries;
        uint8_t ids[3];
        if (pos + 1 >= len) pos = len - 2;
        ids[0] = data[pos - 2];
        ids[1] = data[pos - 1];
        ids[2] = data[pos];
        memcpy(m->query_ids + qi * NANLM_LM_CTX, ids, NANLM_LM_CTX);
        m->expected[qi] = data[pos + 1];
        if (nanlm_lm_encode(m, ids, m->queries[qi]) != 0) {
            free(ones);
            free(total);
            free(seen);
            nanlm_vec_free(q);
            return -1;
        }
        m->hashes[qi] = nanlm_lm_hash(m->queries[qi]);
    }

    /* HDC top-1 and a diagnostic softmax bits/byte. The emitted expected
     * class is the C model's prediction, not the corpus ground truth. */
    for (qi = 0; qi < m->nqueries; qi++) {
        double scores[256 * NANLM_LM_MAX_CLUSTERS];
        double maxs = -1e300, sum = 0.0;
        uint32_t got = 0;
        if (nanlm_lm_predict(m, m->queries[qi], &got) != 0) {
            free(ones);
            free(total);
            free(seen);
            nanlm_vec_free(q);
            return -1;
        }
        m->predicted[qi] = (uint8_t)got;
        if (got == m->expected[qi]) hdc_correct++;
        for (c = 0; c < m->vocab * m->clusters; c++) {
            uint64_t d = nanlm_vec_hamming(m->queries[qi], m->prototypes[c]);
            scores[c] = -(double)d / T;
            if (scores[c] > maxs) maxs = scores[c];
        }
        for (c = 0; c < m->vocab * m->clusters; c++) {
            sum += exp(scores[c] - maxs);
        }
        {
            double logp = (scores[m->expected[qi]] - maxs - log(sum)) / log(2.0);
            hdc_bpc += -logp;
        }
    }

    /* Order-3 byte n-gram baseline on the same held-out queries. */
    npos = (train_len >= 3) ? train_len - 3 : 0;
    if (npos > 0) {
        size_t k = 0;
        pairs = (Pair *)calloc(npos, sizeof(*pairs));
        if (!pairs) {
            free(ones);
            free(total);
            free(seen);
            nanlm_vec_free(q);
            return -1;
        }
        for (t = 2; t + 1 < train_len; t++) {
            pairs[k].key = ((uint32_t)data[t - 2] << 16) |
                            ((uint32_t)data[t - 1] << 8) | data[t];
            pairs[k].next = data[t + 1];
            k++;
        }
        npairs = k;
        qsort(pairs, npairs, sizeof(*pairs), pair_cmp);
        for (qi = 0; qi < m->nqueries; qi++) {
            uint32_t key = ((uint32_t)m->query_ids[qi * NANLM_LM_CTX] << 16) |
                           ((uint32_t)m->query_ids[qi * NANLM_LM_CTX + 1] << 8) |
                           m->query_ids[qi * NANLM_LM_CTX + 2];
            size_t lo = lower_key(pairs, npairs, key);
            size_t hi = upper_key(pairs, npairs, key);
            uint32_t best_c = 0, best_count = 0;
            uint32_t counts[256] = {0};
            double alpha = 0.1;
            uint32_t total_count = (uint32_t)(hi - lo);
            for (size_t j = lo; j < hi; j++) counts[pairs[j].next]++;
            for (c = 0; c < m->vocab; c++) {
                if (counts[c] > best_count) {
                    best_count = counts[c];
                    best_c = c;
                }
            }
            if (best_c == m->expected[qi]) ngram_correct++;
            {
                double p = ((double)counts[m->expected[qi]] + alpha) /
                           ((double)total_count + alpha * (double)m->vocab);
                ngram_bpc += -log(p) / log(2.0);
            }
        }
    }

    m->hdc_top1 = 100.0 * (double)hdc_correct / (double)m->nqueries;
    m->ngram_top1 = 100.0 * (double)ngram_correct / (double)m->nqueries;
    m->hdc_bpc = hdc_bpc / (double)m->nqueries;
    m->ngram_bpc = ngram_bpc / (double)m->nqueries;

    free(pairs);
    free(ones);
    free(total);
    free(seen);
    nanlm_vec_free(q);
    return 0;
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
    fprintf(f, "hdc_top1 %.6f\n", m->hdc_top1);
    fprintf(f, "ngram_top1 %.6f\n", m->ngram_top1);
    fprintf(f, "ngram_bpc %.6f\n", m->ngram_bpc);
    fprintf(f, "data_org 8192\n");
    fprintf(f, "codebook_words %u\n", m->vocab * m->words);
    fprintf(f, "position_words %u\n", NANLM_LM_CTX * m->words);
    fprintf(f, "prototype_words %u\n", m->vocab * m->clusters * m->words);
    fprintf(f, "query_id_words %u\n", m->nqueries * NANLM_LM_CTX);
    return fclose(f) == 0 ? 0 : -1;
}
