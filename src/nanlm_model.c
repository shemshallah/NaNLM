#include "nanlm_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void vec_copy(nanlm_vec *dst, const nanlm_vec *src) {
    memcpy(dst->w, src->w, src->words * sizeof(nan16_word));
}

static void flip_bit(nanlm_vec *v, uint32_t bit) {
    uint32_t wi = bit / 32u;
    uint32_t bi = bit % 32u;
    uint32_t p = nan16_payload(v->w[wi]);
    v->w[wi] = nan16_make(p ^ (1u << bi), NAN16_N_DATA);
}

nanlm_model *nanlm_model_new(uint32_t dim, uint32_t classes,
                             uint32_t nqueries) {
    nanlm_model *m;
    uint32_t i;

    if (dim == 0 || (dim % 32u) != 0 || dim > 65504u || classes == 0 ||
        classes > 256u || nqueries == 0 || nqueries > 65535u) {
        return NULL;
    }
    m = (nanlm_model *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->dim = dim;
    m->words = dim / 32u;
    m->classes = classes;
    m->nqueries = nqueries;
    m->prototypes = (nanlm_vec **)calloc(classes, sizeof(*m->prototypes));
    m->queries = (nanlm_vec **)calloc(nqueries, sizeof(*m->queries));
    m->expected = (uint8_t *)calloc(nqueries, 1);
    if (!m->prototypes || !m->queries || !m->expected) {
        nanlm_model_free(m);
        return NULL;
    }
    for (i = 0; i < classes; i++) {
        m->prototypes[i] = nanlm_vec_new(m->words);
        if (!m->prototypes[i]) {
            nanlm_model_free(m);
            return NULL;
        }
    }
    for (i = 0; i < nqueries; i++) {
        m->queries[i] = nanlm_vec_new(m->words);
        if (!m->queries[i]) {
            nanlm_model_free(m);
            return NULL;
        }
    }
    return m;
}

void nanlm_model_free(nanlm_model *m) {
    uint32_t i;
    if (!m) return;
    if (m->prototypes) {
        for (i = 0; i < m->classes; i++) nanlm_vec_free(m->prototypes[i]);
        free(m->prototypes);
    }
    if (m->queries) {
        for (i = 0; i < m->nqueries; i++) nanlm_vec_free(m->queries[i]);
        free(m->queries);
    }
    free(m->expected);
    free(m);
}

int nanlm_model_predict(const nanlm_model *m, const nanlm_vec *query,
                        uint32_t *out_class) {
    uint32_t c;
    uint64_t best = UINT64_MAX;
    uint32_t best_c = 0;

    if (!m || !query || !out_class || query->words != m->words) return -1;
    for (c = 0; c < m->classes; c++) {
        uint64_t d = nanlm_vec_hamming(query, m->prototypes[c]);
        if (d == UINT64_MAX) return -1;
        if (d < best) {
            best = d;
            best_c = c;
        }
    }
    *out_class = best_c;
    return 0;
}

int nanlm_model_train_toy(nanlm_model *m, uint64_t seed,
                          uint32_t samples_per_class, uint32_t noise_bits) {
    nanlm_vec **centroid;
    nanlm_vec *sample;
    uint32_t *ones;
    uint32_t *total;
    uint32_t *labels = NULL;
    uint32_t c, s, w, b;
    uint64_t state = seed;

    if (!m || samples_per_class == 0 || noise_bits > m->dim) return -1;

    centroid = (nanlm_vec **)calloc(m->classes, sizeof(*centroid));
    ones = (uint32_t *)calloc((size_t)m->classes * m->dim, sizeof(*ones));
    total = (uint32_t *)calloc(m->classes, sizeof(*total));
    labels = (uint32_t *)calloc(m->nqueries, sizeof(*labels));
    sample = nanlm_vec_new(m->words);
    if (!centroid || !ones || !total || !labels || !sample) {
        free(centroid);
        free(ones);
        free(total);
        free(labels);
        nanlm_vec_free(sample);
        return -1;
    }

    for (c = 0; c < m->classes; c++) {
        centroid[c] = nanlm_vec_new(m->words);
        if (!centroid[c]) {
            for (uint32_t k = 0; k < c; k++) nanlm_vec_free(centroid[k]);
            free(centroid);
            free(ones);
            free(total);
            free(labels);
            nanlm_vec_free(sample);
            return -1;
        }
        nanlm_vec_random(centroid[c], &state);
    }

    for (c = 0; c < m->classes; c++) {
        for (s = 0; s < samples_per_class; s++) {
            vec_copy(sample, centroid[c]);
            for (b = 0; b < noise_bits; b++) {
                flip_bit(sample, (uint32_t)(splitmix64(&state) % m->dim));
            }
            total[c]++;
            for (w = 0; w < m->words; w++) {
                uint32_t p = nan16_payload(sample->w[w]);
                for (b = 0; b < 32u; b++) {
                    if ((p >> b) & 1u) {
                        ones[(size_t)c * m->dim + w * 32u + b]++;
                    }
                }
            }
        }
    }

    for (c = 0; c < m->classes; c++) {
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

    /* Shuffle the query labels. Without this, a runtime that returns
     * query_index % classes would pass the differential test. */
    for (uint32_t q = 0; q < m->nqueries; q++) labels[q] = q % m->classes;
    for (uint32_t q = m->nqueries; q > 1; q--) {
        uint32_t j = (uint32_t)(splitmix64(&state) % q);
        uint32_t tmp = labels[q - 1];
        labels[q - 1] = labels[j];
        labels[j] = tmp;
    }
    for (uint32_t q = 0; q < m->nqueries; q++) {
        uint32_t cls = labels[q];
        vec_copy(m->queries[q], centroid[cls]);
        for (b = 0; b < noise_bits; b++) {
            flip_bit(m->queries[q], (uint32_t)(splitmix64(&state) % m->dim));
        }
        m->expected[q] = (uint8_t)cls;
    }

    /* The expected labels are the ground truth. Also verify the C classifier
     * agrees with them before the model is emitted. */
    for (uint32_t q = 0; q < m->nqueries; q++) {
        uint32_t got = 0;
        if (nanlm_model_predict(m, m->queries[q], &got) != 0) {
            goto fail;
        }
        if (got != m->expected[q]) {
            goto fail;
        }
    }

    for (c = 0; c < m->classes; c++) nanlm_vec_free(centroid[c]);
    free(centroid);
    free(ones);
    free(total);
    free(labels);
    nanlm_vec_free(sample);
    return 0;

fail:
    for (c = 0; c < m->classes; c++) nanlm_vec_free(centroid[c]);
    free(centroid);
    free(ones);
    free(total);
    free(labels);
    nanlm_vec_free(sample);
    return -1;
}

/* The runtime below is the tested M1 inference loop. It expects:
 *   #NVEC, #VW, #NQ, #BEST_INIT
 *   labels @prototypes and @queries, both arrays of neutral .word data.
 * It prints two uppercase hex digits per query, followed by a newline.
 */
static const char RUNTIME[] =
    "main:\n"
    "MOVI r13,1\n"
    "MOVI r1,0\n"
    "ST r1,r0,@m_q\n"
    "LA r1,@queries\n"
    "ST r1,r0,@m_ptr\n"
    "m_loop:\n"
    "LD r1,r0,@m_q\n"
    "MOVI r2,#NQ\n"
    "CMP r1,r2\n"
    "JMP[GE] @m_done\n"
    "LD r1,r0,@m_ptr\n"
    "CALL @predict\n"
    "CALL @puthex2\n"
    "MOVI r1,10\n"
    "PUTC r1\n"
    "LD r1,r0,@m_ptr\n"
    "MOVI r2,#VW\n"
    "ADD r1,r1,r2\n"
    "ST r1,r0,@m_ptr\n"
    "LD r1,r0,@m_q\n"
    "ADD r1,r1,r13\n"
    "ST r1,r0,@m_q\n"
    "JMP @m_loop\n"
    "m_done:\n"
    "HALT\n"
    "predict:\n"
    "ST r15,r0,@p_ret\n"
    "ST r1,r0,@p_qptr\n"
    "MOVI r1,#BEST_INIT\n"
    "ST r1,r0,@p_best\n"
    "MOVI r1,0\n"
    "ST r1,r0,@p_bestc\n"
    "MOVI r1,0\n"
    "ST r1,r0,@p_c\n"
    "LA r1,@prototypes\n"
    "ST r1,r0,@p_proto\n"
    "p_loop:\n"
    "LD r1,r0,@p_c\n"
    "MOVI r2,#NVEC\n"
    "CMP r1,r2\n"
    "JMP[GE] @p_done\n"
    "LD r1,r0,@p_qptr\n"
    "LD r2,r0,@p_proto\n"
    "CALL @hamming\n"
    "LD r2,r0,@p_best\n"
    "CMP r1,r2\n"
    "JMP[GE] @p_next\n"
    "ST r1,r0,@p_best\n"
    "LD r2,r0,@p_c\n"
    "ST r2,r0,@p_bestc\n"
    "p_next:\n"
    "LD r1,r0,@p_proto\n"
    "MOVI r2,#VW\n"
    "ADD r1,r1,r2\n"
    "ST r1,r0,@p_proto\n"
    "LD r1,r0,@p_c\n"
    "ADD r1,r1,r13\n"
    "ST r1,r0,@p_c\n"
    "JMP @p_loop\n"
    "p_done:\n"
    "LD r1,r0,@p_bestc\n"
    "LD r15,r0,@p_ret\n"
    "RET\n"
    "hamming:\n"
    "ST r15,r0,@h_ret\n"
    "ADD r8,r1,r0\n"
    "ADD r9,r2,r0\n"
    "MOVI r10,0\n"
    "MOVI r11,0\n"
    "h_loop:\n"
    "MOVI r1,#VW\n"
    "CMP r10,r1\n"
    "JMP[GE] @h_done\n"
    "LD r1,r8,0\n"
    "LD r2,r9,0\n"
    "BOOL.6 r1,r1,r2\n"
    "CALL @popcount32\n"
    "ADD r11,r11,r1\n"
    "ADD r8,r8,r13\n"
    "ADD r9,r9,r13\n"
    "ADD r10,r10,r13\n"
    "JMP @h_loop\n"
    "h_done:\n"
    "ADD r1,r11,r0\n"
    "LD r15,r0,@h_ret\n"
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
    "m_q:\n"
    ".word 0\n"
    "m_ptr:\n"
    ".word 0\n"
    "p_ret:\n"
    ".word 0\n"
    "p_qptr:\n"
    ".word 0\n"
    "p_c:\n"
    ".word 0\n"
    "p_best:\n"
    ".word 0\n"
    "p_bestc:\n"
    ".word 0\n"
    "p_proto:\n"
    ".word 0\n"
    "h_ret:\n"
    ".word 0\n"
    "ph2_ret:\n"
    ".word 0\n"
    "ph2_val:\n"
    ".word 0\n";

int nanlm_model_emit_nas(const nanlm_model *m, const char *path) {
    FILE *f;
    uint32_t c, q, w;
    int ok = 1;

    if (!m || !path) return -1;
    f = fopen(path, "w");
    if (!f) return -1;

    fprintf(f, "; NaNLM-0 model image (generated, do not edit)\n");
    fprintf(f, "; dim=%u words=%u classes=%u queries=%u\n", m->dim, m->words,
            m->classes, m->nqueries);
    fprintf(f, ".equ NVEC %u\n", m->classes);
    fprintf(f, ".equ VW %u\n", m->words);
    fprintf(f, ".equ NQ %u\n", m->nqueries);
    fprintf(f, ".equ BEST_INIT %u\n", m->dim + 1u);
    fprintf(f, ".org 0\n");
    if (fputs(RUNTIME, f) == EOF) ok = 0;
    fprintf(f, ".org 8192\n");
    fprintf(f, "prototypes:\n");
    for (c = 0; c < m->classes && ok; c++) {
        for (w = 0; w < m->words; w++) {
            if (fprintf(f, ".word 0x%08X\n",
                        nan16_payload(m->prototypes[c]->w[w])) < 0) {
                ok = 0;
                break;
            }
        }
    }
    fprintf(f, "queries:\n");
    for (q = 0; q < m->nqueries && ok; q++) {
        for (w = 0; w < m->words; w++) {
            if (fprintf(f, ".word 0x%08X\n",
                        nan16_payload(m->queries[q]->w[w])) < 0) {
                ok = 0;
                break;
            }
        }
    }
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

int nanlm_model_write_expected(const nanlm_model *m, const char *path) {
    FILE *f;
    uint32_t q;
    if (!m || !path) return -1;
    f = fopen(path, "w");
    if (!f) return -1;
    for (q = 0; q < m->nqueries; q++) {
        if (fprintf(f, "%02X\n", m->expected[q] & 0xFFu) < 0) {
            fclose(f);
            return -1;
        }
    }
    return fclose(f) == 0 ? 0 : -1;
}

int nanlm_model_write_manifest(const nanlm_model *m, const char *path,
                               uint64_t seed, uint32_t samples_per_class,
                               uint32_t noise_bits) {
    FILE *f;
    if (!m || !path) return -1;
    f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "magic NANLM0\n");
    fprintf(f, "version 1\n");
    fprintf(f, "dim %u\n", m->dim);
    fprintf(f, "words %u\n", m->words);
    fprintf(f, "classes %u\n", m->classes);
    fprintf(f, "queries %u\n", m->nqueries);
    fprintf(f, "seed 0x%016llX\n", (unsigned long long)seed);
    fprintf(f, "samples_per_class %u\n", samples_per_class);
    fprintf(f, "noise_bits %u\n", noise_bits);
    fprintf(f, "data_org 8192\n");
    fprintf(f, "prototype_words %u\n", m->classes * m->words);
    fprintf(f, "query_words %u\n", m->nqueries * m->words);
    return fclose(f) == 0 ? 0 : -1;
}
