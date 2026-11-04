#include "nanlm_ctx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

nanlm_ctx_model *nanlm_ctx_model_new(uint32_t dim, uint32_t classes,
                                     uint32_t nqueries) {
    nanlm_ctx_model *m;
    uint32_t i, ncb;

    if (dim == 0 || (dim % 32u) != 0 || dim > 65504u || classes == 0 ||
        nqueries == 0 || nqueries > 65535u) {
        return NULL;
    }
    if ((uint64_t)classes * NANLM_CTX_LEN > 256u) return NULL;

    m = (nanlm_ctx_model *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->dim = dim;
    m->words = dim / 32u;
    m->classes = classes;
    m->nqueries = nqueries;
    ncb = classes * NANLM_CTX_LEN;

    m->codebook = (nanlm_vec **)calloc(ncb, sizeof(*m->codebook));
    m->positions = (nanlm_vec **)calloc(NANLM_CTX_LEN, sizeof(*m->positions));
    m->prototypes = (nanlm_vec **)calloc(classes, sizeof(*m->prototypes));
    m->queries = (nanlm_vec **)calloc(nqueries, sizeof(*m->queries));
    m->query_ids = (uint8_t *)calloc((size_t)nqueries * NANLM_CTX_LEN, 1);
    m->expected = (uint8_t *)calloc(nqueries, 1);
    m->hashes = (uint32_t *)calloc(nqueries, sizeof(*m->hashes));
    for (i = 0; i < NANLM_CTX_LEN; i++) m->tmp[i] = NULL;

    if (!m->codebook || !m->positions || !m->prototypes || !m->queries ||
        !m->query_ids || !m->expected || !m->hashes) {
        nanlm_ctx_model_free(m);
        return NULL;
    }
    for (i = 0; i < ncb; i++) {
        m->codebook[i] = nanlm_vec_new(m->words);
        if (!m->codebook[i]) { nanlm_ctx_model_free(m); return NULL; }
    }
    for (i = 0; i < NANLM_CTX_LEN; i++) {
        m->positions[i] = nanlm_vec_new(m->words);
        if (!m->positions[i]) { nanlm_ctx_model_free(m); return NULL; }
    }
    for (i = 0; i < classes; i++) {
        m->prototypes[i] = nanlm_vec_new(m->words);
        if (!m->prototypes[i]) { nanlm_ctx_model_free(m); return NULL; }
    }
    for (i = 0; i < nqueries; i++) {
        m->queries[i] = nanlm_vec_new(m->words);
        if (!m->queries[i]) { nanlm_ctx_model_free(m); return NULL; }
    }
    for (i = 0; i < NANLM_CTX_LEN; i++) {
        m->tmp[i] = nanlm_vec_new(m->words);
        if (!m->tmp[i]) { nanlm_ctx_model_free(m); return NULL; }
    }
    return m;
}

void nanlm_ctx_model_free(nanlm_ctx_model *m) {
    uint32_t i, ncb;
    if (!m) return;
    ncb = m->classes * NANLM_CTX_LEN;
    if (m->codebook) {
        for (i = 0; i < ncb; i++) nanlm_vec_free(m->codebook[i]);
        free(m->codebook);
    }
    if (m->positions) {
        for (i = 0; i < NANLM_CTX_LEN; i++) nanlm_vec_free(m->positions[i]);
        free(m->positions);
    }
    if (m->prototypes) {
        for (i = 0; i < m->classes; i++) nanlm_vec_free(m->prototypes[i]);
        free(m->prototypes);
    }
    if (m->queries) {
        for (i = 0; i < m->nqueries; i++) nanlm_vec_free(m->queries[i]);
        free(m->queries);
    }
    for (i = 0; i < NANLM_CTX_LEN; i++) nanlm_vec_free(m->tmp[i]);
    free(m->query_ids);
    free(m->expected);
    free(m->hashes);
    free(m);
}

int nanlm_ctx_encode(const nanlm_ctx_model *m, const uint8_t ids[3],
                     nanlm_vec *out) {
    uint32_t i, ncb;
    if (!m || !ids || !out || out->words != m->words) return -1;
    ncb = m->classes * NANLM_CTX_LEN;
    for (i = 0; i < NANLM_CTX_LEN; i++) {
        if (ids[i] >= ncb) return -1;
    }
    for (i = 0; i < NANLM_CTX_LEN; i++) {
        if (nanlm_vec_bind(m->tmp[i], m->codebook[ids[i]], m->positions[i]) !=
            0) {
            return -1;
        }
    }
    return nanlm_vec_bundle3(out, m->tmp[0], m->tmp[1], m->tmp[2]);
}

int nanlm_ctx_predict(const nanlm_ctx_model *m, const nanlm_vec *query,
                      uint32_t *out_class) {
    uint32_t c, best_c = 0;
    uint64_t best = UINT64_MAX;
    if (!m || !query || !out_class || query->words != m->words) return -1;
    for (c = 0; c < m->classes; c++) {
        uint64_t d = nanlm_vec_hamming(query, m->prototypes[c]);
        if (d == UINT64_MAX) return -1;
        if (d < best) { best = d; best_c = c; }
    }
    *out_class = best_c;
    return 0;
}

uint32_t nanlm_ctx_hash(const nanlm_vec *v) {
    uint32_t h = 2166136261u;
    size_t i;
    if (!v || !v->w || !nanlm_vec_is_data(v)) return 0;
    for (i = 0; i < v->words; i++) {
        h ^= nan16_payload(v->w[i]);
        h *= 16777619u;
    }
    return h;
}

int nanlm_ctx_build(nanlm_ctx_model *m, uint64_t seed) {
    uint64_t state = seed;
    uint32_t c, i, q, ncb;
    uint32_t *labels = NULL;

    if (!m) return -1;
    ncb = m->classes * NANLM_CTX_LEN;
    for (i = 0; i < ncb; i++) nanlm_vec_random(m->codebook[i], &state);
    for (i = 0; i < NANLM_CTX_LEN; i++) {
        nanlm_vec_random(m->positions[i], &state);
    }

    for (c = 0; c < m->classes; c++) {
        for (i = 0; i < NANLM_CTX_LEN; i++) {
            if (nanlm_vec_bind(m->tmp[i], m->codebook[c * NANLM_CTX_LEN + i],
                               m->positions[i]) != 0) {
                return -1;
            }
        }
        if (nanlm_vec_bundle3(m->prototypes[c], m->tmp[0], m->tmp[1],
                              m->tmp[2]) != 0) {
            return -1;
        }
    }

    labels = (uint32_t *)calloc(m->nqueries, sizeof(*labels));
    if (!labels) return -1;
    for (q = 0; q < m->nqueries; q++) labels[q] = q % m->classes;
    for (q = m->nqueries; q > 1; q--) {
        uint32_t j = (uint32_t)(splitmix64(&state) % q);
        uint32_t tmp = labels[q - 1];
        labels[q - 1] = labels[j];
        labels[j] = tmp;
    }

    for (q = 0; q < m->nqueries; q++) {
        uint8_t ids[NANLM_CTX_LEN];
        uint32_t cls = labels[q];
        for (i = 0; i < NANLM_CTX_LEN; i++) {
            ids[i] = (uint8_t)(cls * NANLM_CTX_LEN + i);
            m->query_ids[(size_t)q * NANLM_CTX_LEN + i] = ids[i];
        }
        if (nanlm_ctx_encode(m, ids, m->queries[q]) != 0) {
            free(labels);
            return -1;
        }
        m->expected[q] = (uint8_t)cls;
        m->hashes[q] = nanlm_ctx_hash(m->queries[q]);
    }
    free(labels);

    for (q = 0; q < m->nqueries; q++) {
        uint32_t got = 0;
        if (nanlm_ctx_predict(m, m->queries[q], &got) != 0) return -1;
        if (got != m->expected[q]) return -1;
    }
    return 0;
}

void nanlm_ctx_zero_positions(nanlm_ctx_model *m) {
    uint32_t i;
    if (!m) return;
    for (i = 0; i < NANLM_CTX_LEN; i++) nanlm_vec_zero(m->positions[i]);
}

int nanlm_ctx_swap_prototypes(nanlm_ctx_model *m, uint32_t a, uint32_t b) {
    nanlm_vec *tmp;
    if (!m || a >= m->classes || b >= m->classes) return -1;
    tmp = m->prototypes[a];
    m->prototypes[a] = m->prototypes[b];
    m->prototypes[b] = tmp;
    return 0;
}

/* Runtime: encode with bind+bundle, hash the vector, then classify.
 * Output format: CC HHHHHHHH\n (two hex class digits, space, FNV-1a hash).
 */
static const char RUNTIME[] =
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
    "MOVI r2,#CTX\n"
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
    "MOVI r2,#CTX\n"
    "CMP r1,r2\n"
    "JMP[GE] @e_bundle\n"
    "LD r3,r0,@e_ids\n"
    "ADD r3,r3,r1\n"
    "LD r4,r3,0\n"
    "MOVI r5,#VW\n"
    "MUL r4,r4,r5\n"
    "LA r5,@codebook\n"
    "ADD r5,r5,r4\n"
    "LD r3,r0,@e_i\n"
    "MOVI r4,#VW\n"
    "MUL r3,r3,r4\n"
    "LA r4,@positions\n"
    "ADD r4,r4,r3\n"
    "LD r3,r0,@e_i\n"
    "MOVI r6,#VW\n"
    "MUL r3,r3,r6\n"
    "LA r6,@tmp0\n"
    "ADD r6,r6,r3\n"
    "ADD r1,r6,r0\n"
    "ADD r2,r5,r0\n"
    "ADD r3,r4,r0\n"
    "CALL @vec_xor\n"
    "LD r1,r0,@e_i\n"
    "ADD r1,r1,r13\n"
    "ST r1,r0,@e_i\n"
    "JMP @e_loop\n"
    "e_bundle:\n"
    "LD r1,r0,@e_dst\n"
    "LA r2,@tmp0\n"
    "LA r3,@tmp1\n"
    "LA r4,@tmp2\n"
    "CALL @vec_majority\n"
    "LD r15,r0,@e_ret\n"
    "RET\n"
    "vec_xor:\n"
    "MOVI r4,0\n"
    "MOVI r5,#VW\n"
    "vx_loop:\n"
    "CMP r4,r5\n"
    "JMP[GE] @vx_done\n"
    "LD r6,r2,0\n"
    "LD r7,r3,0\n"
    "BOOL.6 r6,r6,r7\n"
    "ST r6,r1,0\n"
    "ADD r1,r1,r13\n"
    "ADD r2,r2,r13\n"
    "ADD r3,r3,r13\n"
    "ADD r4,r4,r13\n"
    "JMP @vx_loop\n"
    "vx_done:\n"
    "RET\n"
    "vec_majority:\n"
    "MOVI r5,0\n"
    "MOVI r6,#VW\n"
    "vm_loop:\n"
    "CMP r5,r6\n"
    "JMP[GE] @vm_done\n"
    "LD r7,r2,0\n"
    "LD r8,r3,0\n"
    "LD r9,r4,0\n"
    "BOOL.8 r10,r7,r8\n"
    "BOOL.8 r11,r7,r9\n"
    "BOOL.8 r12,r8,r9\n"
    "BOOL.14 r10,r10,r11\n"
    "BOOL.14 r10,r10,r12\n"
    "ST r10,r1,0\n"
    "ADD r1,r1,r13\n"
    "ADD r2,r2,r13\n"
    "ADD r3,r3,r13\n"
    "ADD r4,r4,r13\n"
    "ADD r5,r5,r13\n"
    "JMP @vm_loop\n"
    "vm_done:\n"
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
    ".word 0\n"
    "ph8_ret:\n"
    ".word 0\n"
    "ph8_val:\n"
    ".word 0\n";

static int emit_vec(FILE *f, const nanlm_vec *v) {
    size_t i;
    for (i = 0; i < v->words; i++) {
        if (fprintf(f, ".word 0x%08X\n", nan16_payload(v->w[i])) < 0) return -1;
    }
    return 0;
}

int nanlm_ctx_emit_nas(const nanlm_ctx_model *m, const char *path) {
    FILE *f;
    uint32_t i, q;
    int ok = 1;

    if (!m || !path) return -1;
    f = fopen(path, "w");
    if (!f) return -1;

    fprintf(f, "; NaNLM M2a context model (generated, do not edit)\n");
    fprintf(f, "; dim=%u words=%u classes=%u queries=%u context=%u\n", m->dim,
            m->words, m->classes, m->nqueries, NANLM_CTX_LEN);
    fprintf(f, ".equ NVEC %u\n", m->classes);
    fprintf(f, ".equ VW %u\n", m->words);
    fprintf(f, ".equ NQ %u\n", m->nqueries);
    fprintf(f, ".equ CTX %u\n", NANLM_CTX_LEN);
    fprintf(f, ".equ BEST_INIT %u\n", m->dim + 1u);
    fprintf(f, ".org 0\n");
    if (fputs(RUNTIME, f) == EOF) ok = 0;
    fprintf(f, ".org 8192\n");
    fprintf(f, "codebook:\n");
    for (i = 0; i < m->classes * NANLM_CTX_LEN && ok; i++) {
        if (emit_vec(f, m->codebook[i]) != 0) ok = 0;
    }
    fprintf(f, "positions:\n");
    for (i = 0; i < NANLM_CTX_LEN && ok; i++) {
        if (emit_vec(f, m->positions[i]) != 0) ok = 0;
    }
    fprintf(f, "prototypes:\n");
    for (i = 0; i < m->classes && ok; i++) {
        if (emit_vec(f, m->prototypes[i]) != 0) ok = 0;
    }
    fprintf(f, "query_ids:\n");
    for (q = 0; q < m->nqueries && ok; q++) {
        for (i = 0; i < NANLM_CTX_LEN; i++) {
            if (fprintf(f, ".word 0x%08X\n",
                        (unsigned)m->query_ids[(size_t)q * NANLM_CTX_LEN + i]) <
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

int nanlm_ctx_write_expected(const nanlm_ctx_model *m, const char *path) {
    FILE *f;
    uint32_t q;
    if (!m || !path) return -1;
    f = fopen(path, "w");
    if (!f) return -1;
    for (q = 0; q < m->nqueries; q++) {
        if (fprintf(f, "%02X %08X\n", m->expected[q] & 0xFFu, m->hashes[q]) <
            0) {
            fclose(f);
            return -1;
        }
    }
    return fclose(f) == 0 ? 0 : -1;
}

int nanlm_ctx_write_manifest(const nanlm_ctx_model *m, const char *path,
                             uint64_t seed) {
    FILE *f;
    if (!m || !path) return -1;
    f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "magic NANLMCTX\n");
    fprintf(f, "version 1\n");
    fprintf(f, "dim %u\n", m->dim);
    fprintf(f, "words %u\n", m->words);
    fprintf(f, "classes %u\n", m->classes);
    fprintf(f, "context %u\n", NANLM_CTX_LEN);
    fprintf(f, "queries %u\n", m->nqueries);
    fprintf(f, "seed 0x%016llX\n", (unsigned long long)seed);
    fprintf(f, "data_org 8192\n");
    fprintf(f, "codebook_words %u\n", m->classes * NANLM_CTX_LEN * m->words);
    fprintf(f, "position_words %u\n", NANLM_CTX_LEN * m->words);
    fprintf(f, "prototype_words %u\n", m->classes * m->words);
    fprintf(f, "query_id_words %u\n", m->nqueries * NANLM_CTX_LEN);
    return fclose(f) == 0 ? 0 : -1;
}
