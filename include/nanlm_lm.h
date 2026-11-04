/* nanlm_lm.h - byte-level order-3 NaNLM language-model base.
 *
 * Codebook: one vector per byte.
 * Positions: three vectors.
 * Prototype: one vector per next byte.
 * Query: majority(C[b0]^P0, C[b1]^P1, C[b2]^P2).
 */
#ifndef NANLM_LM_H
#define NANLM_LM_H

#include <stddef.h>
#include <stdint.h>

#include "nanlm_vec.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NANLM_LM_CTX 3u
#define NANLM_LM_CLUSTERS 8u
#define NANLM_LM_MAX_CLUSTERS 16u

typedef struct {
    uint32_t dim;
    uint32_t words;
    uint32_t vocab;
    uint32_t clusters;
    uint32_t proto_shift;
    uint32_t nqueries;
    nanlm_vec **codebook;  /* vocab vectors */
    nanlm_vec **positions; /* 3 */
    nanlm_vec **prototypes; /* vocab * clusters */
    nanlm_vec **queries;
    nanlm_vec *tmp[NANLM_LM_CTX];
    uint8_t *query_ids; /* nqueries * 3 */
    uint8_t *expected;
    uint8_t *predicted;
    uint32_t *hashes;
    uint32_t train_bytes;
    uint32_t val_bytes;
    double hdc_top1;
    double ngram_top1;
    double hdc_bpc;
    double ngram_bpc;
    int exemplar; /* 0 = majority prototypes, 1 = first-K contexts */
} nanlm_lm_model;

nanlm_lm_model *nanlm_lm_model_new(uint32_t dim, uint32_t vocab,
                                   uint32_t nqueries);
nanlm_lm_model *nanlm_lm_model_new_clusters(uint32_t dim, uint32_t vocab,
                                            uint32_t nqueries,
                                            uint32_t clusters);
void nanlm_lm_model_free(nanlm_lm_model *m);

/* Train on data[0..len). The first train_frac of bytes is training; the rest
 * supplies held-out queries. Returns 0 on success. */
int nanlm_lm_train(nanlm_lm_model *m, const uint8_t *data, size_t len,
                   double train_frac, uint64_t seed);

int nanlm_lm_encode(const nanlm_lm_model *m, const uint8_t ids[3],
                    nanlm_vec *out);
int nanlm_lm_predict(const nanlm_lm_model *m, const nanlm_vec *query,
                     uint32_t *out_byte);
uint32_t nanlm_lm_hash(const nanlm_vec *v);

int nanlm_lm_emit_nas(const nanlm_lm_model *m, const char *path);
int nanlm_lm_write_expected(const nanlm_lm_model *m, const char *path);
int nanlm_lm_write_manifest(const nanlm_lm_model *m, const char *path,
                            uint64_t seed, double train_frac);

void nanlm_lm_zero_positions(nanlm_lm_model *m);
int nanlm_lm_swap_prototypes(nanlm_lm_model *m, uint32_t a, uint32_t b);
/* Swap two prototypes that actually occur in the query predictions. */
int nanlm_lm_swap_predicted(nanlm_lm_model *m);

#ifdef __cplusplus
}
#endif

#endif /* NANLM_LM_H */
