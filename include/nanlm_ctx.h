/* nanlm_ctx.h - M2a: context binding + bundling model.
 *
 * A class prototype is:
 *   majority(bind(codebook[c][0], P0),
 *            bind(codebook[c][1], P1),
 *            bind(codebook[c][2], P2))
 *
 * The NaN16 runtime receives only the codebook, positions, and three byte
 * IDs per query. It must reconstruct the query vector. The differential test
 * compares both the class and an FNV-1a hash of the encoded vector, so a
 * runtime that skips binding cannot pass on labels alone.
 */
#ifndef NANLM_CTX_H
#define NANLM_CTX_H

#include <stddef.h>
#include <stdint.h>

#include "nanlm_vec.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NANLM_CTX_LEN 3u

typedef struct {
    uint32_t dim;
    uint32_t words;
    uint32_t classes;
    uint32_t nqueries;
    nanlm_vec **codebook;  /* classes * NANLM_CTX_LEN */
    nanlm_vec **positions; /* NANLM_CTX_LEN */
    nanlm_vec **prototypes;
    nanlm_vec **queries;
    nanlm_vec *tmp[NANLM_CTX_LEN];
    uint8_t *query_ids; /* nqueries * NANLM_CTX_LEN */
    uint8_t *expected;
    uint32_t *hashes;
} nanlm_ctx_model;

nanlm_ctx_model *nanlm_ctx_model_new(uint32_t dim, uint32_t classes,
                                     uint32_t nqueries);
void nanlm_ctx_model_free(nanlm_ctx_model *m);

/* Build a deterministic synthetic model: one codebook vector per class and
 * position, prototypes by bind+bundle, and held-out queries that reuse the
 * same codebook entries. */
int nanlm_ctx_build(nanlm_ctx_model *m, uint64_t seed);

int nanlm_ctx_encode(const nanlm_ctx_model *m, const uint8_t ids[3],
                     nanlm_vec *out);
int nanlm_ctx_predict(const nanlm_ctx_model *m, const nanlm_vec *query,
                      uint32_t *out_class);
uint32_t nanlm_ctx_hash(const nanlm_vec *v);

int nanlm_ctx_emit_nas(const nanlm_ctx_model *m, const char *path);
int nanlm_ctx_write_expected(const nanlm_ctx_model *m, const char *path);
int nanlm_ctx_write_manifest(const nanlm_ctx_model *m, const char *path,
                             uint64_t seed);

/* Negative-gate mutators. Call after nanlm_ctx_build(), before emit. */
void nanlm_ctx_zero_positions(nanlm_ctx_model *m);
int nanlm_ctx_swap_prototypes(nanlm_ctx_model *m, uint32_t a, uint32_t b);

#ifdef __cplusplus
}
#endif

#endif /* NANLM_CTX_H */
