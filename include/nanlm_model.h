/* nanlm_model.h - first NaNLM model: neutral-word class prototypes. */
#ifndef NANLM_MODEL_H
#define NANLM_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "nanlm_vec.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t dim;       /* bits */
    uint32_t words;     /* dim / 32 */
    uint32_t classes;
    uint32_t nqueries;
    nanlm_vec **prototypes; /* classes vectors */
    nanlm_vec **queries;    /* nqueries vectors */
    uint8_t *expected;      /* nqueries class labels */
} nanlm_model;

nanlm_model *nanlm_model_new(uint32_t dim, uint32_t classes,
                             uint32_t nqueries);
void nanlm_model_free(nanlm_model *m);

/* Deterministic toy task: each class is a random centroid, training samples
 * are noisy copies, queries are held-out noisy copies. The prototype is the
 * per-bit majority of the training samples. */
int nanlm_model_train_toy(nanlm_model *m, uint64_t seed,
                          uint32_t samples_per_class, uint32_t noise_bits);

/* Nearest prototype by Hamming distance. Returns 0 on success, -1 on error. */
int nanlm_model_predict(const nanlm_model *m, const nanlm_vec *query,
                        uint32_t *out_class);

/* Write the model as editable NaN16 assembly. */
int nanlm_model_emit_nas(const nanlm_model *m, const char *path);
/* Write two hex digits per query, newline-terminated. */
int nanlm_model_write_expected(const nanlm_model *m, const char *path);
/* Write a small key/value manifest. */
int nanlm_model_write_manifest(const nanlm_model *m, const char *path,
                               uint64_t seed, uint32_t samples_per_class,
                               uint32_t noise_bits);

#ifdef __cplusplus
}
#endif

#endif /* NANLM_MODEL_H */
