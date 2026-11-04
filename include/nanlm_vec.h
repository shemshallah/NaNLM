/* nanlm_vec.h - high-dimensional neutral-word vectors for NaNLM.
 *
 * A vector is D bits stored as D/32 NaN16 words. Every data word is neutral
 * (n=10, both heads +q). Boolean operations read payloads and write neutral
 * words; the truth table is the operation's n, exactly as in NaN16 BOOL.
 */
#ifndef NANLM_VEC_H
#define NANLM_VEC_H

#include <stddef.h>
#include <stdint.h>

#include "nan16_word.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t words;
    nan16_word *w;
} nanlm_vec;

nanlm_vec *nanlm_vec_new(size_t words);
void nanlm_vec_free(nanlm_vec *v);

/* Gate-valid for every word. */
int nanlm_vec_valid(const nanlm_vec *v);
/* Gate-valid and every word has n=10. */
int nanlm_vec_is_data(const nanlm_vec *v);

void nanlm_vec_zero(nanlm_vec *v);
void nanlm_vec_fill(nanlm_vec *v, nan16_word word);
void nanlm_vec_random(nanlm_vec *v, uint64_t *state);

/* dst = BOOL.n(a, b), word by word. Writes neutral data words. */
int nanlm_vec_bool(nanlm_vec *dst, const nanlm_vec *a, const nanlm_vec *b,
                   unsigned n);
/* XOR binding (n=6). */
int nanlm_vec_bind(nanlm_vec *dst, const nanlm_vec *a, const nanlm_vec *b);
/* Majority of three vectors: (a&b)|(a&c)|(b&c). */
int nanlm_vec_bundle3(nanlm_vec *dst, const nanlm_vec *a, const nanlm_vec *b,
                      const nanlm_vec *c);
/* BSWAPM each neutral data word. Reverses the payload bytes. */
int nanlm_vec_bswap(nanlm_vec *dst, const nanlm_vec *a);

uint64_t nanlm_vec_hamming(const nanlm_vec *a, const nanlm_vec *b);
uint64_t nanlm_vec_popcount(const nanlm_vec *a);
int nanlm_vec_equal(const nanlm_vec *a, const nanlm_vec *b);

#ifdef __cplusplus
}
#endif

#endif /* NANLM_VEC_H */
