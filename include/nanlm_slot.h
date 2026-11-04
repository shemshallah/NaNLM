/* nanlm_slot.h - simplex slot memory: one prototype per observed 3-gram.
 *
 * Key = [C[a] | C[b] | C[c]] with 255-bit simplex codewords, one slot per
 * context position. Pairwise codeword distance is exactly 128, so Hamming
 * distance over the key is a weighted count of mismatching context bytes.
 *
 * Training is additive counting: (3-gram, next byte) -> count. That makes it
 * shard-mergeable and suitable for volunteer / object-store training.
 */
#ifndef NANLM_SLOT_H
#define NANLM_SLOT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NANLM_SLOT_CTX_BYTES
#define NANLM_SLOT_CTX_BYTES 3u
#endif
#ifndef NANLM_SLOT_N
#define NANLM_SLOT_N (2u * NANLM_SLOT_CTX_BYTES)
#endif
#define NANLM_SLOT_CW 8u
#define NANLM_SLOT_VW (NANLM_SLOT_N * NANLM_SLOT_CW)
#define NANLM_SLOT_BITS (NANLM_SLOT_N * 256u)

#if NANLM_SLOT_CTX_BYTES == 3
typedef uint32_t nanlm_slot_key;
#elif NANLM_SLOT_CTX_BYTES == 8
typedef uint64_t nanlm_slot_key;
#else
#error "slot context must be 3 or 8 bytes"
#endif

typedef struct {
    nanlm_slot_key key;   /* packed context bytes, oldest byte high */
    uint8_t next;
    uint16_t weight;
} nanlm_slot_count;

typedef struct {
    uint32_t nkeys;
    uint32_t *keys;    /* nkeys * NANLM_SLOT_VW payload words */
    uint8_t *labels;   /* nkeys majority next byte */
    uint16_t *counts;  /* nkeys * 256 next-byte counts */
    uint32_t *totals;  /* nkeys */
    uint32_t unigram[256];

    uint32_t nqueries;
    uint8_t *qids;     /* nqueries * 6: bytes then ctype bytes */
    uint8_t *predicted;/* nqueries C prediction */
    uint32_t *hashes;  /* nqueries FNV over query key */

    uint32_t calib_n, eval_n;
    double hdc_top1, hdc_bpc, ngram_top1, ngram_bpc;
    double alpha;
} nanlm_slot_model;

nanlm_slot_model *nanlm_slot_build(const uint8_t *data, size_t len,
                                   double train_frac, uint32_t nqueries);
nanlm_slot_model *nanlm_slot_build_counts(const nanlm_slot_count *counts,
                                          size_t ncounts,
                                          const uint8_t *eval_data,
                                          size_t eval_len,
                                          double train_frac,
                                          uint32_t nqueries);
void nanlm_slot_free(nanlm_slot_model *m);

int nanlm_slot_encode(const nanlm_slot_model *m,
                      const uint8_t ctx[NANLM_SLOT_CTX_BYTES],
                      uint32_t out[NANLM_SLOT_VW]);
int nanlm_slot_predict(const nanlm_slot_model *m,
                       const uint8_t ctx[NANLM_SLOT_CTX_BYTES],
                       uint8_t *out_next);
uint32_t nanlm_slot_hash(const uint32_t key[NANLM_SLOT_VW]);

int nanlm_slot_emit_nas(const nanlm_slot_model *m, const char *path);
int nanlm_slot_write_expected(const nanlm_slot_model *m, const char *path);
int nanlm_slot_write_manifest(const nanlm_slot_model *m, const char *path,
                              uint64_t seed, double train_frac);

#ifdef __cplusplus
}
#endif

#endif /* NANLM_SLOT_H */
