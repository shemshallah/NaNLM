/* nanlm_mixer.h - PAQ/lpaq-style multi-order context mixer in integer C99.
 *
 * Components:
 *   1. Template bank (orders 0..8, skip-grams, ctype, word0/word1 hashes,
 *      code indentation/bracket state).
 *   2. Fixed-size 2-way associative hashed count tables with 16-bit check
 *      tags and signed 32-bit counts (mergeable by integer addition).
 *   3. Optional per-document local count cache + rolling match model.
 *   4. Regime-routed 16-bit fixed-point logistic mixer (64 regimes).
 */
#ifndef NANLM_MIXER_H
#define NANLM_MIXER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NANLM_MIX_TEMPLATES 12u
/* 12 global templates + 3 doc-local templates (order 2,3,4) + 1 match model */
#define NANLM_MIX_INPUTS 16u
#define NANLM_MIX_REGIMES 512u
#define NANLM_MIX_WAYS 2u

typedef struct {
    uint16_t check;
    uint16_t pad;
    int32_t n0;
    int32_t n1;
} nanlm_mix_cell;

typedef struct {
    uint32_t table_bits;
    uint32_t buckets; /* 1u << table_bits */
    nanlm_mix_cell *cells; /* NANLM_MIX_TEMPLATES * buckets * NANLM_MIX_WAYS */

    uint32_t doc_bits;
    uint32_t doc_buckets;
    nanlm_mix_cell *doc_cells; /* 3 * doc_buckets * NANLM_MIX_WAYS */

    /* 64 regimes x 16 inputs, Q8 fixed-point weights */
    int16_t weights[NANLM_MIX_REGIMES][NANLM_MIX_INPUTS];

    /* Rolling byte history and hash index for the match model */
    uint8_t *ring;
    uint32_t ring_mask;
    uint32_t ring_pos;
    uint32_t *match_tab;
    uint32_t match_mask;

    /* Running parser state for word and code templates */
    uint64_t word0_hash;
    uint64_t word1_hash;
    uint32_t indent;
    uint32_t bracket_depth;
    uint8_t line_start;
    int adapt_mixer; /* when 0, use fixed equal-weight mixing; when 1, train */
} nanlm_mixer;

typedef struct {
    uint64_t eval_bytes;
    double bpc;
    double top1;
    double ngram_bpc;
    double ngram_top1;
    size_t model_bytes;
    double hutter_total_bpc;
} nanlm_mixer_report;

nanlm_mixer *nanlm_mixer_new(uint32_t table_bits, uint32_t doc_bits);
void nanlm_mixer_free(nanlm_mixer *m);
void nanlm_mixer_reset_doc(nanlm_mixer *m);

/* Predict probability of bit=1 in [1..65535] (Q16). */
uint16_t nanlm_mixer_predict_bit(const nanlm_mixer *m, uint32_t prefix,
                                 uint32_t bit_index,
                                 int16_t stretched_out[NANLM_MIX_INPUTS],
                                 uint32_t *regime_out);

/* Predict full 256-byte probability distribution and argmax byte. */
uint8_t nanlm_mixer_predict_byte(const nanlm_mixer *m, double probs[256]);

/* Observe one byte; update_global=1 during training, update_doc=1 for doc cache. */
void nanlm_mixer_observe_byte(nanlm_mixer *m, uint8_t byte, int update_global,
                              int update_doc);

/* Merge two mixer count tables with identical geometry by signed addition. */
int nanlm_mixer_merge_counts(nanlm_mixer *dst, const nanlm_mixer *src,
                             int sign);

/* Train on [0..train_len) and evaluate on [train_len..len).
 * If log_path != NULL, writes a TSV compatible with experiments/eval_stream.py.
 */
int nanlm_mixer_train_eval(const uint8_t *data, size_t len, double train_frac,
                           uint32_t table_bits, int use_doc_cache,
                           int online_adapt, const char *log_path,
                           nanlm_mixer_report *out);

#ifdef __cplusplus
}
#endif

#endif /* NANLM_MIXER_H */
