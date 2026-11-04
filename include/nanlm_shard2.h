/* nanlm_shard2.h - signed additive count shards.
 *
 * A shard is (3-byte context, next byte, int32 count). Counts are added with
 * two's-complement wrapping, so the set of shards is a group:
 *
 *   S + (-S) == empty
 *   (A + B) + C == A + (B + C)
 *   A + B == B + A
 *
 * That gives exact unlearning: a takedown is a negated shard, not a retrain.
 * The model compiler clamps negative totals when computing probabilities; the
 * shard algebra itself stays invertible.
 */
#ifndef NANLM_SHARD2_H
#define NANLM_SHARD2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t key;   /* a<<16 | b<<8 | c */
    uint8_t next;
    uint8_t pad;
    int32_t count;  /* signed; merge is modular addition */
} nanlm_shard2_rec;

int nanlm_shard2_build(const uint8_t *data, size_t start, size_t end,
                       nanlm_shard2_rec **out_rec, size_t *out_n);
int nanlm_shard2_write(const char *path, const nanlm_shard2_rec *rec, size_t n);
int nanlm_shard2_read(const char *path, nanlm_shard2_rec **out_rec, size_t *out_n);
int nanlm_shard2_merge(const char *out_path, const char **in_paths, int n_in);
int nanlm_shard2_negate(const char *out_path, const char *in_path);
int nanlm_shard2_stats(const char *path, uint64_t *keys, int64_t *total,
                       uint64_t *records, int64_t *min_count,
                       int64_t *max_count);

#ifdef __cplusplus
}
#endif

#endif /* NANLM_SHARD2_H */
