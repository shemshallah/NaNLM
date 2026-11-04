/* nanlm_shard.h - additive, mergeable training shards for NaNLM.
 *
 * A shard is a sorted list of (3-gram key, next byte, count) records. Training
 * is counting; merging is integer addition. That makes training commutative,
 * order-independent, and safe to run on volunteers / object storage.
 */
#ifndef NANLM_SHARD_H
#define NANLM_SHARD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t key;   /* a<<16 | b<<8 | c */
    uint8_t next;
    uint8_t pad;
    uint16_t count; /* saturating 65535 */
} nanlm_shard_rec;

int nanlm_shard_build(const uint8_t *data, size_t start, size_t end,
                      nanlm_shard_rec **out_rec, size_t *out_n);
int nanlm_shard_write(const char *path, const nanlm_shard_rec *rec, size_t n);
int nanlm_shard_read(const char *path, nanlm_shard_rec **out_rec, size_t *out_n);
int nanlm_shard_merge(const char *out_path, const char **in_paths, int n_in);
int nanlm_shard_stats(const char *path, uint64_t *keys, uint64_t *total,
                      uint64_t *records);

#ifdef __cplusplus
}
#endif

#endif /* NANLM_SHARD_H */
