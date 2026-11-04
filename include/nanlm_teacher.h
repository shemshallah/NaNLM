/* nanlm_teacher.h - additive teacher-distillation records.
 *
 * A teacher record is (8-byte context, next byte, weight). A teacher may emit
 * several records for the same context (top-k bytes). Merge is integer
 * addition, so teacher shards are commutative and volunteer-safe.
 */
#ifndef NANLM_TEACHER_H
#define NANLM_TEACHER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t context; /* 8 bytes: ctx[0] high byte ... ctx[7] low byte */
    uint8_t next;
    uint8_t pad;
    uint16_t weight;
} nanlm_teacher_rec;

uint64_t nanlm_teacher_key(const uint8_t ctx[8]);

/* Fixture teacher: an order-3 smoothed distribution over exact 8-byte
 * contexts. This is not a smart teacher; it exercises the pipeline. */
int nanlm_teacher_build_ngram(const uint8_t *data, size_t start, size_t end,
                              uint32_t top_k, nanlm_teacher_rec **out,
                              size_t *out_n);

int nanlm_teacher_write(const char *path, const nanlm_teacher_rec *rec,
                        size_t n);
int nanlm_teacher_read(const char *path, nanlm_teacher_rec **out, size_t *out_n);
int nanlm_teacher_merge(const char *out_path, const char **in_paths, int n_in);
int nanlm_teacher_stats(const char *path, uint64_t *contexts,
                        uint64_t *records, uint64_t *weight);

#ifdef __cplusplus
}
#endif

#endif /* NANLM_TEACHER_H */
