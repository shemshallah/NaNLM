/* nanlm_runtime.h - shared source of the context-model NaN16 runtime.
 *
 * The runtime expects .equ NVEC, VW, NQ, CTX, BEST_INIT and data labels
 * codebook, positions, prototypes, query_ids, tmp0, tmp1, tmp2, qbuf.
 */
#ifndef NANLM_RUNTIME_H
#define NANLM_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

const char *nanlm_runtime_source(void);

#ifdef __cplusplus
}
#endif

#endif /* NANLM_RUNTIME_H */
