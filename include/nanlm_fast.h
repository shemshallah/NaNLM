/* nanlm_fast.h - native kernels that must match the VM reference. */
#ifndef NANLM_FAST_H
#define NANLM_FAST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t nanlm_fast_hamming32(const uint32_t *a, const uint32_t *b, size_t n);
uint32_t nanlm_fast_fnv32(const uint32_t *a, size_t n, uint32_t seed);

#ifdef __cplusplus
}
#endif

#endif /* NANLM_FAST_H */
