/* nanlm_asm.h - small standalone assembler for NaNLM's generated .nas subset. */
#ifndef NANLM_ASM_H
#define NANLM_ASM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Assemble SRC into DST (.nan text). Returns 0 on success, -1 on error.
 * On error, err receives a short line-numbered message. */
int nanlm_assemble_file(const char *src, const char *dst, char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif /* NANLM_ASM_H */
