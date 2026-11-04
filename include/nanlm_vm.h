/* nanlm_vm.h - minimal standalone NaN16 runtime for NaNLM model images.
 *
 * This is not the NaN/0S kernel. It implements the small NaN16 subset that
 * NaNLM emits: BOOL, MOVI/MOVHI, ADD/SUB, SHL/SHR, CMP, LD/ST, JMP/CALL/RET,
 * JMPR, NOP, PUTC, MUL, HALT, plus the both-face gate.
 */
#ifndef NANLM_VM_H
#define NANLM_VM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    NANLM_VM_BUDGET = 0,
    NANLM_VM_TRAP = 1,
    NANLM_VM_HALTED = 3
};

typedef struct nanlm_vm nanlm_vm;

nanlm_vm *nanlm_vm_new(size_t words);
void nanlm_vm_free(nanlm_vm *v);

/* Load .nan text: NNNN_PPPPPPPP_FFFF (near head, payload, byte-swapped far). */
int nanlm_vm_load_text(nanlm_vm *v, const char *path);
int nanlm_vm_load_words(nanlm_vm *v, const uint64_t *words, size_t n);
int nanlm_vm_load_error(const nanlm_vm *v);

int nanlm_vm_run(nanlm_vm *v, uint64_t max_steps);
void nanlm_vm_set_output(nanlm_vm *v, FILE *out);

uint64_t nanlm_vm_steps(const nanlm_vm *v);
uint32_t nanlm_vm_pc(const nanlm_vm *v);
const char *nanlm_vm_trap(const nanlm_vm *v);
uint32_t nanlm_vm_reg(const nanlm_vm *v, unsigned i);
uint64_t nanlm_vm_word(const nanlm_vm *v, size_t i);
int nanlm_vm_flags(const nanlm_vm *v);

#ifdef __cplusplus
}
#endif

#endif /* NANLM_VM_H */
