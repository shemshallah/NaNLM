#include "nanlm_vm.h"

#include <stdlib.h>
#include <string.h>

#include "nan16_word.h"

struct nanlm_vm {
    uint64_t *mem;
    size_t nwords;
    uint32_t r[16];
    uint32_t pc;
    int k;
    int halted;
    uint64_t steps;
    char trap[32];
    int load_err;
    FILE *out;
};

static void vm_trap(nanlm_vm *v, char kind, uint32_t pc) {
    snprintf(v->trap, sizeof(v->trap), "!%c%04X", kind, (unsigned)pc);
    v->pc = pc;
    v->halted = 1;
}

nanlm_vm *nanlm_vm_new(size_t words) {
    nanlm_vm *v;
    if (words == 0) words = 1u << 22;
    if ((words & (words - 1)) != 0) return NULL;
    v = (nanlm_vm *)calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->mem = (uint64_t *)calloc(words, sizeof(*v->mem));
    if (!v->mem) {
        free(v);
        return NULL;
    }
    v->nwords = words;
    v->out = stdout;
    return v;
}

void nanlm_vm_free(nanlm_vm *v) {
    if (!v) return;
    free(v->mem);
    free(v);
}

void nanlm_vm_set_output(nanlm_vm *v, FILE *out) {
    if (v) v->out = out ? out : stdout;
}

int nanlm_vm_load_error(const nanlm_vm *v) { return v ? v->load_err : -1; }

static int parse_line(const char *line, uint32_t *near, uint32_t *payload,
                      uint32_t *far_swapped) {
    return sscanf(line, "%4x_%8x_%4x", near, payload, far_swapped) == 3;
}

int nanlm_vm_load_text(nanlm_vm *v, const char *path) {
    FILE *f;
    char line[512];
    size_t idx = 0;
    if (!v || !path) return -1;
    f = fopen(path, "r");
    if (!f) {
        v->load_err = -1;
        return -1;
    }
    v->load_err = 0;
    while (fgets(line, sizeof(line), f)) {
        char *semi = strchr(line, ';');
        uint32_t near, payload, far_swapped, far, n;
        char *p = line;
        if (semi) *semi = '\0';
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n' || *p == '\r') continue;
        if (!parse_line(p, &near, &payload, &far_swapped)) {
            v->load_err = (int)idx + 1;
            fclose(f);
            return -1;
        }
        far = ((far_swapped & 0xFFu) << 8) | (far_swapped >> 8);
        if (idx >= v->nwords) {
            v->load_err = (int)idx + 1;
            fclose(f);
            return -1;
        }
        if (nan16_head_valid(near) && nan16_head_valid(far)) {
            n = (nan16_head_class(near) << 2) | nan16_head_class(far);
            v->mem[idx] = nan16_make(payload, n);
        } else {
            v->mem[idx] = 0; /* invalid; executes as !G if reached */
        }
        idx++;
    }
    fclose(f);
    return 0;
}

int nanlm_vm_load_words(nanlm_vm *v, const uint64_t *words, size_t n) {
    if (!v || !words || n > v->nwords) return -1;
    memcpy(v->mem, words, n * sizeof(*words));
    return 0;
}

int nanlm_vm_run(nanlm_vm *v, uint64_t max_steps) {
    uint32_t *r;
    uint64_t *mem;
    uint32_t pc;
    int k;
    uint64_t steps;
    size_t mask;
    uint64_t done;

    if (!v) return NANLM_VM_TRAP;
    if (v->halted) return NANLM_VM_HALTED;
    r = v->r;
    mem = v->mem;
    pc = v->pc;
    k = v->k;
    steps = v->steps;
    mask = v->nwords - 1;

#define SETK(z, l) (k = 2 * ((z) != 0) + ((l) != 0))
#define LEAVE(code)                                                            \
    do {                                                                       \
        v->pc = pc;                                                            \
        v->k = k;                                                              \
        v->steps = steps;                                                      \
        return (code);                                                         \
    } while (0)

    for (done = 0; done < max_steps; done++) {
        uint64_t w = mem[pc & mask];
        uint32_t at, p, op, d, a, imm, A, B, res, n;
        if (!nan16_gate(w)) {
            vm_trap(v, 'G', pc + 1);
            return NANLM_VM_TRAP;
        }
        n = nan16_n(w);
        at = pc;
        pc++;
        p = nan16_payload(w);
        op = p >> 24;
        if (op == 0xFF) {
            v->halted = 1;
            v->pc = pc;
            v->k = k;
            v->steps = steps + 1;
            return NANLM_VM_HALTED;
        }
        if (op > 0x45) {
            vm_trap(v, 'O', pc);
            v->steps = steps + 1;
            return NANLM_VM_TRAP;
        }
        steps++;
        if (op != 0 && !((n >> k) & 1u)) continue;
        d = (p >> 20) & 15u;
        a = (p >> 16) & 15u;
        imm = p & 0xFFFFu;
        A = r[a];
        B = r[imm & 15u];
        switch (op) {
        case 0x00:
            res = nan16_bool(n, A, B);
            r[d] = res;
            SETK(res == 0, 0);
            break;
        case 0x01: r[d] = imm; break;
        case 0x02: r[d] = (r[d] & 0xFFFFu) | (imm << 16); break;
        case 0x03: {
            uint64_t s = (uint64_t)A + B;
            r[d] = (uint32_t)s;
            SETK((uint32_t)s == 0, (s >> 32) != 0);
            break;
        }
        case 0x04:
            res = A - B;
            r[d] = res;
            SETK(res == 0, A < B);
            break;
        case 0x05:
            res = A << (imm & 31u);
            r[d] = res;
            SETK(res == 0, 0);
            break;
        case 0x06:
            res = A >> (imm & 31u);
            r[d] = res;
            SETK(res == 0, 0);
            break;
        case 0x07: SETK(A == B, A < B); break;
        case 0x08: r[d] = nan16_payload(mem[(A + imm) & mask]); break;
        case 0x09: mem[(A + imm) & mask] = nan16_make(r[d], NAN16_N_DATA); break;
        case 0x0A: pc = imm; break;
        case 0x0B: r[15] = pc; pc = imm; break;
        case 0x0C: pc = r[15]; break;
        case 0x0D: pc = A; break;
        case 0x0F: break;
        case 0x14: mem[(A + imm) & mask] = nan16_bswap(mem[(A + imm) & mask]); break;
        case 0x15: fputc((int)(r[d] & 0xFFu), v->out); break;
        case 0x16: mem[A & mask] = nan16_make(r[d], B & 15u); break;
        case 0x20:
            res = A * B;
            r[d] = res;
            SETK(res == 0, 0);
            break;
        case 0x2D: {
            uint32_t dst = r[d] & (uint32_t)mask;
            uint32_t src = A & (uint32_t)mask;
            uint32_t cnt = B;
            if (cnt) {
                if (((dst - src) & (uint32_t)mask) < cnt) {
                    uint32_t s = (src + cnt - 1) & (uint32_t)mask;
                    uint32_t t = (dst + cnt - 1) & (uint32_t)mask;
                    for (uint32_t i = 0; i < cnt; i++) {
                        mem[t] = mem[s];
                        s = (s - 1) & (uint32_t)mask;
                        t = (t - 1) & (uint32_t)mask;
                    }
                } else {
                    for (uint32_t i = 0; i < cnt; i++) {
                        mem[dst] = mem[src];
                        dst = (dst + 1) & (uint32_t)mask;
                        src = (src + 1) & (uint32_t)mask;
                    }
                }
            }
            break;
        }
        case 0x2E: {
            uint32_t dst = r[d];
            uint64_t wd = nan16_make(A, NAN16_N_DATA);
            for (uint32_t i = 0; i < B; i++) mem[(dst + i) & mask] = wd;
            break;
        }
        default:
            vm_trap(v, 'O', pc);
            v->steps = steps;
            return NANLM_VM_TRAP;
        }
        (void)at;
    }
    LEAVE(NANLM_VM_BUDGET);
#undef LEAVE
#undef SETK
}

uint64_t nanlm_vm_steps(const nanlm_vm *v) { return v ? v->steps : 0; }
uint32_t nanlm_vm_pc(const nanlm_vm *v) { return v ? v->pc : 0; }
const char *nanlm_vm_trap(const nanlm_vm *v) { return v ? v->trap : ""; }
uint32_t nanlm_vm_reg(const nanlm_vm *v, unsigned i) {
    return v ? v->r[i & 15u] : 0;
}
uint64_t nanlm_vm_word(const nanlm_vm *v, size_t i) {
    return (v && i < v->nwords) ? v->mem[i] : 0;
}
int nanlm_vm_flags(const nanlm_vm *v) { return v ? v->k : 0; }
