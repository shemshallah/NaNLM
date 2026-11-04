#include "nanlm_asm.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nan16_word.h"

#define MAX_SYMS 512

typedef struct {
    char name[64];
    uint32_t value;
} Sym;

typedef struct {
    Sym item[MAX_SYMS];
    int n;
} SymTab;

typedef struct {
    const char *name;
    unsigned op;
    const char *form;
} Ins;

static const Ins INS[] = {
    {"BOOL", 0x00, "dab"},  {"MOVI", 0x01, "di"},   {"MOVHI", 0x02, "di"},
    {"ADD", 0x03, "dab"},   {"SUB", 0x04, "dab"},   {"SHL", 0x05, "dai"},
    {"SHR", 0x06, "dai"},   {"CMP", 0x07, "ab"},    {"LD", 0x08, "dai"},
    {"ST", 0x09, "dai"},    {"JMP", 0x0A, "i"},     {"CALL", 0x0B, "i"},
    {"RET", 0x0C, ""},      {"JMPR", 0x0D, "a"},    {"NOP", 0x0F, ""},
    {"BSWAPM", 0x14, "ai"}, {"PUTC", 0x15, "d"},    {"STW", 0x16, "dab"},
    {"MUL", 0x20, "dab"},
    {"BCOPY", 0x2D, "dab"}, {"BFILL", 0x2E, "dab"}, {"HALT", 0xFF, ""},
};

static const struct {
    const char *name;
    unsigned n;
} PREDS[] = {
    {"ALWAYS", 15}, {"NEVER", 0}, {"EQ", 4}, {"NE", 3},
    {"LT", 2},      {"GE", 5},    {"GT", 1}, {"LE", 6},
};

static void set_err(char *err, size_t errlen, size_t line, const char *msg) {
    if (err && errlen) {
        snprintf(err, errlen, "line %zu: %s", line, msg);
    }
}

static char *trim(char *s) {
    char *end;
    while (*s == ' ' || *s == '\t') s++;
    end = s + strlen(s);
    while (end > s && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r')) {
        *--end = '\0';
    }
    return s;
}

static int read_lines(const char *path, char ***out_lines, size_t *out_n,
                      char **out_buf) {
    FILE *f;
    long sz;
    char *buf;
    char **lines;
    size_t n = 0, cap = 1024, i;

    f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    sz = ftell(f);
    if (sz < 0) { fclose(f); return -1; }
    rewind(f);
    buf = (char *)malloc((size_t)sz + 1);
    lines = (char **)malloc(cap * sizeof(*lines));
    if (!buf || !lines) {
        free(buf);
        free(lines);
        fclose(f);
        return -1;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        free(lines);
        fclose(f);
        return -1;
    }
    fclose(f);
    buf[sz] = '\0';

    lines[n++] = buf;
    for (i = 0; i < (size_t)sz; i++) {
        if (buf[i] == '\n') {
            buf[i] = '\0';
            if (n == cap) {
                char **tmp;
                cap *= 2;
                tmp = (char **)realloc(lines, cap * sizeof(*lines));
                if (!tmp) {
                    free(buf);
                    free(lines);
                    return -1;
                }
                lines = tmp;
            }
            lines[n++] = buf + i + 1;
        }
    }

    /* Strip comments and trim every line. */
    for (i = 0; i < n; i++) {
        char *semi = strchr(lines[i], ';');
        if (semi) *semi = '\0';
        lines[i] = trim(lines[i]);
    }
    *out_lines = lines;
    *out_n = n;
    *out_buf = buf;
    return 0;
}

static void free_lines(char **lines, char *buf) {
    free(lines);
    free(buf);
}

static int sym_add(SymTab *t, const char *name, uint32_t value) {
    int i;
    if (t->n >= MAX_SYMS || strlen(name) >= sizeof(t->item[0].name)) return -1;
    for (i = 0; i < t->n; i++) {
        if (strcmp(t->item[i].name, name) == 0) return -1;
    }
    strcpy(t->item[t->n].name, name);
    t->item[t->n].value = value;
    t->n++;
    return 0;
}

static int sym_find(const SymTab *t, const char *name, uint32_t *out) {
    int i;
    for (i = 0; i < t->n; i++) {
        if (strcmp(t->item[i].name, name) == 0) {
            *out = t->item[i].value;
            return 0;
        }
    }
    return -1;
}

static int parse_num(const char *s, uint32_t *out) {
    char *end = NULL;
    unsigned long v;
    if (!s || !*s) return -1;
    v = strtoul(s, &end, 0);
    if (!end || *end != '\0' || v > 0xFFFFFFFFul) return -1;
    *out = (uint32_t)v;
    return 0;
}

static int resolve(const char *s, const SymTab *labels, const SymTab *equs,
                   uint32_t *out) {
    if (*s == '@') return sym_find(labels, s + 1, out);
    if (*s == '#') return sym_find(equs, s + 1, out);
    return parse_num(s, out);
}

static int is_label(const char *s) {
    size_t n = strlen(s);
    size_t i;
    if (n < 2 || s[n - 1] != ':') return 0;
    for (i = 0; i + 1 < n; i++) {
        if (!(isalnum((unsigned char)s[i]) || s[i] == '_')) return 0;
    }
    return 1;
}

static int is_la(const char *s) {
    return strncmp(s, "LA", 2) == 0 &&
           (s[2] == '\0' || s[2] == ' ' || s[2] == '\t' || s[2] == '[');
}

static int find_pred(const char *name, unsigned *out) {
    size_t i;
    for (i = 0; i < sizeof(PREDS) / sizeof(PREDS[0]); i++) {
        if (strcmp(PREDS[i].name, name) == 0) {
            *out = PREDS[i].n;
            return 0;
        }
    }
    return -1;
}

static int find_ins(const char *name, const Ins **out) {
    size_t i;
    for (i = 0; i < sizeof(INS) / sizeof(INS[0]); i++) {
        if (strcmp(INS[i].name, name) == 0) {
            *out = &INS[i];
            return 0;
        }
    }
    return -1;
}

static int parse_mnemonic(char *line, char *base, size_t baselen,
                          unsigned *direct_n, unsigned *pred_n,
                          char **rest) {
    char *p = line;
    size_t n = 0;
    *direct_n = 16u;
    *pred_n = 15u;
    while ((isalpha((unsigned char)*p) || *p == '_') && n + 1 < baselen) {
        base[n++] = *p++;
    }
    base[n] = '\0';
    if (n == 0) return -1;
    if (*p == '.') {
        unsigned v = 0;
        p++;
        if (!isdigit((unsigned char)*p)) return -1;
        while (isdigit((unsigned char)*p)) {
            v = v * 10u + (unsigned)(*p - '0');
            p++;
        }
        if (v > 15u) return -1;
        *direct_n = v;
    }
    if (*p == '[') {
        char pred[32];
        size_t k = 0;
        p++;
        while (*p && *p != ']' && k + 1 < sizeof(pred)) pred[k++] = *p++;
        pred[k] = '\0';
        if (*p != ']' || find_pred(pred, pred_n) != 0) return -1;
        p++;
    }
    while (*p == ' ' || *p == '\t') p++;
    *rest = p;
    return 0;
}

static int split_args(char *rest, char *args[], int max) {
    int n = 0;
    char *p = rest;
    while (*p && n < max) {
        char *start = p;
        while (*p && *p != ',') p++;
        if (*p == ',') *p++ = '\0';
        args[n++] = trim(start);
    }
    if (*p) return -1;
    return n;
}

static int parse_reg(const char *s, unsigned *out) {
    char *end = NULL;
    unsigned long v;
    if (!s || s[0] != 'r' || !isdigit((unsigned char)s[1])) return -1;
    v = strtoul(s + 1, &end, 10);
    if (!end || *end != '\0' || v > 15) return -1;
    *out = (unsigned)v;
    return 0;
}

typedef struct {
    uint64_t *w;
    size_t n;
    size_t cap;
} Out;

static int out_emit(Out *o, uint64_t word) {
    if (o->n == o->cap) {
        size_t cap = o->cap ? o->cap * 2 : 4096;
        uint64_t *tmp = (uint64_t *)realloc(o->w, cap * sizeof(*tmp));
        if (!tmp) return -1;
        o->w = tmp;
        o->cap = cap;
    }
    o->w[o->n++] = word;
    return 0;
}

static int emit_instruction(Out *o, const Ins *ins, unsigned n, unsigned d,
                            unsigned a, unsigned imm) {
    uint32_t payload = (ins->op << 24) | ((d & 15u) << 20) |
                       ((a & 15u) << 16) | (imm & 0xFFFFu);
    return out_emit(o, nan16_make(payload, n));
}

int nanlm_assemble_file(const char *src, const char *dst, char *err,
                        size_t errlen) {
    char **lines = NULL;
    char *buf = NULL;
    size_t nlines = 0, i;
    SymTab labels = {0}, equs = {0};
    uint32_t pc = 0;
    Out out = {0, 0, 0};

    if (err && errlen) err[0] = '\0';
    if (!src || !dst) return -1;
    if (read_lines(src, &lines, &nlines, &buf) != 0) {
        if (err && errlen) snprintf(err, errlen, "cannot read %s", src);
        return -1;
    }

    /* Pass 1: labels, equates, and addresses. */
    for (i = 0; i < nlines; i++) {
        char *line = lines[i];
        if (*line == '\0') continue;
        if (is_label(line)) {
            char name[64];
            size_t len = strlen(line);
            if (len - 1 >= sizeof(name)) {
                set_err(err, errlen, i + 1, "label too long");
                goto fail;
            }
            memcpy(name, line, len - 1);
            name[len - 1] = '\0';
            if (sym_add(&labels, name, pc) != 0) {
                set_err(err, errlen, i + 1, "duplicate or invalid label");
                goto fail;
            }
            continue;
        }
        if (strncmp(line, ".equ", 4) == 0 &&
            (line[4] == ' ' || line[4] == '\t')) {
            char name[64], value[64];
            if (sscanf(line + 4, "%63s %63s", name, value) != 2) {
                set_err(err, errlen, i + 1, "bad .equ");
                goto fail;
            }
            {
                uint32_t v;
                if (parse_num(value, &v) != 0 ||
                    sym_add(&equs, name, v) != 0) {
                    set_err(err, errlen, i + 1, "bad .equ value");
                    goto fail;
                }
            }
            continue;
        }
        if (strncmp(line, ".org", 4) == 0 &&
            (line[4] == ' ' || line[4] == '\t')) {
            uint32_t v;
            if (parse_num(trim(line + 4), &v) != 0 || v < pc) {
                set_err(err, errlen, i + 1, "bad .org");
                goto fail;
            }
            pc = v;
            continue;
        }
        if (strncmp(line, ".word", 5) == 0 &&
            (line[5] == ' ' || line[5] == '\t')) {
            pc++;
            continue;
        }
        if (is_la(line)) {
            pc += 2;
            continue;
        }
        pc++;
    }

    /* Pass 2: emit. */
    pc = 0;
    for (i = 0; i < nlines; i++) {
        char *line = lines[i];
        if (*line == '\0' || is_label(line)) continue;
        if (strncmp(line, ".equ", 4) == 0 &&
            (line[4] == ' ' || line[4] == '\t')) {
            continue;
        }
        if (strncmp(line, ".org", 4) == 0 &&
            (line[4] == ' ' || line[4] == '\t')) {
            uint32_t v;
            if (parse_num(trim(line + 4), &v) != 0 || v < pc) {
                set_err(err, errlen, i + 1, "bad .org");
                goto fail;
            }
            while (pc < v) {
                if (out_emit(&out, nan16_make(0, NAN16_N_DATA)) != 0) {
                    set_err(err, errlen, i + 1, "out of memory");
                    goto fail;
                }
                pc++;
            }
            continue;
        }
        if (strncmp(line, ".word", 5) == 0 &&
            (line[5] == ' ' || line[5] == '\t')) {
            uint32_t v;
            if (parse_num(trim(line + 5), &v) != 0) {
                set_err(err, errlen, i + 1, "bad .word");
                goto fail;
            }
            if (out_emit(&out, nan16_make(v, NAN16_N_DATA)) != 0) {
                set_err(err, errlen, i + 1, "out of memory");
                goto fail;
            }
            pc++;
            continue;
        }
        if (is_la(line)) {
            char base[32], *rest;
            unsigned direct_n, pred_n, d, addr;
            char *args[4];
            int na;
            if (parse_mnemonic(line, base, sizeof(base), &direct_n, &pred_n,
                               &rest) != 0 ||
                strcmp(base, "LA") != 0) {
                set_err(err, errlen, i + 1, "bad LA");
                goto fail;
            }
            (void)direct_n;
            na = split_args(rest, args, 4);
            if (na != 2 || parse_reg(args[0], &d) != 0 ||
                resolve(args[1], &labels, &equs, &addr) != 0) {
                set_err(err, errlen, i + 1, "bad LA operands");
                goto fail;
            }
            if (emit_instruction(&out, &INS[1], pred_n, d, 0, addr & 0xFFFFu) !=
                    0 ||
                emit_instruction(&out, &INS[2], pred_n, d, 0,
                                 (addr >> 16) & 0xFFFFu) != 0) {
                set_err(err, errlen, i + 1, "out of memory");
                goto fail;
            }
            pc += 2;
            continue;
        }
        {
            char base[32], *rest;
            unsigned direct_n, pred_n, d = 0, a = 0, imm = 0, n;
            char *args[4];
            int na, k;
            const Ins *ins;
            if (parse_mnemonic(line, base, sizeof(base), &direct_n, &pred_n,
                               &rest) != 0 ||
                find_ins(base, &ins) != 0) {
                set_err(err, errlen, i + 1, "unknown mnemonic");
                goto fail;
            }
            n = direct_n != 16u ? direct_n : pred_n;
            na = split_args(rest, args, 4);
            if (na < 0 || (size_t)na < strlen(ins->form)) {
                set_err(err, errlen, i + 1, "missing operands");
                goto fail;
            }
            if ((size_t)na > strlen(ins->form)) {
                set_err(err, errlen, i + 1, "too many operands");
                goto fail;
            }
            for (k = 0; k < na; k++) {
                switch (ins->form[k]) {
                case 'd':
                    if (parse_reg(args[k], &d) != 0) goto bad_operand;
                    break;
                case 'a':
                    if (parse_reg(args[k], &a) != 0) goto bad_operand;
                    break;
                case 'b': {
                    unsigned b;
                    if (parse_reg(args[k], &b) != 0) goto bad_operand;
                    imm = b;
                    break;
                }
                case 'i':
                    if (resolve(args[k], &labels, &equs, &imm) != 0 ||
                        imm > 0xFFFFu) {
                        goto bad_operand;
                    }
                    break;
                case 'j': {
                    uint32_t v;
                    if (resolve(args[k], &labels, &equs, &v) != 0) {
                        goto bad_operand;
                    }
                    d = (v >> 20) & 15u;
                    a = (v >> 16) & 15u;
                    imm = v & 0xFFFFu;
                    break;
                }
                default: goto bad_operand;
                }
            }
            if (emit_instruction(&out, ins, n, d, a, imm) != 0) {
                set_err(err, errlen, i + 1, "out of memory");
                goto fail;
            }
            pc++;
            continue;
        bad_operand:
            set_err(err, errlen, i + 1, "bad operand");
            goto fail;
        }
    }

    {
        FILE *f = fopen(dst, "w");
        size_t k;
        if (!f) {
            set_err(err, errlen, nlines, "cannot write output");
            goto fail;
        }
        for (k = 0; k < out.n; k++) {
            uint64_t w = out.w[k];
            uint32_t near = nan16_near(w);
            uint32_t far = nan16_far(w);
            uint32_t payload = nan16_payload(w);
            if (fprintf(f, "%04X_%08X_%02X%02X   ; %03zu\n", near, payload,
                        far & 0xFFu, (far >> 8) & 0xFFu, k) < 0) {
                fclose(f);
                set_err(err, errlen, nlines, "write error");
                goto fail;
            }
        }
        if (fclose(f) != 0) {
            set_err(err, errlen, nlines, "close error");
            goto fail;
        }
    }

    free(out.w);
    free_lines(lines, buf);
    return 0;

fail:
    free(out.w);
    free_lines(lines, buf);
    return -1;
}
