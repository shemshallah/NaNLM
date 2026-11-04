#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanlm_teacher.h"

static unsigned char *read_file(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb");
    long sz;
    unsigned char *b;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    sz = ftell(f);
    if (sz <= 0) { fclose(f); return NULL; }
    rewind(f);
    b = (unsigned char *)malloc((size_t)sz);
    if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) {
        free(b);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

static int cmd_build(int argc, char **argv) {
    const char *corpus, *out;
    size_t start = 0, end = 0, len = 0;
    uint32_t topk = 4;
    unsigned char *data;
    nanlm_teacher_rec *rec = NULL;
    size_t n = 0;
    int rc;
    if (argc < 4) return 2;
    corpus = argv[2];
    out = argv[3];
    if (argc >= 6) {
        start = (size_t)strtoull(argv[4], NULL, 0);
        end = (size_t)strtoull(argv[5], NULL, 0);
    }
    if (argc >= 7) topk = (uint32_t)strtoul(argv[6], NULL, 0);
    data = read_file(corpus, &len);
    if (!data) return 1;
    if (end == 0 || end > len) end = len;
    if (start >= end) { free(data); return 1; }
    if (nanlm_teacher_build_ngram(data, start, end, topk, &rec, &n) != 0) {
        free(data);
        return 1;
    }
    rc = nanlm_teacher_write(out, rec, n);
    printf("teacher: %s records=%zu topk=%u\n", out, n, topk);
    free(rec);
    free(data);
    return rc == 0 ? 0 : 1;
}

static int cmd_merge(int argc, char **argv) {
    if (argc < 4) return 2;
    return nanlm_teacher_merge(argv[2], (const char **)&argv[3], argc - 3) == 0
               ? 0 : 1;
}

static int cmd_scale(int argc, char **argv) {
    uint32_t num, den;
    if (argc != 6) return 2;
    num = (uint32_t)strtoul(argv[4], NULL, 0);
    den = (uint32_t)strtoul(argv[5], NULL, 0);
    return nanlm_teacher_scale(argv[2], argv[3], num, den) == 0 ? 0 : 1;
}

static int cmd_stats(int argc, char **argv) {
    uint64_t c = 0, r = 0, w = 0;
    if (argc != 3) return 2;
    if (nanlm_teacher_stats(argv[2], &c, &r, &w) != 0) return 1;
    printf("teacher: %s contexts=%llu records=%llu weight=%llu\n", argv[2],
           (unsigned long long)c, (unsigned long long)r,
           (unsigned long long)w);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s build-ngram CORPUS OUT [START END] [TOPK]\n"
                        "       %s merge OUT IN...\n"
                        "       %s scale OUT IN NUM DEN\n"
                        "       %s stats FILE\n",
                argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "build-ngram") == 0) return cmd_build(argc, argv);
    if (strcmp(argv[1], "merge") == 0) return cmd_merge(argc, argv);
    if (strcmp(argv[1], "scale") == 0) return cmd_scale(argc, argv);
    if (strcmp(argv[1], "stats") == 0) return cmd_stats(argc, argv);
    return 2;
}
