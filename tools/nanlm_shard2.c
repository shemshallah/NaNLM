#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanlm_shard2.h"

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
    unsigned char *data;
    nanlm_shard2_rec *rec = NULL;
    size_t n = 0;
    int rc;
    if (argc < 4) return 2;
    corpus = argv[2];
    out = argv[3];
    if (argc >= 6) {
        start = (size_t)strtoull(argv[4], NULL, 0);
        end = (size_t)strtoull(argv[5], NULL, 0);
    }
    data = read_file(corpus, &len);
    if (!data) return 1;
    if (end == 0 || end > len) end = len;
    if (start >= end) { free(data); return 1; }
    if (nanlm_shard2_build(data, start, end, &rec, &n) != 0) {
        free(data);
        return 1;
    }
    rc = nanlm_shard2_write(out, rec, n);
    printf("shard2: %s records=%zu bytes=%zu signed=1\n", out, n, end - start);
    free(rec);
    free(data);
    return rc == 0 ? 0 : 1;
}

static int cmd_merge(int argc, char **argv) {
    if (argc < 4) return 2;
    return nanlm_shard2_merge(argv[2], (const char **)&argv[3], argc - 3) == 0
               ? 0 : 1;
}

static int cmd_negate(int argc, char **argv) {
    if (argc != 4) return 2;
    return nanlm_shard2_negate(argv[2], argv[3]) == 0 ? 0 : 1;
}

static int cmd_stats(int argc, char **argv) {
    uint64_t keys = 0, records = 0;
    int64_t total = 0, lo = 0, hi = 0;
    if (argc != 3) return 2;
    if (nanlm_shard2_stats(argv[2], &keys, &total, &records, &lo, &hi) != 0)
        return 1;
    printf("shard2: %s keys=%llu records=%llu total=%lld min=%lld max=%lld\n",
           argv[2], (unsigned long long)keys, (unsigned long long)records,
           (long long)total, (long long)lo, (long long)hi);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s build CORPUS OUT [START END]\n"
                        "       %s merge OUT IN...\n"
                        "       %s negate OUT IN\n"
                        "       %s stats FILE\n",
                argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "build") == 0) return cmd_build(argc, argv);
    if (strcmp(argv[1], "merge") == 0) return cmd_merge(argc, argv);
    if (strcmp(argv[1], "negate") == 0) return cmd_negate(argc, argv);
    if (strcmp(argv[1], "stats") == 0) return cmd_stats(argc, argv);
    return 2;
}
