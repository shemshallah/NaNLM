#include <stdio.h>
#include <stdlib.h>

#include "nanlm_asm.h"

int main(int argc, char **argv) {
    char err[256];
    if (argc != 3) {
        fprintf(stderr, "usage: %s SRC.nas DST.nan\n", argv[0]);
        return 2;
    }
    if (nanlm_assemble_file(argv[1], argv[2], err, sizeof(err)) != 0) {
        fprintf(stderr, "nanlm_asm: %s\n", err[0] ? err : "failed");
        return 1;
    }
    return 0;
}
