#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanlm_vm.h"

static void usage(const char *argv0) {
    fprintf(stderr, "usage: %s --img FILE.nan [--steps N] [--serial]\n", argv0);
}

int main(int argc, char **argv) {
    const char *img = NULL;
    uint64_t steps = 100000000ull;
    int serial = 0;
    nanlm_vm *v;
    int rc;
    int i;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--img") == 0 && i + 1 < argc) {
            img = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = strtoull(argv[++i], NULL, 0);
        } else if (strcmp(argv[i], "--serial") == 0) {
            serial = 1;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    (void)serial;
    if (!img) {
        usage(argv[0]);
        return 2;
    }

    v = nanlm_vm_new(0);
    if (!v) {
        fprintf(stderr, "nanlm_vm: allocation failed\n");
        return 1;
    }
    if (nanlm_vm_load_text(v, img) != 0) {
        fprintf(stderr, "nanlm_vm: load failed at line %d\n",
                nanlm_vm_load_error(v));
        nanlm_vm_free(v);
        return 1;
    }
    rc = nanlm_vm_run(v, steps);
    fprintf(stderr, "[nanlm-vm] nan-text: %llu steps, rc=%d, pc=%u, trap='%s'\n",
            (unsigned long long)nanlm_vm_steps(v), rc, nanlm_vm_pc(v),
            nanlm_vm_trap(v));
    nanlm_vm_free(v);
    return rc == NANLM_VM_TRAP ? 1 : 0;
}
