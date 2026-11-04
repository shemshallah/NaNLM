/* SPDX-License-Identifier: MIT */
#include <stddef.h>
#include <stdint.h>

typedef struct {
    const uint8_t *data;
    size_t len;
    size_t pos;
} cursor;

static int cursor_peek(const cursor *c, uint8_t *out) {
    if (c->pos >= c->len) return -1;
    *out = c->data[c->pos];
    return 0;
}

static int cursor_take(cursor *c, uint8_t *out) {
    if (cursor_peek(c, out) != 0) return -1;
    c->pos++;
    return 0;
}

int cursor_skip_ws(cursor *c) {
    uint8_t b;
    while (cursor_peek(c, &b) == 0) {
        if (b != ' ' && b != '\t' && b != '\n' && b != '\r') break;
        c->pos++;
    }
    return 0;
}

uint32_t cursor_hash(const cursor *c) {
    uint32_t h = 2166136261u;
    size_t i;
    for (i = 0; i < c->len; i++) {
        h ^= c->data[i];
        h *= 16777619u;
    }
    return h;
}
