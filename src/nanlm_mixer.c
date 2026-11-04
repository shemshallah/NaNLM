#include "nanlm_mixer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q 16
#define ONE (1 << Q)
#define STRETCH_ONE 256
#define SQUASH_MIN (-4096)
#define SQUASH_MAX 4096
#define RING_BITS 16u
#define MATCH_BITS 16u
#define CTX_N 12u
#define DOC_N 3u

static int16_t stretch_tab[65536];
static int32_t squash_tab[SQUASH_MAX - SQUASH_MIN + 1];
static int tables_ready = 0;

static int clampi(int x, int lo, int hi) {
    return x < lo ? lo : x > hi ? hi : x;
}

static uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static void make_tables(void) {
    int i;
    if (tables_ready) return;
    for (i = 0; i < 65536; i++) {
        double p = (i + 0.5) / 65536.0;
        double s;
        if (i == 0) p = 0.5 / 65536.0;
        if (i == 65535) p = 1.0 - 0.5 / 65536.0;
        s = log(p / (1.0 - p)) * STRETCH_ONE;
        stretch_tab[i] = (int16_t)clampi((int)lround(s), -4096, 4096);
    }
    for (i = SQUASH_MIN; i <= SQUASH_MAX; i++) {
        double p = 1.0 / (1.0 + exp(-(double)i / STRETCH_ONE));
        squash_tab[i - SQUASH_MIN] =
            clampi((int)lround(p * ONE), 1, 65535);
    }
    tables_ready = 1;
}

static int32_t squash(int x) {
    return squash_tab[clampi(x, SQUASH_MIN, SQUASH_MAX) - SQUASH_MIN];
}

static uint32_t table_index(uint64_t h, uint32_t mask) {
    return (uint32_t)(h & mask);
}

static uint32_t hist_byte(const nanlm_mixer *m, uint32_t back) {
    if (back == 0 || back > m->ring_mask || m->ring_pos < back) return 0;
    return m->ring[(m->ring_pos - back) & m->ring_mask];
}

static uint64_t pack_back(const nanlm_mixer *m, uint32_t n) {
    uint64_t v = 0;
    uint32_t i;
    if (n > 8) n = 8;
    for (i = n; i > 0; i--) v = (v << 8) | hist_byte(m, i);
    return v;
}

static uint32_t ctype(uint32_t b) {
    if (b == 32 || b == 9 || b == 10 || b == 13) return 0;
    if ((b >= 65 && b <= 90) || (b >= 97 && b <= 122)) return 1;
    if (b >= 48 && b <= 57) return 2;
    return 3;
}

static uint64_t context_for(const nanlm_mixer *m, uint32_t t) {
    switch (t) {
    case 0: return 0;
    case 1: return hist_byte(m, 1);
    case 2: return pack_back(m, 2);
    case 3: return pack_back(m, 3);
    case 4: return pack_back(m, 4);
    case 5: return pack_back(m, 6);
    case 6: return pack_back(m, 8);
    case 7:
        return ((uint64_t)hist_byte(m, 2) << 16) |
               ((uint64_t)hist_byte(m, 4) << 8) | hist_byte(m, 6);
    case 8: {
        uint64_t v = 0;
        uint32_t i;
        for (i = 0; i < 5; i++) v = (v << 2) | ctype(hist_byte(m, i + 1));
        return v;
    }
    case 9: return m->word0_hash;
    case 10: return m->word1_hash;
    default:
        return ((uint64_t)(m->indent & 255) << 24) |
               ((uint64_t)(m->bracket_depth & 255) << 16) |
               ((uint64_t)hist_byte(m, 1) << 8) | m->line_start;
    }
}

static uint64_t cell_hash(uint32_t table, uint64_t ctx, uint32_t prefix,
                          uint32_t bit, uint32_t way, int doc) {
    uint64_t h = splitmix64(0x9E3779B97F4A7C15ull ^ (uint64_t)table);
    h = splitmix64(h ^ ctx);
    h = splitmix64(h ^ ((uint64_t)prefix << 8 | bit));
    h = splitmix64(h ^ (uint64_t)(way + 1));
    if (doc) h = splitmix64(h ^ 0xD0C0A11Bull);
    return h;
}

static nanlm_mix_cell *cell_at(nanlm_mix_cell *base, uint32_t buckets,
                               uint32_t table, uint32_t idx, uint32_t way) {
    return &base[((table * buckets + idx) * NANLM_MIX_WAYS) + way];
}

static int cell_predict(nanlm_mix_cell *base, uint32_t buckets,
                        uint32_t table, uint64_t ctx, uint32_t prefix,
                        uint32_t bit, int doc) {
    uint32_t way;
    int64_t n0 = 0, n1 = 0;
    for (way = 0; way < NANLM_MIX_WAYS; way++) {
        uint64_t h = cell_hash(table, ctx, prefix, bit, way, doc);
        uint32_t idx = table_index(h, buckets - 1);
        nanlm_mix_cell *c = cell_at(base, buckets, table, idx, way);
        int64_t a = c->n0 > 0 ? c->n0 : 0;
        int64_t b = c->n1 > 0 ? c->n1 : 0;
        if (way == 0) { n0 = a; n1 = b; }
        else {
            if (a < n0) n0 = a;
            if (b < n1) n1 = b;
        }
    }
    return (int)((((int64_t)n1 + 1) << Q) / (n0 + n1 + 2));
}

static void cell_update(nanlm_mix_cell *base, uint32_t buckets,
                        uint32_t table, uint64_t ctx, uint32_t prefix,
                        uint32_t bit, int doc, uint32_t actual_bit) {
    uint32_t way;
    for (way = 0; way < NANLM_MIX_WAYS; way++) {
        uint64_t h = cell_hash(table, ctx, prefix, bit, way, doc);
        uint32_t idx = table_index(h, buckets - 1);
        nanlm_mix_cell *c = cell_at(base, buckets, table, idx, way);
        c->check = (uint16_t)(h >> 48);
        if (actual_bit) {
            if (c->n1 < 0x3FFFFFFF) c->n1++;
        } else {
            if (c->n0 < 0x3FFFFFFF) c->n0++;
        }
    }
}


static uint32_t choose_regime(const nanlm_mixer *m, uint32_t bit_index) {
    uint64_t h = pack_back(m, 3) ^ ((uint64_t)m->indent << 12) ^
                 ((uint64_t)m->bracket_depth << 20) ^
                 ((uint64_t)m->line_start << 28) ^
                 (m->word0_hash >> 7);
    return (uint32_t)((splitmix64(h) & 63u) | (bit_index << 6));
}

static int match_predict(const nanlm_mixer *m, uint32_t prefix,
                         uint32_t bit_index) {
    uint64_t h;
    uint32_t pos, candidate;
    if (m->ring_pos < 4) return ONE / 2;
    h = splitmix64(pack_back(m, 4) ^ 0x4D41544348ull);
    pos = m->match_tab[h & m->match_mask];
    if (pos == 0 || pos >= m->ring_pos) return ONE / 2;
    if (m->ring_pos - pos > m->ring_mask) return ONE / 2;
    candidate = m->ring[pos & m->ring_mask];
    (void)prefix;
    (void)bit_index;
    /* Confidence grows with a simple match length probe. */
    {
        uint32_t back = 4, len = 0;
        while (len < 16 && pos > back && m->ring_pos > back &&
               hist_byte(m, back) ==
                   m->ring[(pos - back) & m->ring_mask]) {
            len++;
            back++;
        }
        {
            int p = 20 + (int)len * 4; /* 0.32 .. 0.96 */
            if (p > 60) p = 60;
            return ((candidate >> (7 - bit_index)) & 1u)
                       ? p * ONE / 64
                       : (64 - p) * ONE / 64;
        }
    }
}

uint16_t nanlm_mixer_predict_bit(const nanlm_mixer *m, uint32_t prefix,
                                 uint32_t bit_index,
                                 int16_t stretched_out[NANLM_MIX_INPUTS],
                                 uint32_t *regime_out) {
    int16_t st[NANLM_MIX_INPUTS];
    int64_t dot = 0;
    uint32_t i, regime;
    int p;
    make_tables();
    for (i = 0; i < CTX_N; i++) {
        int pi = cell_predict(m->cells, m->buckets, i,
                              context_for(m, i), prefix, bit_index, 0);
        st[i] = stretch_tab[pi];
    }
    for (i = 0; i < DOC_N; i++) {
        int pi;
        if (m->doc_cells) {
            uint64_t ctx = pack_back(m, i + 2);
            pi = cell_predict(m->doc_cells, m->doc_buckets, i, ctx,
                              prefix, bit_index, 1);
        } else {
            pi = ONE / 2;
        }
        st[CTX_N + i] = stretch_tab[pi];
    }
    {
        int pm = match_predict(m, prefix, bit_index);
        st[CTX_N + DOC_N] = stretch_tab[pm];
    }
    regime = choose_regime(m, bit_index);
    if (regime_out) *regime_out = regime;
    for (i = 0; i < NANLM_MIX_INPUTS; i++) {
        dot += (int64_t)m->weights[regime][i] * st[i];
        if (stretched_out) stretched_out[i] = st[i];
    }
    p = squash((int)(dot >> 8));
    return (uint16_t)p;
}

static void mix_update(nanlm_mixer *m, const int16_t st[NANLM_MIX_INPUTS],
                       uint32_t regime, int actual_bit, int mixed) {
    int64_t err = ((int64_t)actual_bit << Q) - mixed;
    uint32_t i;
    for (i = 0; i < NANLM_MIX_INPUTS; i++) {
        int32_t w = m->weights[regime][i];
        int32_t delta = (int32_t)((err * st[i]) >> 30);
        w = clampi(w + delta, -512, 512);
        m->weights[regime][i] = (int16_t)w;
    }
}

static void update_state(nanlm_mixer *m, uint8_t byte) {
    uint64_t h;
    /* match table maps the previous 4-byte context to this byte's position */
    if (m->ring_pos >= 4) {
        h = splitmix64(pack_back(m, 4) ^ 0x4D41544348ull);
        m->match_tab[h & m->match_mask] = m->ring_pos;
    }
    m->ring[m->ring_pos & m->ring_mask] = byte;
    m->ring_pos++;

    if ((byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
        (byte >= '0' && byte <= '9') || byte == '_' || byte >= 128) {
        m->word0_hash = splitmix64(m->word0_hash ^ byte);
    } else if (m->word0_hash) {
        m->word1_hash = m->word0_hash;
        m->word0_hash = 0;
    }

    if (byte == '\n') {
        m->indent = 0;
        m->line_start = 1;
    } else if (m->line_start && byte == ' ') {
        m->indent++;
    } else if (m->line_start && byte == '\t') {
        m->indent += 4;
    } else if (m->line_start && byte != ' ' && byte != '\t' && byte != '\r') {
        m->line_start = 0;
    }
    if (byte == '(' || byte == '[' || byte == '{') m->bracket_depth++;
    else if ((byte == ')' || byte == ']' || byte == '}') && m->bracket_depth)
        m->bracket_depth--;
}

void nanlm_mixer_observe_byte(nanlm_mixer *m, uint8_t byte, int update_global,
                              int update_doc) {
    uint32_t bi;
    uint32_t prefix = 0;
    for (bi = 0; bi < 8; bi++) {
        uint32_t bit = (byte >> (7 - bi)) & 1u;
        int16_t st[NANLM_MIX_INPUTS];
        uint32_t regime = 0;
        int mixed = nanlm_mixer_predict_bit(m, prefix, bi, st, &regime);
        uint32_t i;
        if ((update_global || update_doc) && m->adapt_mixer) {
            mix_update(m, st, regime, bit, mixed);
        }
        if (update_global) {
            for (i = 0; i < CTX_N; i++) {
                cell_update(m->cells, m->buckets, i, context_for(m, i),
                            prefix, bi, 0, bit);
            }
        }
        if (update_doc && m->doc_cells) {
            for (i = 0; i < DOC_N; i++) {
                cell_update(m->doc_cells, m->doc_buckets, i,
                            pack_back(m, i + 2), prefix, bi, 1, bit);
            }
        }
        prefix = (prefix << 1) | bit;
    }
    update_state(m, byte);
}

uint8_t nanlm_mixer_predict_byte(const nanlm_mixer *m, double probs[256]) {
    double logp[256];
    double best = -1e300;
    uint8_t best_byte = 0;
    uint32_t b;
    for (b = 0; b < 256; b++) logp[b] = 0.0;
    for (b = 0; b < 256; b++) {
        uint32_t prefix = 0, bi;
        for (bi = 0; bi < 8; bi++) {
            uint32_t bit = (b >> (7 - bi)) & 1u;
            int p = nanlm_mixer_predict_bit(m, prefix, bi, NULL, NULL);
            double pd = (double)p / ONE;
            pd = pd < 1e-9 ? 1e-9 : pd > 1.0 - 1e-9 ? 1.0 - 1e-9 : pd;
            logp[b] += log2(bit ? pd : 1.0 - pd);
            prefix = (prefix << 1) | bit;
        }
        if (probs) probs[b] = pow(2.0, logp[b]);
        if (logp[b] > best) { best = logp[b]; best_byte = (uint8_t)b; }
    }
    return best_byte;
}

nanlm_mixer *nanlm_mixer_new(uint32_t table_bits, uint32_t doc_bits) {
    nanlm_mixer *m = (nanlm_mixer *)calloc(1, sizeof(*m));
    uint32_t i, r;
    if (!m) return NULL;
    m->adapt_mixer = 0;
    if (table_bits < 10) table_bits = 10;
    if (table_bits > 22) table_bits = 22;
    if (doc_bits < 8) doc_bits = 8;
    if (doc_bits > 20) doc_bits = 20;
    m->table_bits = table_bits;
    m->buckets = 1u << table_bits;
    m->doc_bits = doc_bits;
    m->doc_buckets = 1u << doc_bits;
    m->cells = (nanlm_mix_cell *)calloc(
        (size_t)CTX_N * m->buckets * NANLM_MIX_WAYS, sizeof(*m->cells));
    m->doc_cells = (nanlm_mix_cell *)calloc(
        (size_t)DOC_N * m->doc_buckets * NANLM_MIX_WAYS, sizeof(*m->doc_cells));
    m->ring_mask = (1u << RING_BITS) - 1;
    m->ring = (uint8_t *)calloc((size_t)m->ring_mask + 1, 1);
    m->match_mask = (1u << MATCH_BITS) - 1;
    m->match_tab = (uint32_t *)calloc((size_t)m->match_mask + 1,
                                      sizeof(*m->match_tab));
    if (!m->cells || !m->doc_cells || !m->ring || !m->match_tab) {
        nanlm_mixer_free(m);
        return NULL;
    }
    for (r = 0; r < NANLM_MIX_REGIMES; r++)
        for (i = 0; i < NANLM_MIX_INPUTS; i++)
            m->weights[r][i] = 16;
    return m;
}

void nanlm_mixer_free(nanlm_mixer *m) {
    if (!m) return;
    free(m->cells);
    free(m->doc_cells);
    free(m->ring);
    free(m->match_tab);
    free(m);
}

void nanlm_mixer_reset_doc(nanlm_mixer *m) {
    if (!m || !m->doc_cells) return;
    memset(m->doc_cells, 0,
           (size_t)DOC_N * m->doc_buckets * NANLM_MIX_WAYS *
               sizeof(*m->doc_cells));
}

int nanlm_mixer_merge_counts(nanlm_mixer *dst, const nanlm_mixer *src,
                             int sign) {
    size_t i, n;
    if (!dst || !src || dst->buckets != src->buckets ||
        dst->doc_buckets != src->doc_buckets) return -1;
    n = (size_t)CTX_N * dst->buckets * NANLM_MIX_WAYS;
    for (i = 0; i < n; i++) {
        dst->cells[i].n0 += sign * src->cells[i].n0;
        dst->cells[i].n1 += sign * src->cells[i].n1;
        dst->cells[i].check = (uint16_t)(dst->cells[i].check ^
                                         src->cells[i].check);
    }
    n = (size_t)DOC_N * dst->doc_buckets * NANLM_MIX_WAYS;
    for (i = 0; i < n; i++) {
        dst->doc_cells[i].n0 += sign * src->doc_cells[i].n0;
        dst->doc_cells[i].n1 += sign * src->doc_cells[i].n1;
        dst->doc_cells[i].check = (uint16_t)(dst->doc_cells[i].check ^
                                             src->doc_cells[i].check);
    }
    return 0;
}

/* Simple interpolated 3/2/1-gram baseline for paired evaluation. */
static uint32_t *build_obs(const uint8_t *data, size_t train_len, int order,
                           size_t *out_n) {
    size_t n = (train_len > (size_t)order) ? train_len - (size_t)order : 0;
    uint32_t *obs;
    size_t t, k = 0;
    if (n == 0) { *out_n = 0; return NULL; }
    obs = (uint32_t *)malloc(n * sizeof(*obs));
    if (!obs) return NULL;
    for (t = (size_t)order - 1; t + 1 < train_len; t++) {
        uint32_t key = 0;
        int i;
        for (i = (int)order - 1; i >= 0; i--)
            key = (key << 8) | data[t - (size_t)i];
        obs[k++] = (key << 8) | data[t + 1];
    }
    *out_n = k;
    return obs;
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static void ngram_counts(const uint32_t *obs, size_t n, uint32_t key,
                         uint32_t counts[256], uint32_t *total) {
    size_t lo = 0, hi = n;
    uint32_t k = key << 8;
    memset(counts, 0, 256 * sizeof(*counts));
    *total = 0;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (obs[mid] < k) lo = mid + 1; else hi = mid;
    }
    while (lo < n && (obs[lo] >> 8) == key) {
        counts[obs[lo] & 0xFFu]++;
        (*total)++;
        lo++;
    }
}

static double ngram_bits(const uint32_t *o3, size_t n3,
                         const uint32_t *o2, size_t n2,
                         const uint32_t *o1, size_t n1,
                         const uint32_t uni[256], uint32_t utot,
                         const uint8_t ctx[3], uint8_t actual, int *correct) {
    uint32_t c3[256], c2[256], c1[256];
    uint32_t t3, t2, t1, b;
    uint32_t key3 = ((uint32_t)ctx[0] << 16) | ((uint32_t)ctx[1] << 8) | ctx[2];
    uint32_t key2 = ((uint32_t)ctx[1] << 8) | ctx[2];
    uint32_t key1 = ctx[2];
    double bestq = -1.0;
    int best = 0;
    ngram_counts(o3, n3, key3, c3, &t3);
    ngram_counts(o2, n2, key2, c2, &t2);
    ngram_counts(o1, n1, key1, c1, &t1);
    for (b = 0; b < 256; b++) {
        double a = 0.1;
        double p3 = ((double)c3[b] + a) / ((double)t3 + a * 256.0);
        double p2 = ((double)c2[b] + a) / ((double)t2 + a * 256.0);
        double p1 = ((double)c1[b] + a) / ((double)t1 + a * 256.0);
        double pu = ((double)uni[b] + a) / ((double)utot + a * 256.0);
        double q = 0.6 * p3 + 0.3 * p2 + 0.09 * p1 + 0.01 * pu;
        if (t3 == 0 && t2 == 0 && t1 == 0) q = pu;
        else if (t3 == 0) q = 0.75 * p2 + 0.2 * p1 + 0.05 * pu;
        if (q > bestq) { bestq = q; best = (int)b; }
    }
    {
        double a = 0.1;
        double p3 = ((double)c3[actual] + a) / ((double)t3 + a * 256.0);
        double p2 = ((double)c2[actual] + a) / ((double)t2 + a * 256.0);
        double p1 = ((double)c1[actual] + a) / ((double)t1 + a * 256.0);
        double pu = ((double)uni[actual] + a) / ((double)utot + a * 256.0);
        double pe;
        if (t3 == 0 && t2 == 0 && t1 == 0) pe = pu;
        else if (t3 == 0) pe = 0.75 * p2 + 0.2 * p1 + 0.05 * pu;
        else pe = 0.6 * p3 + 0.3 * p2 + 0.09 * p1 + 0.01 * pu;
        if (correct) *correct = (best == actual);
        return -log2(pe > 1e-12 ? pe : 1e-12);
    }
}

int nanlm_mixer_train_eval(const uint8_t *data, size_t len, double train_frac,
                           uint32_t table_bits, int use_doc_cache,
                           int online_adapt, const char *log_path,
                           nanlm_mixer_report *out) {
    nanlm_mixer *m;
    size_t train_len, pos;
    uint32_t *o3 = NULL, *o2 = NULL, *o1 = NULL;
    size_t n3 = 0, n2 = 0, n1 = 0;
    uint32_t uni[256] = {0};
    uint32_t utot = 0;
    uint64_t correct = 0, ngcorrect = 0, n = 0;
    double bits = 0.0, ngbits = 0.0;
    FILE *fp = NULL;
    int i;

    if (!data || len < 32 || train_frac <= 0 || train_frac >= 1) return -1;
    train_len = (size_t)((double)len * train_frac);
    if (train_len < 16) return -1;
    m = nanlm_mixer_new(table_bits, use_doc_cache ? 14 : 8);
    if (!m) return -1;
    /* Keep the mixer weights fixed. `online_adapt` only adapts the
     * document-local count table; the global tables stay frozen. */
    m->adapt_mixer = 0;
    if (!use_doc_cache) {
        free(m->doc_cells);
        m->doc_cells = NULL;
    }
    for (pos = 0; pos < train_len; pos++) {
        nanlm_mixer_observe_byte(m, data[pos], 1, use_doc_cache ? 1 : 0);
    }
    for (i = 0; i < 256; i++) {
        uni[i] = 0;
    }
    for (pos = 0; pos < train_len; pos++) uni[data[pos]]++;
    for (i = 0; i < 256; i++) utot += uni[i];
    o3 = build_obs(data, train_len, 3, &n3);
    qsort(o3, n3, sizeof(*o3), cmp_u32);
    o2 = build_obs(data, train_len, 2, &n2);
    qsort(o2, n2, sizeof(*o2), cmp_u32);
    o1 = build_obs(data, train_len, 1, &n1);
    qsort(o1, n1, sizeof(*o1), cmp_u32);

    if (log_path) {
        fp = fopen(log_path, "w");
        if (!fp) { free(o3); free(o2); free(o1); nanlm_mixer_free(m); return -1; }
        fprintf(fp, "pos actual pred slot_bits ngram_bits slot_correct ngram_correct\n");
    }
    for (pos = train_len + 7; pos + 1 < len; pos++) {
        uint8_t actual = data[pos + 1];
        uint8_t ctx3[3] = {data[pos - 2], data[pos - 1], data[pos]};
        uint32_t bi, prefix = 0;
        double nll = 0.0;
        int guess;
        int ngc = 0;
        double ngb;
        if (use_doc_cache && online_adapt) {
            /* document cache adapts on the held-out stream; global does not */
        }
        for (bi = 0; bi < 8; bi++) {
            uint32_t bit = (actual >> (7 - bi)) & 1u;
            int p = nanlm_mixer_predict_bit(m, prefix, bi, NULL, NULL);
            double pd = (double)p / ONE;
            pd = pd < 1e-9 ? 1e-9 : pd > 1.0 - 1e-9 ? 1.0 - 1e-9 : pd;
            nll += -log2(bit ? pd : 1.0 - pd);
            prefix = (prefix << 1) | bit;
        }
        guess = nanlm_mixer_predict_byte(m, NULL);
        if (guess == actual) correct++;
        ngb = ngram_bits(o3, n3, o2, n2, o1, n1, uni, utot, ctx3, actual, &ngc);
        ngbits += ngb;
        if (ngc) ngcorrect++;
        bits += nll;
        if (fp) {
            fprintf(fp, "%zu %u %u %.8f %.8f %d %d\n",
                    pos, actual, guess, nll, ngb,
                    guess == actual ? 1 : 0, ngc);
        }
        n++;
        nanlm_mixer_observe_byte(m, actual,
                                0,
                                use_doc_cache && online_adapt ? 1 : 0);
    }
    if (fp) fclose(fp);
    free(o3); free(o2); free(o1);
    if (out) {
        out->eval_bytes = n;
        out->bpc = n ? bits / (double)n : 0.0;
        out->top1 = n ? 100.0 * (double)correct / (double)n : 0.0;
        out->ngram_bpc = n ? ngbits / (double)n : 0.0;
        out->ngram_top1 = n ? 100.0 * (double)ngcorrect / (double)n : 0.0;
        out->model_bytes = sizeof(*m) +
            (size_t)CTX_N * m->buckets * NANLM_MIX_WAYS * sizeof(nanlm_mix_cell) +
            (m->doc_cells ? (size_t)DOC_N * m->doc_buckets * NANLM_MIX_WAYS *
                                sizeof(nanlm_mix_cell) : 0) +
            ((size_t)m->ring_mask + 1) +
            ((size_t)m->match_mask + 1) * sizeof(uint32_t);
        out->hutter_total_bpc = n
            ? (bits + 8.0 * (double)out->model_bytes) / (double)n : 0.0;
    }
    nanlm_mixer_free(m);
    return 0;
}
