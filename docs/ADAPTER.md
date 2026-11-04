# The C adapter

## 1. Why C, and why only C

The NaN16 machine is not a good target for arbitrary C. It has 16 32-bit
registers, a 32-bit payload, word-addressed memory, a four-bit predicate, and
no native floating point unit in the model plane. A full C front end would be a
multi-month project that produces a slow, partial C and teaches us little about
the model.

The useful C is the **adapter**:

- train and quantize the model;
- pack vectors into NaN16 words;
- emit `.nas` / `.nan`;
- load and verify a model image;
- act as the differential oracle for the NaN runtime.

This is S6: a host test bench, not the shipped machine.
The model must run without it.

## 2. The boundary

```
+-----------------------------+       +-----------------------------+
| C adapter (S6)              |       | NaN16 machine (S2..S5)      |
|                             |       |                             |
| train prototypes            | emit  | .nan image                  |
| pack neutral words          | ----> | codebook + code + manifest  |
| emit .nas / .nan            |       | BOOL / LD / ST / JMP        |
| verify seals                | <---- | prediction / trace          |
| compare reference           |       |                             |
+-----------------------------+       +-----------------------------+
```

Rules:

1. The adapter may use C, the host libc, and host floats for diagnostics, but
   the model data and the model program are integer/Boolean only.
2. The adapter must not be required at inference time.
3. The adapter must emit only NaN16 forms that `nanlm_asm` can assemble.
4. The adapter's `BOOL` must match the C core's `BOOL` bit for bit.
5. The adapter's word packing must match `nan_mkword` bit for bit.

## 3. Word helpers

`include/nan16_word.h` is a header-only implementation of the exact word model:

```c
typedef uint64_t nan16_word;

int      nan16_head_valid(uint32_t h);
uint32_t nan16_head_class(uint32_t h);
uint32_t nan16_near(nan16_word w);
uint32_t nan16_far(nan16_word w);
uint32_t nan16_n(nan16_word w);
uint32_t nan16_payload(nan16_word w);
int      nan16_gate(nan16_word w);
nan16_word nan16_make(uint32_t payload, unsigned n);
nan16_word nan16_bswap(nan16_word w);
uint32_t nan16_bool(unsigned n, uint32_t a, uint32_t b);
uint32_t nan16_popcount32(uint32_t x);
```

The canonical data word is:

```c
nan16_make(payload, 10)   /* both heads +q */
```

The canonical `BOOL` instruction uses the truth table in `n`, for example:

```
BOOL.6  rd, ra, rb     ; XOR
BOOL.8  rd, ra, rb     ; AND
BOOL.14 rd, ra, rb     ; OR
```

## 4. Vector helpers

`include/nanlm_vec.h` defines a vector as an array of neutral words:

```c
typedef struct {
    size_t words;
    nan16_word *w;
} nanlm_vec;
```

Operations:

```c
nanlm_vec *nanlm_vec_new(size_t words);
void nanlm_vec_free(nanlm_vec *v);
int  nanlm_vec_valid(const nanlm_vec *v);       /* every gate valid */
int  nanlm_vec_is_data(const nanlm_vec *v);     /* gate valid and n=10 */
void nanlm_vec_zero(nanlm_vec *v);
void nanlm_vec_fill(nanlm_vec *v, nan16_word w);
void nanlm_vec_random(nanlm_vec *v, uint64_t *state);

int  nanlm_vec_bool(nanlm_vec *dst,
                    const nanlm_vec *a,
                    const nanlm_vec *b,
                    unsigned n);
int  nanlm_vec_bind(nanlm_vec *dst,
                    const nanlm_vec *a,
                    const nanlm_vec *b);        /* XOR, n=6 */
int  nanlm_vec_bundle3(nanlm_vec *dst,
                       const nanlm_vec *a,
                       const nanlm_vec *b,
                       const nanlm_vec *c);     /* majority */
int  nanlm_vec_bswap(nanlm_vec *dst,
                     const nanlm_vec *a);        /* neutral data only */
uint64_t nanlm_vec_hamming(const nanlm_vec *a,
                           const nanlm_vec *b);
uint64_t nanlm_vec_popcount(const nanlm_vec *a);
int  nanlm_vec_equal(const nanlm_vec *a,
                     const nanlm_vec *b);
```

`nanlm_vec_bool` reads payloads and writes neutral words. It checks the gate
on every input word. A future fast path may skip checks after a loader has
validated the whole image; the checked path stays the reference.

## 5. Model adapters

There are two implemented model shapes. Both keep all model data neutral and
emit a `.nas` source plus an expected-output file.

### 5.1 M1 prototype model

```c
typedef struct {
    uint32_t dim;
    uint32_t words;
    uint32_t classes;
    uint32_t nqueries;
    nanlm_vec **prototypes;
    nanlm_vec **queries;
    uint8_t *expected;
} nanlm_model;
```

Functions:

```c
nanlm_model *nanlm_model_new(uint32_t dim, uint32_t classes,
                             uint32_t nqueries);
void nanlm_model_free(nanlm_model *m);
int  nanlm_model_train_toy(nanlm_model *m, uint64_t seed,
                           uint32_t samples_per_class, uint32_t noise_bits);
int  nanlm_model_predict(const nanlm_model *m, const nanlm_vec *query,
                         uint32_t *out_class);
int  nanlm_model_emit_nas(const nanlm_model *m, const char *path);
int  nanlm_model_write_expected(const nanlm_model *m, const char *path);
int  nanlm_model_write_manifest(const nanlm_model *m, const char *path,
                                uint64_t seed, uint32_t samples,
                                uint32_t noise);
```

This is the pipeline proof. It does not bind or bundle; it trains class
prototypes from noisy samples and classifies by Hamming distance.

### 5.2 M2a context model

```c
typedef struct {
    uint32_t dim;
    uint32_t words;
    uint32_t classes;
    uint32_t nqueries;
    nanlm_vec **codebook;  /* classes * 3 */
    nanlm_vec **positions; /* 3 */
    nanlm_vec **prototypes;
    nanlm_vec **queries;
    nanlm_vec *tmp[3];
    uint8_t *query_ids;    /* nqueries * 3 */
    uint8_t *expected;
    uint32_t *hashes;
} nanlm_ctx_model;
```

Functions:

```c
nanlm_ctx_model *nanlm_ctx_model_new(uint32_t dim, uint32_t classes,
                                     uint32_t nqueries);
void nanlm_ctx_model_free(nanlm_ctx_model *m);
int  nanlm_ctx_build(nanlm_ctx_model *m, uint64_t seed);
int  nanlm_ctx_encode(const nanlm_ctx_model *m, const uint8_t ids[3],
                      nanlm_vec *out);
int  nanlm_ctx_predict(const nanlm_ctx_model *m, const nanlm_vec *query,
                       uint32_t *out_class);
uint32_t nanlm_ctx_hash(const nanlm_vec *v);
int  nanlm_ctx_emit_nas(const nanlm_ctx_model *m, const char *path);
int  nanlm_ctx_write_expected(const nanlm_ctx_model *m, const char *path);
int  nanlm_ctx_write_manifest(const nanlm_ctx_model *m, const char *path,
                              uint64_t seed);
void nanlm_ctx_zero_positions(nanlm_ctx_model *m);
int  nanlm_ctx_swap_prototypes(nanlm_ctx_model *m, uint32_t a, uint32_t b);
```

The last two are negative-gate mutators. They are called after `build()` and
before `emit_nas()`.

### 5.3 Model I/O status

`emit_nas`, `write_expected`, and `write_manifest` are implemented. A general
`.nan` loader, a model verifier, and a seal writer are planned; M2a currently
relies on the emitted expected hash and the host-side C adapter as the oracle.
The C adapter trains and emits, but the release model still runs as a `.nan`
image on `nanlm_vm`.

The first implementation trains with integer bit-sliced counters. It does not
need floats. A diagnostic build may use doubles for reporting bits/byte, but
the model arrays remain integer/Boolean.

## 6. Emitting `.nan`

The `.nan` text format is the NaN16 format:

```
NNNN_PPPPPPPP_FFFF
```

where:

- `NNNN` is the near head value (`7FF1`, `FFF1`, `7FF8`, `FFF8`);
- `PPPPPPPP` is the payload as the big-endian hex of the little-endian u32;
- `FFFF` is the far head written byte-swapped (`F17F`, `F1FF`, `F87F`,
  `F8FF`).

For a neutral data word with payload `0x12345678`:

```
7FF8_12345678_F87F
```

The emitter must produce the same bytes as `nan_mkword(payload, 10)` in the C
core. The test suite compares them.

For code, the emitter uses the instruction payload layout:

```
[31:24] op
[23:20] d
[19:16] a
[15:0]  imm16
```

and sets the instruction's `n` to the desired truth table or predicate. For
example, `BOOL.6 r3,r1,r2` is encoded with `n=6` (near class 1, far class 2)
and payload `(0x00<<24)|(3<<20)|(1<<16)|(2)`.

The emitter should prefer `.word` data and labels over a raw binary blob so the
`.nas` file is reviewable. If the codebook is too large for a readable `.nas`,
emit a compact generator loop or a separate data file; do not hide the model in
an opaque binary.

## 7. Loading and verification

The loader:

1. Parses `.nan` text or raw words.
2. Checks every word's gate.
3. Checks every data word is `n=10`.
4. Reconstructs codebook, positions, and prototypes.
5. Recomputes seal A/B/C.
6. Rejects the model on any mismatch.

Seal A: FNV-1a/32 over near payloads.
Seal B: FNV-1a/32 over byte-swapped payloads.
Seal C: rotate/XOR over class nibbles.

This follows the dual-face seal idea. It is an integrity check, not a
signature. A future version can add a signature, but do not pretend the current
seal is authentication.

## 8. Differential testing

The adapter is useful only if it is a strict oracle.

For each test:

1. Generate a random model or a trained model.
2. Compute predictions with the C adapter.
3. Emit `.nan`.
4. Run the same inputs through `nanlm_vm`.
5. Compare outputs and, where possible, traces.

The trace comparison is stronger than the output comparison. A model can
produce the same answer through a different path; a trace mismatch means the
emitter or the runtime diverged.

## 9. Why not a C-to-NaN compiler

A C front end would need:

- a parser and type checker;
- a calling convention;
- a stack frame layout;
- pointer arithmetic and aliasing rules;
- 64-bit integer support on a 32-bit payload;
- floating-point emulation for programs that use it;
- a linker and relocation model.

That is a separate project. The model needs a **model compiler**, not a general
C compiler. If the community wants to write models in C, the header-only
adapter gives them that today: compile with `cc`, run on the host, emit `.nan`
for the machine. That is the 80/20.

## 10. The x86 question

"Back to x86" should mean:

- run the `.nan` image on the existing portable C core (`nanlm_vm`), or
- run it on a future native core.

It should **not** mean writing a new x86 backend. The project already has a
native x86 core and a portable C core. Adding a third backend is duplication,
not progress.

If speed becomes the bottleneck, the order is:

1. optimize the NaN16 code (unroll, BCOPY, fewer loads);
2. optimize the C core's dispatch (computed goto, direct threading);
3. use a future native core;
4. only then consider a specialized JIT for the model's inner loop.

The model's inner loop is small: XOR + popcount + compare. A specialized
compiler for that loop is a reasonable later optimization. A general x86
backend is not.
