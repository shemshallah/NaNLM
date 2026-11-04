# NaN16 foundation study (provenance)

This is the word-format study that had to happen before the design. It is
deliberately specific: the claims below are taken from the NaN/0S tree, which
is the provenance of the NaN16 format. **NaNLM does not depend on that tree.**
NaNLM has its own assembler (`nanlm_asm`), runtime (`nanlm_vm`), model format,
and gates.

Primary sources read:

- `~/nan0s/README.md`
- `~/nan0s/spec/NAN16_SPEC.md`
- `~/nan0s/spec/NANVM_SPEC.md`
- `~/nan0s/aarch64/android/core/OPS.md`
- `~/nan0s/spec/TRANSLATOR.md`
- `~/nan0s/spec/ARCH_ADAPTERS.md`
- `~/nan0s/spec/MEMORY_PLAN.md`
- `~/nan0s/spec/SELF_HOSTING.md`
- `~/nan0s/spec/NAN_MIGRATION_PLAN.md`
- `~/nan0s/aarch64/android/core/nan16.h`
- `~/nan0s/aarch64/android/core/nan16_int.h`
- `~/nan0s/aarch64/android/core/nan16.c`
- `~/nan0s/tools/nmk.awk`
- `~/nan0s/src/lib/n16de.nas`
- `~/nan0s/src/tools/n16i.nas`

## 1. The word is fixed

A NaN16 word is 64 bits:

```
 byte     b0 b1   |  b2 b3 b4 b5   |  b6 b7
          FAR head |  32-bit payload | NEAR head
          big-endian                 little-endian
```

- The **far head** is read big-endian from bytes 0-1.
- The **payload** is a 32-bit little-endian value in bytes 2-5.
- The **near head** is read little-endian from bytes 6-7.
- Legal heads are only `7FF1`, `FFF1`, `7FF8`, `FFF8`.
- A word with any other head fails the both-face gate and traps `!G`.
- `BSWAPM` reverses all eight bytes, swapping the faces and reversing the
  payload bytes.

The four heads are classes `0..3`:

| head | meaning | class |
|---|---|---|
| `7FF1` | positive signaling | 0 |
| `FFF1` | negative signaling | 1 |
| `7FF8` | positive quiet | 2 |
| `FFF8` | negative quiet | 3 |

The class nibble is:

```
n = (near_class << 2) | far_class
```

## 2. `n` is the truth table

For `BOOL`, bit `k` of `n` is `f(a,b)` with `k = 2a + b`:

```
k=0  a=0 b=0   ~(A|B)
k=1  a=0 b=1   ~A & B
k=2  a=1 b=0   A & ~B
k=3  a=1 b=1   A & B
```

This is the exact implementation in `nan16.c`:

```c
uint32_t res = 0;
if (n & 1) res |= ~(A | B);
if (n & 2) res |= ~A & B;
if (n & 4) res |= A & ~B;
if (n & 8) res |= A & B;
```

The lane identity is:

```
BOOL.n(0xCCCCCCCC, 0xAAAAAAAA) == n * 0x11111111
```

That is not a metaphor. The 32-lane output *is* the repeated nibble.

Useful functions:

| n | function |
|---:|---|
| 0 | constant 0 |
| 1 | NOR |
| 2 | `~A & B` |
| 3 | NOT A |
| 4 | `A & ~B` |
| 5 | NOT B |
| 6 | XOR |
| 7 | NAND |
| 8 | AND |
| 9 | XNOR |
| 10 | B |
| 11 | `~A | B` (A implies B) |
| 12 | A |
| 13 | `A | ~B` |
| 14 | OR |
| 15 | constant 1 |

XOR (`n=6`) is the important one for hyperdimensional binding because it is
its own inverse. AND/OR are useful for masks and majorities.

## 3. `n` is also the predicate

For every op except `BOOL`, the op runs only if bit `k = 2Z + L` of `n` is
set. The flag index is `2*Z + L`.

| n | predicate | runs when |
|---:|---|---|
| 15 | ALWAYS | always |
| 0 | NEVER | never |
| 4 | EQ | Z=1, L=0 |
| 3 | NE | Z=0, L=1 |
| 2 | LT | Z=0, L=0 (unsigned A<B) |
| 1 | GT | Z=0, L=1 |
| 5 | GE | Z=1, L=0 |
| 6 | LE | Z=1, L=0 |

`(Z,L) = (1,1)` is unreachable for normal integer compare results, but the C
core can produce it for unordered FCMP and divide-by-zero paths.

Consequence for NaNLM: **do not overload `n` as data metadata unless you also
want the predicate**. The clean split is:

- data words: `n = 10` (neutral, both +q), predicate-irrelevant;
- `BOOL` instructions: `n` is the truth table;
- control instructions: `n = 15` (ALWAYS) unless a predicate is intended.

## 4. The double-endian law

`BSWAPM` swaps the faces. For a word with class `n`:

```
rot2(n) = ((n & 3) << 2) | (n >> 2)
BOOL.rot2(n)(a,b) == BOOL.n(~a,b)
```

Neutral nibbles are `0, 5, 10, 15`; their two faces agree. `n=10` is the
canonical data word: both heads are `+q`, so a data word is not accidentally
a predicate or a special signaling value.

Consequences for NaNLM:

1. Store all model data as `n=10` words. This makes the data neutral and
   keeps `BOOL` operands free of head surprises.
2. `BSWAPM` on a neutral data word keeps `n=10` but reverses the 32 payload
   bytes. That is a free fixed permutation of the bit vector, useful for
   position binding and for a cheap second view of a vector.
3. A corrupted head traps at instruction fetch; a corrupted payload does not.
   Payload integrity needs the existing dual-face seal (FNV A/B/C), not the
   gate.
4. Do not `BSWAPM` code words unless you intend to mirror the instruction.

## 5. The machine

`NAN16_SPEC.md` and `OPS.md` define:

- registers `r0..r15`, 32-bit; `r0=0`, `r13=1`, `r14=SP`, `r15=link`;
- flags `Z,L`; predicate index `k=2Z+L`;
- 4M words (32 MB) in the current C core, 8192 words in the museum VM;
- word-addressed memory, wraparound by mask;
- payload layout for instructions:
  `[31:24]=op [23:20]=d [19:16]=a [15:0]=imm16`, with `b = imm[3:0]`;
- `BOOL`, `MOVI`, `MOVHI`, `ADD`, `SUB`, `SHL`, `SHR`, `CMP`, `LD`, `ST`,
  `JMP`, `CALL`, `RET`, `JMPR`, `NOP`;
- `VPOKE`, `VPEEK`, `KEY`, `TICK`, `BSWAPM`, `PUTC`, `STW`, `LDN`, display,
  keyboard, speaker;
- `MUL/DIV/MOD`, signed compare, `ADDI`, `LDB/STB`, `CALLR`, `BCOPY`, `BFILL`;
- `FADD..CVTFI` float ops (not used by NaNLM);
- `JMPF/CALLF`, framebuffer, port I/O, `CPUID`, `RDTSC`;
- `HALT`.

Relevant details for the model:

- `BOOL` ignores the predicate; it always runs and uses `n` as the truth table.
- `LD` reads only the payload; it does **not** check the gate.
- `LDN` returns the class nibble, or `0xFFFFFFFF` if a head is invalid.
- `ST` writes a `+q/+q` word (`n=10`).
- `STW` writes an arbitrary nibble from `r[b]&15`.
- `BCOPY` and `BFILL` move raw 8-byte words and preserve class.
- `BSWAPM` reverses the raw 8 bytes in memory.

## 6. The toolchain

The provenance tree has its own assembler, interpreter, C core, native cores,
image builder, filesystem, seals, and gates. NaNLM learns the format from
those, but owns its toolchain:

- `.nas` is the source of record.
- `.nan` is the assembled word text.
- `nanlm_asm` assembles the NaNLM subset.
- `nanlm_vm` executes the model image and traps on a bad gate.
- the C adapter trains, emits, and is the differential oracle.
- seals are the planned model-integrity format.

The C adapter is S6: a test bench, not the shipped runtime. The model runs on
`nanlm_vm`, with no external repository and no native kernel dependency.

## 7. Rules that carry over

- New model code is raw NaN16: `.nas` → `nanlm_asm` → `.nan`.
- C is the host test bench, not the shipped runtime.
- Every stored word is written in both faces.
- The gate checks both faces before execution.
- The seal checks near payload, far payload, and class channel.
- A check that cannot fail is not a check.
- The native boundary stays small.

NaNLM adds one more rule:

- **The model must run as a `.nan` image without the C adapter.** The C
  adapter may train, pack, emit, and verify. It may not be the only runtime.

## 8. What NaN/0S does not have yet

- A high-dimensional model layer.
- A model format (codebook + memory + program + manifest).
- A training/export adapter for such a model.
- A benchmark that compares a NaN-native model to a conventional baseline.
- A public "bring your own model" surface.

That is NaNLM's territory.

## 9. Design consequences

1. **Do not invent a new word layout.** Use NaN16 exactly.
2. **Do not store per-word n as model metadata.** Data is `n=10`; `BOOL`
   instructions carry the function; control instructions use `n=15`.
3. **The 32-bit payload is the vector slice.** A D-bit vector is `D/32`
   words. D=8192 is 256 words, 2 KB.
4. **Binding is `BOOL.6` (XOR).** Bundling is majority-of-three. Similarity
   is Hamming/popcount.
5. **Double-endian is integrity and permutation**, not a second arithmetic
   unit. Use neutral words and `BSWAPM` deliberately.
6. **C is the compiler/trainer, not the model.** The artifact is `.nan`.
7. **The first model is an HDC associative memory**, not a dense transformer.
8. **Everything gets a gate.** The lane identity, mirror law, word roundtrip,
   vector ops, model equivalence, and seal must all have tests that can fail.
