# M1 retrospective: what the first pipeline proved

M1 was deliberately a pipeline proof, not a model-quality claim. It did its
job: the C adapter, the emitter, the assembler, the NaN16 runtime, and the
differential harness now agree.

## What is solid

- `nan16_make` matches the real `nan_mkword` on 160 cases.
- `BOOL.n` matches the lane identity and mirror law for all 16 functions.
- The vector operations match scalar references.
- The 2,048-bit, 16-class toy model trains deterministically.
- The emitted `.nas` assembles to 11,264 words.
- The `.nan` image runs on the real `nanlm_vm` with no trap.
- The runtime output matches the expected labels, 32/32.
- The query labels are shuffled, so `query_index % classes` cannot pass.
- The negative gate swaps prototypes 0/1 and the differential test fails.
- The pipeline generalizes to other dimensions/classes (512-bit/8-class,
  512-bit/32-class) with the same code.

## What is not solid

- It is not a language model.
- The runtime does not yet bind or bundle. It uses `BOOL.6` inside Hamming.
- No seal or loader verification.
- No real corpus, no n-gram baseline.
- The runtime template is embedded in C; it is hard to review and edit.
- The `.nas`/`.nan` text is larger than the raw image (360 KB vs 90 KB).
- The model image includes 8,192 words of padding before data.
- The `.nan` format is convenient for review, not for large models.
- CI now runs the full VM differential (`make lane-check model-check model-ctx-check`).

## What was surprising

- The C core runs the 32-query model in ~1.1M steps and the whole build,
  assemble, run, and diff cycle takes under a second on the phone.
- The fixed `.org 8192` padding is most of the image. The actual model data is
  only 3,072 words (24 KB).
- The two-digit hex output was necessary as soon as the class count exceeded
  16; the first one-digit version would have silently truncated the future
  byte model.
- The weakest part of the first test was the sequential label order. It made a
  trivial `q % classes` runtime pass. Shuffling labels was a one-line idea with
  a large effect on test quality.

## What to change next

1. **Move the runtime template to a `.nas` file.** The C emitter can prepend
   `.equ` values and append data. The template becomes reviewable and testable
   on its own.
2. **Emit a raw `.nan`/binary image for large models.** Keep the `.nas` source
   for review; do not force `nanlm_vm` to parse 100k text lines.
3. **Add a query-vector hash gate.** Labels alone can be produced by a
   classifier that skips the encoder. A hash of the encoded vector catches
   that.
4. **Add the seal.** FNV A/B/C over near payloads, byte-swapped payloads, and
   class nibbles.
5. **Add the negative gates from M2.** Broken encoder, seal corruption, step
   starvation.
6. **Reduce the padding.** Put code and data immediately after each other, or
   use a real image section table. `.org 8192` was inherited from the museum;
   it is not required for a standalone model.

## The honest summary

M1 proves the boundary, not the intelligence. M2a has since made the NaN16
runtime actually bind and bundle, added an encoded-vector hash gate, and made
zeroed positions, swapped prototypes, and step starvation fail. The next
milestone is M2b: bytes, a held-out corpus, and an n-gram baseline.
