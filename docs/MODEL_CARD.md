# Model card: NaNLM byte-level base

## Identity

- **Name:** `nanlm_lm` (M2b structural base)
- **Type:** order-3 byte-level hyperdimensional associative memory
- **Runtime:** `nanlm_vm`, NaN16 `.nan` image
- **Trainer:** C adapter, integer bit-sliced counters
- **Version:** 1
- **License:** Apache-2.0

## Intended use

Research on sparse Boolean / hyperdimensional models and the NaN16 substrate.
It is a reproducible structural base, not a production language model.

Do not use it for:

- text generation quality;
- safety-critical decisions;
- claims of transformer parity;
- anything requiring calibrated probabilities.

## Data

- **Corpus:** first 30,000 bytes of Project Gutenberg #11, *Alice's Adventures
  in Wonderland*.
- **Provenance:** `data/PROVENANCE.md`.
- **Split:** 85% train (25,500 bytes), 15% validation (4,500 bytes), by byte
  offset.
- **Queries:** 32 held-out contexts, evenly spaced over the validation slice.
- **Cleaning:** none. The Gutenberg header is included.

## Architecture

- Word: NaN16, two legal NaN heads + 32-bit payload.
- D = 2,048 bits (64 words).
- Vocabulary: 256 bytes.
- Context: 3 bytes.
- Binding: `BOOL.6` (XOR) with position vectors.
- Bundling: majority-of-three.
- Memory: 8 prototypes per next byte (2,048 prototypes total), assigned by a
  query hash during training.
- Decoding: nearest Hamming prototype; byte = prototype index >> 3.
- Encoding hash: FNV-1a over the query words.

## Metrics

Measured on 32 held-out contexts:

| metric | value |
|---|---:|
| HDC top-1 | 34.4% |
| order-3 n-gram top-1 | 50.0% |
| order-3 n-gram bits/byte | 3.78 |
| HDC bits/byte | not reported (uncalibrated) |
| VM steps for 32 queries | 140,442,599 |
| VM trap | none |
| model image | ~4.4 MB `.nan` text, 148k words |

The n-gram wins. The HDC model is a real `.nan` program and matches the C
adapter on every predicted byte and query hash.

## Evaluation caveats

- Only 32 queries; one query is 3.1 percentage points.
- One corpus; no cross-corpus claim.
- The 8-prototype count was selected on the same small validation set.
- HDC probabilities are not calibrated; only the n-gram BPC is meaningful.
- The model image is not sealed yet; the query hash is a differential gate,
  not a weight seal.

## Runtime and reproduction

```sh
make test
make model-lm-check
```

The build:

1. trains in C;
2. emits `.nas`;
3. assembles with `nanlm_asm`;
4. runs on `nanlm_vm`;
5. requires `rc=3`, no trap, and an exact byte+hash diff;
6. runs negative gates for zeroed positions, swapped predicted prototypes,
   and step starvation.

## Negative gates

- Zeroing positions changes the query hash and fails.
- Swapping two predicted prototypes changes the class and fails.
- A 1,000-step starvation run truncates output and fails.

## Environmental cost

The model trains and runs on a phone-class CPU in seconds. No GPU is used.

## Open work

- D=8,192 and k-means memory.
- Calibrated decoding / bits-per-byte.
- Seal A/B/C and a `.nan` loader that rejects corruption.
- Top-k decoding.
- A 64-symbol tokenizer as a separate task.
- A browser demo.
