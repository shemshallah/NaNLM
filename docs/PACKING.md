# Packing, fast kernels, and the T1/T2/T3 split

The NaN16 word is a 64-bit object with 32 payload bits. That is a 2x storage
and bandwidth overhead on every weight. It is the right representation for
code, control, small tables, and self-modifying words. It is the wrong
representation for a multi-megabyte prototype body.

## Page format: NANPG1

Use 4 MiB pages. Each page has one 64-bit NaN header word and a payload body:

```
magic/version/type/n   (NaN header word, 64 bits)
payload_bits           32 bit-planes x 128 KiB = 4 MiB
Merkle root            32 bytes, in the page directory, not in the header word
```

The header word carries:

- type: `CODE`, `CONTROL`, `PROTO`, `CODEBOOK`, `LABELS`;
- `n`: number of logical words in the page;
- layout: `FULL_WORD`, `PAYLOAD_PLANES`, `SPARSE`;
- flags: endian, compressed, sealed.

The Merkle root is a 256-bit value, so it lives in a side directory:

```
page_id -> (body_hash, merkle_root, parent_generation, shard_set_hash)
```

A reader:

1. fetches the header word;
2. verifies the magic and page geometry;
3. fetches the body and Merkle proof;
4. unpacks the bit-planes into the runtime’s native counter/word form;
5. hands the page to the fast kernel.

Keep full per-word NaN for:

- program code and control flow;
- labels and small tables;
- self-modifying regime words;
- pages below one cache line or below the packing break-even.

Use bit-planes for:

- prototype matrices;
- codebooks;
- integer counter vectors;
- immutable generation bodies.

### Why bit-planes

A matrix of `n` 32-bit payloads stored as 32 bit-planes is the same number of
bits, but:

- no per-word 64-bit head;
- `POPCNT`/NEON/AVX/WASM SIMD can compare 32 lanes at once;
- pages compress well because planes are sparse/structured;
- range reads fetch only the planes needed for a candidate set;
- Merkle verification is over a 4 MiB page, not millions of small words.

The packing is not a model change. It is a storage and execution layout. The
VM remains the specification.

## Fast engine and conformance

`nanlm_vm` stays the reference implementation. It is a **conformance oracle**,
like a WASM spec interpreter, not the production evaluator.

Native paths:

- scalar POPCNT;
- ARM NEON `vcntq_u8` / `vaddvq_u32`;
- x86 AVX2 / AVX-512 popcount;
- WASM SIMD128;
- optional GPU kernels for teacher inference only.

Every generation runs:

1. the `.nan` image through `nanlm_vm`;
2. the same image through the native kernel;
3. compare the query hash, top-1 bytes, and per-query step/result hash;
4. reject the native path on any mismatch.

This is the same contract as a differential compiler test. The native path is
allowed to be faster; it is not allowed to be different.

Add explicit `POPCNT` and `HAMM` primitives to the keystone VM version. They
are the operations the slot memory and the fast engine actually need; keeping
them as generated assembly sequences is a performance tax and a conformance
ambiguity.

## Additive HDC prototypes

The current slot prototypes are bit-simplex keys. A more general HDC layer can
keep **integer counter vectors** per prototype:

```
prototype[p][d] += sign * index_vector(context)
```

At compile time:

```
bit[p][d] = counter[p][d] > 0
```

The runtime compares binarized prototypes with Hamming distance. The counter
form is mergeable by integer addition; the binarized form is the immutable
compiled generation. This preserves the CRDT property all the way through the
HDC layer.

## Random Indexing

Random Indexing is the additive semantic layer:

```
semantic[word] += random_index[context_word]
```

Accumulate over co-occurrence windows, then binarize at compile time. Search by
Hamming. It is not a transformer; it is a real generalization path that stays
additive and mergeable. Expect modest quality, and measure it against a
same-size n-gram baseline.

## T1 / T2 / T3

### T1 — compressor

Score BPC on enwik8, PG-19, and a code stream. The target is strong-compressor
territory, not LLM parity. Model size is part of the score:

```
total_bpc = (compressed_stream_bits + model_bits) / stream_bytes
```

The mixer is the T1 path. It must beat the 3-slot memory on a declared stream at
a declared model size.

### T2 — datastore LM

NaNLM is a community-updatable memory interpolated with a small Apache
transformer:

```
p(next | context) =
    lambda * p_transformer(next | context)
  + (1-lambda) * p_memory(next | context)
```

The transformer supplies general capability. NaNLM supplies freshness,
provenance, exact unlearning, and a per-document cache. This is the kNN-LM /
infini-gram line, but with mergeable counts and a repo-anchored generation
manifest. This is the most plausible “frontier-capable” path.

### T3 — distilled gate network

Train a differentiable logic-gate network (16 logits per gate, relaxed forward,
then harden to a truth table), ship it as `BOOL.n` words. This is a research
bet. Do not promise it. Gate it with the same full-stream evaluation and
conformance contract.

## Status

- Packed data-page writer/reader: implemented for neutral data words in
  `tools/nanlm_pack_pages.py`; `make pack-check` round-trips 10,000 payloads,
  verifies the page hash, and proves the packed body is smaller than full
  64-bit words. Code/control pages still use full NaN words.
- Native popcount/NEON kernel: implemented in `src/nanlm_fast.c`, tested
  against a scalar reference, and used by the C slot adapter. It is not yet a
  VM loader or a WASM SIMD path.
- VM conformance: the VM is still the only runtime; a fast engine that claims
  bit-exactness must hash-match the VM per generation.
- Additive HDC counters: design; slot memory uses bit prototypes.
- Random Indexing: design.
- T1 mixer: first C implementation, static mixer loses BPC.
- T2 interpolation: design.
- T3: not started.

The next implementation slice is a native slot evaluator that loads the packed
prototype pages and matches `nanlm_vm` query hashes byte-for-byte.
