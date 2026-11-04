# Alien architecture: what else NaN gives

NaN is not just a container. It gives four computational primitives that a
normal 64-bit integer does not have:

1. **The class nibble is the instruction.** `n` is a 4-bit truth table for
   `BOOL`, and a 4-bit predicate for everything else.
2. **The class nibble is writable code.** `STW` can rewrite the `n` of an
   instruction word, so a program can change its own Boolean function at
   runtime.
3. **The word has two faces.** `BSWAPM` swaps them, and the mirror law
   `BOOL.rot2(n)(a,b) = BOOL.n(~a,b)` is exact.
4. **The gate is a type system.** A word with an illegal head traps when
   fetched, so the encoding itself is a capability check.

The rest of this document separates what is real, what is promising, and what
is a dead end.

## 1. Regime machine — tested, works

A program can:

1. compute a 4-bit function from data;
2. write it into the class nibble of a `BOOL` instruction with `STW`;
3. jump to that instruction and execute it.

`tests/regime.nas` does exactly this:

- slot starts as `BOOL.0` (constant zero);
- data word contains `6`;
- `STW` rewrites the slot to `BOOL.6` (XOR);
- `JMPR` executes it on `0xCCCCCCCC` and `0xAAAAAAAA`;
- result is `0x66666666`; the gate prints `PASS`.

This is run by `make regime-check` on `nanlm_vm` in 31 steps, no trap.

That is not a gimmick. It is a **self-modifying Boolean regime**. The same
instruction slot can be any of the 16 two-input functions at runtime.

### What it enables

- **Input-dependent Boolean functions.** A router computes `n` from a token,
  a hash, a confidence, or a small circuit, then rewrites the next gate.
- **Regime Mixture-of-Experts.** Sixteen experts = sixteen truth tables; the
  router selects one per token/layer. The expert identity and the function are
  the same 4-bit object.
- **Self-modifying circuits.** A model can specialize its own gates during
  inference; the program is part of the state.
- **Branchless specialization.** No `if`; write `n` and execute the same slot.

### What it does not give

- It does not give a fifth data bit. `n` is still 4 bits.
- It does not make training free. The router/program still has to be learned.
- It does not make `BOOL` a 4-input gate. It is still two-input, 32-lane.

### Next test

A **regime classifier**: a small circuit with two or three `BOOL` slots whose
`n` values are selected by a hash of the input; compare against a fixed-`n`
circuit of the same size. If the dynamic regime wins, the architecture is
worth pursuing.

## 2. Mirror machine — promising, not yet tested

`BSWAPM` reverses all eight bytes. It swaps the faces, so:

```
n -> rot2(n)
payload bytes -> reversed
```

The mirror law is exact:

```
BOOL.rot2(n)(a,b) == BOOL.n(~a,b)
```

Three possible uses:

1. **Free negation.** A circuit's mirror computes the same function with `~a`
   instead of `a`. You can implement a NOT on one operand without an extra
   instruction.
2. **Reversible / bidirectional circuits.** A mirror pair can run a forward
   and a backward pass over the same word. This is a representation-level
   capability, not free compute.
3. **Dual-face ensemble.** Store a circuit and its mirror; execute both and
   compare. Disagreement is a fault detector, not a better classifier.

The honest limit: a random mirror is not a second model. It is a second view
of the same payload.

### Next test

A **mirror ensemble** on the slot memory: run the same query against a
byte-swapped prototype table and combine the two votes. If it improves top-1,
the second face is information; if it does not, it is only integrity.

## 3. Capability words — real, and useful for volunteers

The both-face gate is a type system:

- a word with an illegal head traps when fetched;
- `LDN` returns the class nibble or `0xFFFFFFFF` for a bad head;
- `STW` writes a chosen class;
- sealed cartridges check both faces and payload hashes.

For an open, volunteer-trained model, this is a genuine differentiator:

- a shard can be **sealed**; a corrupted shard is rejected by hash and gate;
- a model page can be **tagged** by class (data, code, label, sentinel);
- a loader can refuse a word with the wrong class before it becomes state.

That is not intelligence, but it is **trust at word granularity**, which is
rare in open model formats.

### Next test

A model loader that verifies:

- every code word has a legal head;
- every data word is `n=10`;
- every label word is within range;
- a deliberately corrupted shard fails closed.

## 4. Signaling NaN / quiet NaN — mostly a dead end here

The four heads are `+s, -s, +q, -q`. In IEEE arithmetic, signaling NaNs trap
and quiet NaNs propagate. In NaN16, all four are legal instruction classes;
the model never uses floating point.

So sNaN is **not** a poison pill in the integer path. It is a different class
nibble, nothing more. Using it as a "taint" requires an extra convention, not a
hardware trap.

Use sNaN for:

- an explicit "needs review" tag in a shard;
- a museum/Orrery control-flow demo;
- a class-channel seal.

Do not claim it gives a hardware exception in the model.

## 5. Boolean reservoir — tested, failed

A random Boolean network with 512 nodes, 3 update steps per byte, and a linear
ridge readout:

```
test top-1: 11.15%
test BPC:   7.91
```

Worse than the n-gram. The reservoir is a real NaN-native architecture (each
node's rule can be a class nibble), but this configuration is not competitive.
It may improve with:

- leaky state (keep a fraction of the old state);
- input injection at more nodes;
- a trained readout with a hidden layer;
- a reservoir per byte class;
- evolutionary rule search.

But it is not the first win. Keep it as a research branch, not the base.

## 6. Per-slice regimes — promising, with a cost

Every 32-bit payload can carry its own class nibble, so a vector can have a
different Boolean function per slice. That is a form of **per-slice routing**.

The cost: the class nibble is also the predicate; a data word with a
non-neutral `n` is not neutral data. You must decide whether the slice is:

- data (`n=10`, neutral, no predicate);
- a function selector (`n` = truth table);
- a control word (`n=15` or a predicate).

A model that uses per-slice `n` as metadata must keep a parallel data plane or
accept that the slice is no longer neutral.

### Next test

A **regime Mixture-of-Experts**:

- 16 experts, each a short `BOOL` circuit with its own preferred `n`;
- a router computes a 4-bit expert id from a hash/popcount;
- the selected expert rewrites its `n` into a slot and runs.

Compare against a single circuit of the same total size.

## 7. Neutral words as sentinels — small but real

`n = 0, 5, 10, 15` are neutral: their two faces agree. That makes them useful
as:

- padding that survives `BSWAPM`;
- end-of-sequence markers;
- all-zero/all-one constants;
- canonical data words.

This is a format convenience, not a model breakthrough.

## 8. The alien architecture candidate

The most NaN-specific architecture is not a transformer with NaN weights. It is
a **Regime Machine**:

```
state: D-bit hypervector + a 4-bit regime
step:
  router(state, token) -> n in 0..15
  STW n into the next BOOL slot
  execute BOOL.n over selected lanes
  update state
```

The model's computation is a **program that rewrites its own Boolean
functions**. The 16 truth tables are the instruction set; the router is the
policy; the state is the hypervector.

This is genuinely different from an HDC memory because the *function* changes
at runtime. It is different from a normal MoE because the expert is not a
matrix; it is a truth table plus a lane mask.

### What must be proven

1. A dynamic regime beats a fixed regime of the same size on a small task.
2. The router can be learned cheaply (hash, popcount, small circuit, evolution).
3. The self-modifying write is safe: no execution of corrupted code, no gate
   trap, no runaway.
4. The model still runs on `nanlm_vm` and emits a deterministic hash.

Until (1) and (2) are measured, the Regime Machine is a research frontier, not
a claim.

## 9. What not to do

- Do not call `n` a fifth data bit. It is a truth table/predicate.
- Do not claim sNaN traps in the integer model path.
- Do not use `BSWAPM` as a second arithmetic unit. It is one payload.
- Do not add per-word metadata without a separate data plane.
- Do not put a reservoir in the base model until it beats the n-gram.
- Do not self-modify code without a writable-code boundary and a gate.

## 10. Test queue

1. **Regime vs fixed circuit** on a synthetic task. (highest value)
2. **Mirror ensemble** on the slot memory; does the second face add accuracy?
3. **Per-slice expert routing** with 16 truth tables.
4. **Leaky Boolean reservoir** with a hidden readout.
5. **Capability loader** that rejects a corrupted shard.
6. **Self-modifying loop** that cycles all 16 regimes and checks the lane
   identity at runtime.

The tested primitive is already in the repo: `make regime-check`.
