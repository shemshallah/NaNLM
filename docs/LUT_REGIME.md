# The NaN LUT / Regime Network

This is the strongest "special" architecture candidate to come out of the
layer work.

## The thesis

A `BOOL.n` gate is not a weak ALU op. It is a **2-input lookup table** with a
4-bit truth table, replicated over 32 lanes. A layer of these gates is a
bit-sliced LUT network. A stack is a programmable Boolean circuit.

The weights are not matrices. They are:

- the 4-bit truth table `n` per gate;
- the wiring/mask that selects the two input bits;
- the router that selects `n` per region.

That is a genuinely different weight class from a transformer.

## The layer

One 32-bit word holds 32 gates. Each gate computes:

```
out_i = LUT[n](a_i, b_i)
```

The same `n` applies to all 32 lanes of that word. A layer can use several
words, each with its own `n`, to get per-word regimes.

A **Regime Network** adds:

```
router(x) -> n in 0..15
STW n into the next BOOL slot
```

So the LUT is selected at runtime, not fixed at compile time.

## Why layers can use NaN differently

| layer | interpretation | role |
|---|---|---|
| B | 32 Boolean lanes | LUT gates, masks, routing |
| N | 8×4-bit | quantized features |
| Y | 4×8-bit | byte text / ctype |
| H | 2×16-bit | counters, scales |
| S | 1×32-bit | accumulation, readout |
| V | D-bit hypervector | bind/bundle |
| M | Hamming memory | retrieval |
| R | writable `n` | regime selection |
| X | BSWAPM / rot2 | permutation, integrity |

A special model can mix them:

```
B (LUT features) -> V (bind) -> M (memory) -> S (readout)
```

with an R router that changes the B LUT per token.

## Training a LUT layer

Discrete `n` is trainable without backprop through the VM:

1. **Soft LUT relaxation.** Each gate has 16 logits. Forward:
   `sum_n softmax(logits)[n] * BOOL.n(a,b)`. Backprop through the soft mixture
   in a host framework.
2. **STE.** Forward uses argmax `n`; backward uses the soft gradient.
3. **Export.** Harden each gate to its argmax `n`, emit `BOOL.n`.
4. **Verify.** C adapter vs `.nan` on a held-out set; exact output/hash.

Alternative: evolutionary search over `n` and wiring. Slower, but no gradients
and fits the discrete substrate.

## The decisive experiments

### E1 — piecewise Boolean function

- A function that is parity in one region and majority in another.
- Fixed 2-gate stack vs routed 2-gate stack at equal gate count.
- Status: a first synthetic probe was inconclusive; the harness split leaked
  the selector. Redo with a random split.

### E2 — learned 2-input LUT layer

- Learn a random Boolean function of 8 bits with a 2-layer LUT network.
- Compare:
  - fixed `n` per layer;
  - per-region `n` with a router;
  - random wiring vs learned wiring.
- Metric: exact function accuracy, gate count.

### E3 — LUT features + slot memory

- Use a small LUT network to map a 3-byte context to 32 bits.
- Feed those bits into the slot memory key.
- Compare to raw slot memory (52.3% / 2.67 BPC).
- If LUT features improve top-1/BPC, the special architecture is real.

### E4 — teacher distillation into LUT features

- Teacher distributes next-byte probabilities.
- Train a LUT feature layer with STE to predict the teacher's top-k.
- Store the resulting features in slot memory.
- Compare to raw teacher distillation.

## What would make it special

Not "it runs." These:

1. A routed LUT stack beats a fixed LUT stack of equal size on a held-out task.
2. A LUT feature layer improves slot memory top-1 or BPC.
3. A learned router beats a hash router at the same number of gates.
4. The compiled `.nan` matches the soft-trained model exactly.
5. The model remains interpretable: every gate is a truth table, every route
   is a 4-bit decision.

If (1) and (2) fail, the layer idea is a taxonomy, not an architecture.

## Risks

- 2-input LUTs are weak; a 4-input LUT needs composition (3 gates) or a wider
  truth table that NaN16 does not have natively.
- Discrete training is unstable; STE can collapse to one `n`.
- Wiring search is a large space; random wiring may be enough for small tasks.
- Per-region routing can overfit; need held-out regime selection.
- The VM's self-modifying write must be bounded and verified.

## The honest current state

- Self-modifying all 16 regimes: **PASS** (`regime_all.nas`, 248 steps).
- One payload, four interpretations: **PASS** (`layers.nas`).
- Slot memory: **52.3% top-1 / 2.67 BPC** vs n-gram **55.5% / 3.13**.
- Boolean reservoir: **11.15% / 7.91 BPC** — failed.
- Regime-vs-fixed synthetic probe: inconclusive; harness bug, not a result.

The next code is E1/E2, not another model claim.

## The special sentence

> **A NaN word can be a 32-lane LUT whose truth table is a writable weight.
> A layer is a Boolean circuit. A regime is a runtime choice of circuit.**

That is the architecture worth chasing.
