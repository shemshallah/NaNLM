# In-VM plasticity

The first plasticity test now runs entirely on `nanlm_vm`:

```sh
make plasticity-check
```

`tests/plasticity.nas` demonstrates two things without leaving the VM:

1. **Bit-sliced 32-lane counters** stored in three NaN16 data words. Each
   observation is added with `BOOL.8` (AND) and `BOOL.6` (XOR) half-adders.
   A threshold `proto = c2 | (c1 & c0)` recovers the clean prototype
   `0xCAFEBABE` from five noisy observations.
2. **STW truth-table search**: the program rewrites the live `BOOL.n` word for
   candidate `n = 0..15` until the resulting circuit output matches the target.

Result:

```
plasticity: PASS (bit-sliced counter + STW regime evolution in 325 steps)
plasticity: NEGATIVE PASS (broken bit-sliced carry rejected)
plasticity: NEGATIVE PASS (frozen STW regime rejected)
```

This is a narrow but important claim:

> The VM can update its own Boolean substrate and select a regime without a
> round-trip through C.

It is not yet a learning rule for the full model. The next step is to use the
same bit-sliced counter/threshold pattern for token-level prototypes, and to
make regime selection depend on token class or syntactic state.

## Why this matters

The current C adapter still does most training. A model that only *runs* on
NaN16 is not the long-term bet. This test proves the VM can:

- count;
- threshold;
- rewrite a live `BOOL.n` word via `STW`;
- execute the rewritten circuit;
- reject a broken carry rule;
- reject frozen self-modification.

The negative controls are the important part: a plasticity gate that passes
when counters or STW are broken would be worthless.
