# Contributing

NaNLM is small on purpose. The useful contributions are models, adapters,
benchmarks, and documentation that make the word model easier to inspect.

## Ground rules

1. **Use NaN16 exactly.** Do not add a second word format. The word is two
   legal NaN heads around a 32-bit payload, with the class nibble `n` as the
   truth table or predicate.
2. **Data is neutral.** Model data words use `n=10`. `BOOL` instructions carry
   the function. Control instructions use `n=15` unless predicated.
3. **The model must run as `.nan`.** A C adapter may train, pack, or verify.
   It may not be the only runtime.
4. **No floats in the model.** Integer and Boolean operations only. Host
   diagnostics may use floats, but they must not change the artifact.
5. **Every claim gets a gate.** Add a test that fails when the implementation
   is wrong. A check that cannot fail is not a check.
6. **No transformer-parity claims.** The project is a substrate experiment.
   Report baselines, including the ones that beat us.

## Good first contributions

- A tiny model in `models/` with a manifest and a reproduction command.
- A benchmark against a simple baseline (n-gram, nearest centroid, etc.).
- A `BOOL.n` circuit with a truth-table explanation.
- A portability fix for the C adapter.
- A new document that explains a failure honestly.

## Running the tests

```sh
make test
```

The tests run on the host C compiler. The `.nan` runtime is tested separately
with `nan_cli` from NaN/0S.

## RFCs

Changes to the word model, the model format, or the learning rule need an RFC
in `docs/rfc/`. Keep it short: problem, proposal, failure modes, gate.

## Licensing

By contributing, you agree that your contribution is licensed under
Apache-2.0.
