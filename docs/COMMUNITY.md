# Community plan

NaNLM needs a reason to exist that fits in one sentence and survives a
skeptical reader:

> **A language-model-shaped system where every weight is a gated NaN word and
> every operation is a 32-lane Boolean function.**

That is interesting to three groups:

1. **NaN/0S people** — a new use for the word model, with gates and seals.
2. **HDC / neurosymbolic people** — a concrete Boolean substrate for binding,
   bundling, and associative memory.
3. **systems / esolang people** — a model that is a `.nan` image, not a Python
   stack.

The project should not try to be a transformer competitor. It should be the
best-documented, most inspectable substrate for sparse high-dimensional
Boolean models.

## Lessons from deeprepo

deeprepo works as a public project because:

- the constraint is audacious and stated in one sentence;
- the architecture is concrete, not a vibe;
- the tests prove the audacious claim;
- performance is reported honestly, including the bad parts;
- there is a way for outsiders to interact (issues, chat, model loads);
- multiple repos exist only where the boundary is real.

NaNLM should copy that shape, not the scale.

- Audacious constraint: **no floating-point arithmetic; every weight is a
  NaN.**
- Concrete proof: a `.nan` model that runs on `nanlm_vm` with no C model code.
- Honest benchmark: bits/byte against an n-gram, win or lose.
- Interaction: a model zoo, a word inspector, a seal challenge, and a
  reproducible build.

## What to ship before the first public launch

1. **A 60-second demo.** Train a tiny model, emit `.nan`, run it, show the
   prediction and the Hamming distances. No slides.
2. **A word inspector.** Paste `7FF8_..._F87F`, see both heads, `n`, the truth
   table, the payload, and the `BSWAPM` mirror. This teaches the format in one
   screen.
3. **A model card.** Dimension, vocabulary, context, training data hash,
   baseline, accuracy/BPC, size, steps/token, and failure modes.
4. **One negative gate.** Show a corrupted model or swapped prototype failing
   the differential test. A check that cannot fail is not a check.
5. **A reproduction command.** One line, no hidden notebooks.

## Repo surface

- `README.md` — pitch, 30-second demo, honest limits, quick start.
- `docs/` — the word model, foundations, model cards, evidence.
- `models/` — tiny models with manifests and expected outputs.
- `.github/ISSUE_TEMPLATE/` — model submission, bug, benchmark, RFC.
- `DISCUSSIONS` — “show your model,” “negative results,” “hardware.”
- `docs/rfc/` — word/model/format changes.

Do not create a multi-repo ring until the single-repo model works. A model
that fits on a phone does not need five runners.

## Launch sequence

### Week 0 — private proof

- M1 pipeline passes on `nanlm_vm`.
- M2a binding/bundling runtime passes exact-vector and class gates.
- The word inspector works.
- One negative gate fails as expected.

### Week 1 — public artifact

- Publish the repo with the 60-second demo.
- Publish the word inspector as a static page (WASM or JS reimplementation of
  the word model; clearly label which parts are native).
- Publish one model card with an n-gram baseline.

### Week 2 — technical posts

- “The 32-lane BOOL is not a weak SIMD unit.”
- “Why every weight is a NaN, and why the gate matters.”
- “What we learned from a 1 MB Boolean associative memory.”
- “The negative results: where the n-gram wins.”

### Week 3 — community hooks

- **NaN golf:** smallest `.nan` model that solves a fixed task.
- **Regime zoo:** submit a short `BOOL.n` circuit and its truth table.
- **Seal challenge:** corrupt one byte; the loader must reject it.
- **Bring your own hypervector:** a manifest plus `.nan`; the harness runs it.
- **Double-endian mirror:** submit a circuit and its `BSWAPM` mirror.

### Week 4 — RFC and roadmap

- First RFC for the model format.
- Decide whether to split `nanlm-models`, `nanlm-web`, `nanlm-bench`.
- Publish the first external contribution.

## Launch posts

Tailor, do not spam:

- **Hacker News:** “A language model where every weight is a NaN” — lead with
  the demo and the honest limit, not AGI.
- **r/ProgrammingLanguages:** the word model and the 32-lane truth table.
- **r/MachineLearning:** the HDC/Boolean associative memory and the baseline.
- **HDC / Tsetlin communities:** binding, bundling, clause learning.
- **NaN/0S / esolang circles:** the `.nan` image and the seal.
- **FPGA / hardware:** the 32-lane LUT and the gate.

## Contribution ladder

1. **First issue:** run the demo, report the exact output and step count.
2. **Model:** add a tiny model with a manifest, expected output, and baseline.
3. **Benchmark:** add a baseline or a corpus split.
4. **Adapter:** improve the C emitter or the `.nan` loader.
5. **Runtime:** optimize the Hamming/popcount loop with a trace.
6. **RFC:** change the model format or the learning rule.

Every contribution needs a gate. A model without an expected output is not a
model; it is a screenshot.

## Governance

- Maintainer-led for the word model and the runtime.
- RFC for format changes; lazy consensus for models and benchmarks.
- Apache-2.0; no CLA.
- A short code of conduct; enforce it.
- Publish negative results and corrections.
- Do not let the C adapter become the runtime.

## Success metrics

Not stars. These:

- an external contributor runs the demo without asking a question;
- an external model is submitted with a manifest and a baseline;
- a negative gate fails as expected in public;
- the n-gram baseline is reproduced by someone else;
- a hardware person can explain the 32-lane LUT after reading one page;
- the model image is byte-identical across two builds.

## Anti-patterns

- “This beats transformers.” It does not.
- “AGI.” No.
- “NaN is faster.” It is not inherently faster.
- “C to NaN compiler” before the model works.
- “High-dimensional” without a dimension and a benchmark.
- A five-repo ring before one model runs.
- A demo that hides the step count or the baseline.
- A model that only runs through the C adapter.
