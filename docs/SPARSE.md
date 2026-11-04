# Sparse high-dimensional sweep

`experiments/sparse_dim.py` measures the basic tradeoff:

- dimension: 2,048 / 8,192 / 32,768 bits;
- prototype density: 0.5 / 0.1 / 0.02;
- 64 classes, 16 noisy samples per class, bit-flip noise;
- nearest-centroid Hamming classification;
- active fraction and raw model bits.

## Low-noise result (3% bit flips)

| dim | density | accuracy | active % | model bits |
|---:|---:|---:|---:|---:|
| 2,048 | 0.5 | 100% | 49.8 | 131,072 |
| 2,048 | 0.1 | 100% | 12.3 | 131,072 |
| 2,048 | 0.02 | 100% | 4.9 | 131,072 |
| 8,192 | 0.02 | 100% | 4.9 | 524,288 |
| 32,768 | 0.02 | 100% | 4.9 | 2,097,152 |

Extreme sparsity works when noise is low. Higher dimension does not improve
accuracy on this easy task; it only increases model bits.

## Higher-noise result (22% bit flips)

| dim | density | accuracy | active % |
|---:|---:|---:|---:|
| 2,048 | 0.5 | 100% | 49.9 |
| 2,048 | 0.1 | 95.3% | 27.5 |
| 2,048 | 0.02 | 39.8% | 23.2 |
| 8,192 | 0.1 | 97.7% | 27.6 |
| 8,192 | 0.02 | 39.6% | 23.1 |
| 32,768 | 0.02 | 44.5% | 23.1 |

At 22% flips, a 2% density prototype is fragile. Noise flips both active and
inactive bits; that raises the active fraction and destroys the sparse code.
This is a warning for the token HDC path: sparse codes need **noise-aware
binding and active masks**, not raw bit flips.

## What this means

- Sparse high dimension is viable at low noise.
- Accuracy saturates; adding bits alone is not a win.
- Active fraction and model bits must be reported with accuracy.
- The next sparse experiment should use **drop-only** noise on active bits and
  measure page faults / active lanes in the VM.

This is a control, not a language-model result.
