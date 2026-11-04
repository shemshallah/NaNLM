#!/usr/bin/env python3
"""Sparse high-dimensional HDC sweep: accuracy, active fraction, model bits.

This is a substrate experiment, not a language model. It asks whether sparse
random prototypes remain separable at 2k/8k/32k bits, and how much of the
space is actually active. It is the control for the "higher dimension, extreme
sparsity" direction.
"""
from __future__ import annotations

import argparse
import numpy as np


def bits_to_matrix(bits: np.ndarray) -> np.ndarray:
    # 0/1 byte matrix; popcount via numeric sum of XOR.
    return bits.astype(np.uint8)


def hamming_matrix(q: np.ndarray, P: np.ndarray) -> np.ndarray:
    # q [D], P [N,D] -> [N] Hamming distances.
    return np.count_nonzero(P != q, axis=1)


def run(dim: int, density: float, classes: int, samples: int, noise: float,
        seed: int) -> dict:
    rng = np.random.default_rng(seed)
    # class prototypes with requested active fraction
    proto = (rng.random((classes, dim)) < density).astype(np.uint8)
    # samples are noisy copies: flip a fraction of all bits
    vals = []
    labels = []
    for c in range(classes):
        for _ in range(samples):
            x = proto[c].copy()
            flip = rng.random(dim) < noise
            x[flip] ^= 1
            vals.append(x)
            labels.append(c)
    X = np.stack(vals)
    y = np.array(labels)
    # train prototypes are class centroids from first half, test second half
    tr = rng.permutation(len(X))
    half = len(X) // 2
    train, test = tr[:half], tr[half:]
    centroids = np.zeros((classes, dim), dtype=np.uint8)
    for c in range(classes):
        m = y[train] == c
        if m.any():
            centroids[c] = (X[train][m].mean(axis=0) >= 0.5).astype(np.uint8)
    correct = 0
    active = 0
    for i in test:
        d = hamming_matrix(X[i], centroids)
        if int(np.argmin(d)) == int(y[i]):
            correct += 1
        active += int(X[i].sum())
    acc = 100.0 * correct / len(test)
    active_frac = 100.0 * active / (len(test) * dim)
    # description length of the centroid matrix (raw bits)
    model_bits = classes * dim
    return {
        "dim": dim,
        "density": density,
        "noise": noise,
        "classes": classes,
        "test": len(test),
        "acc": acc,
        "active_pct": active_frac,
        "model_bits": model_bits,
        "bits_per_class": dim,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dims", default="2048,8192,32768")
    p.add_argument("--densities", default="0.5,0.1,0.02")
    p.add_argument("--classes", type=int, default=16)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--noise", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    dims = [int(x) for x in args.dims.split(",")]
    densities = [float(x) for x in args.densities.split(",")]
    print("dim density noise classes test acc active% model_bits")
    for dim in dims:
        for density in densities:
            r = run(dim, density, args.classes, args.samples, args.noise,
                    args.seed + dim + int(density * 1000))
            print(f"{r['dim']:5d} {r['density']:7.3f} {r['noise']:5.3f} "
                  f"{r['classes']:7d} {r['test']:4d} {r['acc']:5.1f} "
                  f"{r['active_pct']:6.2f} {r['model_bits']:10d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
