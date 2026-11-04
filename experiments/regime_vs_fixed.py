#!/usr/bin/env python3
"""Regime vs fixed LUT-tree experiment.

Fixed model: one 3-gate 2-input LUT tree for all inputs.
Routed model: a single input bit selects one of two 3-gate trees.

Both execute 3 gates per example. The routed model has more parameters, but
the same inference cost; that is the sparse-regime claim.

All router and LUT choices are made on the training split only. Test is
untouched. The random-label task is a negative control.
"""
import numpy as np

GATE = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], dtype=np.uint8)
# Truth table n: bit k = f(a,b), k=2a+b. Evaluate arrays of 0/1.
def gate(n, a, b):
    a = a.astype(np.uint8); b = b.astype(np.uint8)
    out = np.zeros_like(a)
    if n & 1: out |= (~(a | b)) & 1
    if n & 2: out |= (~a & b) & 1
    if n & 4: out |= (a & ~b) & 1
    if n & 8: out |= (a & b) & 1
    return out & 1

def tree(n1, n2, n3, x):
    g1 = gate(n1, x & 1, (x >> 1) & 1)
    g2 = gate(n2, (x >> 2) & 1, (x >> 3) & 1)
    return gate(n3, g1, g2)

def fit_fixed(X, Y):
    best = (-1.0, (6, 6, 6))
    for n1 in range(16):
        g1 = gate(n1, X & 1, (X >> 1) & 1)
        for n2 in range(16):
            g2 = gate(n2, (X >> 2) & 1, (X >> 3) & 1)
            for n3 in range(16):
                acc = (gate(n3, g1, g2) == Y).mean()
                if acc > best[0]:
                    best = (acc, (n1, n2, n3))
    return best

def fit_routed(X, Y, route_bit):
    r = (X >> route_bit) & 1
    ns = []
    for route in (0, 1):
        m = r == route
        if m.sum() == 0:
            ns.append((6, 6, 6))
            continue
        acc, n = fit_fixed(X[m], Y[m])
        ns.append(n)
    return ns

def pred_routed(ns, X, route_bit):
    r = (X >> route_bit) & 1
    out = np.zeros(len(X), dtype=np.uint8)
    for route in (0, 1):
        m = r == route
        if m.sum():
            out[m] = tree(*ns[route], X[m])
    return out

def task_parity(X):
    return ((X >> 0) ^ (X >> 1) ^ (X >> 2) ^ (X >> 3)) & 1

def task_majority(X):
    b = [(X >> i) & 1 for i in (0, 1, 2, 3)]
    return ((b[0] + b[1] + b[2] + b[3]) >= 2).astype(np.uint8)

def task_mixed(X):
    # route by x0; route 0 parity of x1..x3, route 1 majority of x1..x3
    p = ((X >> 1) ^ (X >> 2) ^ (X >> 3)) & 1
    b = [(X >> i) & 1 for i in (1, 2, 3)]
    m = ((b[0] + b[1] + b[2]) >= 2).astype(np.uint8)
    return np.where(((X >> 0) & 1) == 0, p, m).astype(np.uint8)

def task_random(X, seed):
    rng = np.random.default_rng(seed)
    table = rng.integers(0, 2, 256).astype(np.uint8)
    return table[X]

def task_teacher_piecewise(X, seed):
    rng = np.random.default_rng(seed)
    ns = [(int(rng.integers(0, 16)), int(rng.integers(0, 16)), int(rng.integers(0, 16)))
          for _ in range(2)]
    r = (X >> 0) & 1
    return np.where(r == 0, tree(*ns[0], X), tree(*ns[1], X)).astype(np.uint8)

TASKS = [
    ("parity4", task_parity),
    ("majority4", task_majority),
    ("mixed(parity|majority by x0)", task_mixed),
    ("teacher piecewise (3-gate trees, x0)", lambda X, s: task_teacher_piecewise(X, s)),
    ("random labels (control)", task_random),
]

def run(seeds=8, n_train=128):
    X = np.arange(256, dtype=np.uint32)
    print(f"train={n_train} test={256-n_train} seeds={seeds} gates=3")
    for name, fn in TASKS:
        fixed_accs = []; routed_accs = []; random_router_accs = []; router_hits = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(256)
            tr, te = perm[:n_train], perm[n_train:]
            Y = fn(X, seed).astype(np.uint8) if name.startswith("teacher") or name.startswith("random") else fn(X).astype(np.uint8)
            # ensure both routes in train for the mixed tasks
            tr = np.concatenate([tr, np.where(((X >> 0) & 1) == 0)[0][:1], np.where(((X >> 0) & 1) == 1)[0][:1]])
            tr = np.unique(tr)
            if len(tr) > 240: tr = tr[:240]
            # routed: select router bit on a validation split of train
            val = tr[::5]; fit = np.setdiff1d(tr, val)
            # fixed uses exactly the same fit subset, never val/test
            _, nf = fit_fixed(X[fit], Y[fit])
            fixed_accs.append((tree(*nf, X[te]) == Y[te]).mean())
            best = (-1, 0, None)
            for rb in range(4):
                ns = fit_routed(X[fit], Y[fit], rb)
                acc = (pred_routed(ns, X[val], rb) == Y[val]).mean()
                if acc > best[0]: best = (acc, rb, ns)
            ns = fit_routed(X[fit], Y[fit], best[1])
            routed_accs.append((pred_routed(ns, X[te], best[1]) == Y[te]).mean())
            router_hits.append(best[1])
            # random-router control: same parameter count, router not learned
            rr = seed % 8
            nsr = fit_routed(X[fit], Y[fit], rr)
            random_router_accs.append((pred_routed(nsr, X[te], rr) == Y[te]).mean())
        p = len(TASKS)
        def fmt(a): return f"{100*np.mean(a):6.2f}%±{100*np.std(a):4.2f}"
        print(f"{name:38s} fixed={fmt(fixed_accs)} routed={fmt(routed_accs)} "
              f"random-router={fmt(random_router_accs)} routers={router_hits}")

if __name__ == "__main__":
    run()
