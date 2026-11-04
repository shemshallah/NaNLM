# Regime vs fixed: first controlled result

## Protocol

- 8-bit inputs, all 256 values.
- 3-gate tree:
  - `g1 = gate(n1, x0, x1)`
  - `g2 = gate(n2, x2, x3)`
  - `out = gate(n3, g1, g2)`
- **Fixed**: one `(n1,n2,n3)` for all inputs.
- **Routed**: one input bit selects one of two `(n1,n2,n3)` trees.
- Both execute **3 gates per example**.
- Fixed has 12 LUT bits; routed has 24 LUT bits + 1 router bit, but the
  same inference gate count.
- Train: 128 inputs. Test: 128 inputs. 8 seeds.
- Router and LUTs are selected on a fit subset; the router is selected on a
  disjoint validation subset of train. Test is untouched.
- Controls: a random router (same parameters, not learned) and random labels.

Script: `experiments/regime_vs_fixed.py`.

## Results

| task | fixed | learned router | random router |
|---|---:|---:|---:|
| parity4 | 100.00% ± 0.00 | 100.00% ± 0.00 | 100.00% ± 0.00 |
| majority4 | 86.23% ± 1.03 | **92.77% ± 1.28** | 88.96% ± 5.07 |
| mixed parity/majority | 84.28% ± 7.14 | 87.99% ± 4.81 | 84.08% ± 4.99 |
| teacher piecewise 3-gate trees | 98.34% ± 4.39 | **100.00% ± 0.00** | 98.14% ± 4.91 |
| random labels (control) | 48.54% ± 3.63 | 49.71% ± 4.72 | 48.93% ± 3.54 |

## What this says

- On **majority4**, learned routing beats fixed by **+6.5 points** at the same
  inference gate count.
- On a **teacher piecewise** task, learned routing reaches **100%** vs 98.3%.
- A **random router** gets less of the gain (88.96 on majority, 98.14 on
  teacher), so the gain is not only extra parameters.
- On **random labels**, no method wins; the control behaves like chance.
- Parity is already solved by the fixed tree; routing adds nothing.

This is the first controlled model-level signal that **self-modifying/regime
selection can help**, not just execute.

## What it does not prove

- It is a synthetic 256-input Boolean task, not a language model.
- The routed model has more parameters (24 vs 12 LUT bits), although the same
  inference gate count.
- The majority gain is high-variance across seeds.
- The mixed task is not significant at this sample size.
- The 3-gate tree is tiny; scaling is unknown.

## Next tests

1. Parameter-matched control: a fixed model with twice the gates, at twice the
   inference cost.
2. 20+ seeds and 64/128/192 train sizes.
3. Compile a learned regime tree to `.nas` and run it on `nanlm_vm` for all
   256 inputs.
4. Replace the synthetic task with an 8×8 binary digit task.
5. Add a LUT feature layer in front of the slot memory.

The result is promising, not conclusive. The next gate is the NaN16 export.
