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

| task | fixed3 (3 gates) | fixed6 (6 gates) | routed3 (3 gates) | random router |
|---|---:|---:|---:|---:|
| parity4 | 100.00% ± 0.00 | 100.00% ± 0.00 | 100.00% ± 0.00 | 100.00% ± 0.00 |
| majority4 | 86.23% ± 1.03 | 100.00% ± 0.00 | **92.77% ± 1.28** | 88.96% ± 5.07 |
| mixed parity/majority | 84.28% ± 7.14 | 90.92% ± 3.51 | 87.99% ± 4.81 | 84.08% ± 4.99 |
| teacher piecewise | 98.34% ± 4.39 | 98.93% ± 2.84 | **100.00% ± 0.00** | 98.14% ± 4.91 |
| random labels (control) | 48.54% ± 3.63 | 48.44% ± 4.21 | 49.71% ± 4.72 | 48.93% ± 3.54 |

## What this says

- On **majority4**, learned routing beats the equal-inference fixed3 by
  **+6.5 points**.
- A **6-gate fixed tree** reaches 100% on majority, so the routed gain is an
  **efficiency win**, not a raw-capacity win: 3 gates versus 6.
- On **teacher piecewise**, routed3 reaches 100% while fixed3 is 98.3%.
- A **random router** gets less of the gain, so learned selection matters.
- On **random labels**, no method wins; the control behaves like chance.
- Parity is already solved by fixed3; routing adds nothing.

This is the first controlled model-level signal that **self-modifying/regime
selection can help**, not just execute.

## What it does not prove

- It is a synthetic 256-input Boolean task, not a language model.
- The routed model has more parameters (24 vs 12 LUT bits), although the same
  inference gate count; the fixed6 control shows a dense 6-gate tree can match
  some gains at twice the inference cost.
- The majority gain is high-variance across seeds.
- The mixed task is not significant at this sample size.
- The 3-gate tree is tiny; scaling is unknown.

## NaN16 export

A learned routed tree was compiled to `.nas` and run on `nanlm_vm`:

```sh
sh tools/build_lut.sh
```

Output:

```
lut_gate: PASS (256 routed LUT outputs, 256 lines)
```

The seed-2 tree is:

```
route 0: (n1,n2,n3) = (1,4,1)
route 1: (n1,n2,n3) = (4,6,13)
router: x0
```

The VM executes 6,278 steps for all 256 inputs, `rc=3`, no trap, and every
output matches the Python model. This proves the compiler/runtime bridge; it
does not prove language-model quality.

## Next tests

1. Parameter-matched control: a fixed model with twice the gates, at twice the
   inference cost.
2. 20+ seeds and 64/128/192 train sizes.
3. Replace the synthetic task with an 8×8 binary digit task.
4. Add a LUT feature layer in front of the slot memory.
5. Scale the routed tree from 3 gates to 32–128 gates and measure capacity.
