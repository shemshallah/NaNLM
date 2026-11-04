# `nanlm_vm` and `nanlm_asm`

NaNLM has its own assembler and runtime. They implement the NaN16 subset that
the model compiler emits; they are not the NaN/0S kernel.

## Assembler

`build/nanlm_asm SRC.nas DST.nan`

Supported source forms:

- `.equ NAME VALUE`
- `.org WORD_INDEX`
- `label:`
- `.word VALUE` (emitted as neutral `n=10` data)
- `LA rX,@label` (two words: `MOVI` low half, `MOVHI` high half)
- `BOOL.n`, `MOVI`, `MOVHI`, `ADD`, `SUB`, `SHL`, `SHR`, `CMP`, `LD`, `ST`,
  `JMP`, `CALL`, `RET`, `JMPR`, `NOP`, `BSWAPM`, `PUTC`, `MUL`, `BCOPY`,
  `BFILL`, `HALT`
- predicates `[ALWAYS] [NEVER] [EQ] [NE] [LT] [GE] [GT] [LE]`

It intentionally does **not** support the full NaN16 instruction set, macros,
or `.include`. The generated `.nas` files use only the forms above. A duplicate
label is an error (`tools/check_labels.sh` also checks before assembly).

Output is `.nan` text:

```
NNNN_PPPPPPPP_FFFF
```

- `NNNN` — near head
- `PPPPPPPP` — 32-bit payload
- `FFFF` — far head, byte-swapped

## Runtime

`build/nanlm_vm --img FILE.nan [--steps N]`

- 64-bit words, both-face gate; a bad head traps `!G`.
- 16 registers, `r0=0`, `r13=1`, `r14=SP` convention, `r15=link`.
- flags `Z,L`; predicate index `k=2Z+L`.
- 4M words of memory by default.
- `BOOL` is unpredicated and uses `n` as the truth table.
- `ST` writes `n=10` neutral data.
- `LD` reads the payload without a gate check.
- `HALT` returns `rc=3`.
- A trap returns `rc=1` and prints `!G`/`!O`.
- A budget exhaustion returns `rc=0`; **this is not a pass**.

The CLI writes one status line to stderr:

```
[nanlm-vm] nan-text: 612743 steps, rc=3, pc=39, trap=''
```

The model gates require `rc=3` and `trap=''`, then diff the serial output.
A run that prints the right prefix before exhausting its budget is rejected.

## Scope and limits

`nanlm_vm` implements the model runtime, not the museum OS. It does not provide
VGA, keyboard, ATA, NaNFS, timers, or the full NaN16 op set. The native
`kan16.xa` core is a separate performance path in the provenance tree; NaNLM
does not depend on it.

The interpreter is intentionally simple and auditable. A future native/JIT
backend must pass the same `.nan` differential gates.
