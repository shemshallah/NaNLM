# Modal runbook: Qwen 7B teacher

This is the shortest path. Modal runs one GPU job; the student remains CPU
counting.

## 1. Install and authenticate

On your laptop, in the NaNLM repo:

```sh
python3 -m pip install --upgrade modal
modal setup
```

If `modal setup` is unavailable in your CLI version:

```sh
modal token new
# or, from the Modal dashboard:
modal token set --token-id ... --token-secret ...
```

Check:

```sh
modal profile current
```

## 2. One smoke run

From the repository root:

```sh
modal run modal_teacher.py \
  --corpus data/alice_excerpt.txt \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-tokens 2048 \
  --top-k 16
```

The app:

- builds a Python image with PyTorch, Transformers, Accelerate, bitsandbytes;
- loads Qwen2.5-7B-Instruct in 4-bit on an A10G;
- caches the model in the `nanlm-hf-cache` Volume;
- runs one batched pass over 2,048 tokens in 128-token chunks;
- writes an additive `qwen_teacher.ntr` and a teacher card to the
  `nanlm-teacher-shards` Volume.

The first run downloads the model; later runs reuse the Volume.

If A10G is unavailable or not allowed on your plan, choose an allowed GPU:

```sh
NANLM_GPU=L4 modal run modal_teacher.py \
  --corpus data/alice_excerpt.txt \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-tokens 2048 --top-k 16
```

## 3. Fetch the card and shard

```sh
mkdir -p build
modal volume get nanlm-teacher-shards qwen_teacher.ntr ./build/qwen_teacher.ntr
modal volume get nanlm-teacher-shards qwen_teacher.ntr.card.json \
  ./build/qwen_teacher.ntr.card.json
```

Read the card:

```sh
cat build/qwen_teacher.ntr.card.json
```

Gate:

- `teacher_top1` should beat the 3-gram top-1;
- `teacher_bpc` should beat `ngram_bpc`;
- if not, do not compile a student for promotion. Try a better teacher/corpus.

## 4. Compile the students locally

The teacher shard is small; the CPU work is local.

```sh
make build/nanlm_teacher build/nanlm_slot8_train \
     build/nanlm_teacher_slot build/nanlm_asm build/nanlm_vm

./build/nanlm_teacher stats build/qwen_teacher.ntr

TEACHER=build/qwen_teacher.ntr \
CORPUS=data/alice_excerpt.txt \
MAX_KEYS=2048 \
sh tools/build_slot8.sh
```

Full-stream logs:

```sh
./build/nanlm_slot8_train \
  --teacher build/qwen_teacher.ntr \
  --corpus data/alice_excerpt.txt \
  --max-keys 2048 \
  --eval-log build/qwen_slot8.tsv \
  --out build/qwen_slot8.nas \
  --expected build/qwen_slot8.expected \
  --manifest build/qwen_slot8.manifest

./build/nanlm_teacher_slot \
  --teacher build/qwen_teacher.ntr \
  --corpus data/alice_excerpt.txt \
  --eval-log build/qwen_slot3.tsv \
  --out build/qwen_slot3.nas \
  --expected build/qwen_slot3.expected \
  --manifest build/qwen_slot3.manifest

python3 experiments/eval_stream.py \
  --run slot8=build/qwen_slot8.tsv=build/qwen_slot8.nas \
  --run slot3=build/qwen_slot3.tsv=build/qwen_slot3.nas \
  --boot 5000
```

## 5. After Alice: one larger stream

```sh
modal run modal_teacher.py \
  --corpus build/continuous/corpus.txt \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-tokens 8192 \
  --top-k 16 \
  --out-name qwen_code.ntr
```

`build/continuous/corpus.txt` comes from `make continuous-check`, or generate
it from a pinned repository with `tools/nanlm_ingest.py`.

## Cost discipline

- One model load, many tokens.
- Keep `max_tokens` small until the teacher card passes.
- Do not fine-tune 7B on free hours.
- The model is cached in a Volume; do not delete it between runs.
- A10G/L4/T4 all run the 4-bit teacher. Check the Modal dashboard for current
  free-credit and GPU availability.
- The teacher shard and card are the only artifacts that need to leave Modal.

## If it fails

| symptom | fix |
|---|---|
| A10G unavailable | `NANLM_GPU=L4` or `NANLM_GPU=T4` |
| image build fails on bitsandbytes | use a CUDA 12.x image or remove 4-bit and use a larger GPU |
| model download slow | rerun; the HF Volume is cached |
| teacher loses to 3-gram | do not distill; try 7B on a better corpus or a larger teacher |
| Modal credits exhausted | fall back to AMD Radeon Cloud or Kaggle T4 |

The notebook and AMD fallback are in [AMD_GPU.md](AMD_GPU.md).
