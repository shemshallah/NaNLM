# AMD Radeon Cloud / ROCm runbook: Qwen 7B teacher

Source: [AMD Radeon Cloud docs](https://amd-aim.github.io/radeon-cloud-docs/).
The quickstart uses **Deploy type: Notebook**, image **AMD OneClick Base**,
and optional **Persistent /workspace** storage.

The notebook version of this runbook is at
`notebooks/qwen7b_teacher.ipynb`.

## 0. Launch the instance

1. Sign in to the Radeon Cloud console.
2. Keep **Deploy type = Notebook**.
3. Choose **Image = AMD OneClick Base** (ROCm 7.2, Python 3.12, PyTorch 2.9.1,
   JupyterLab 4.5).
4. Turn on **Customize → Workspace storage = Persistent /workspace** so the
   model and shards survive instance destruction.
5. Launch, then **Open notebook** or connect over SSH.

Verify:

```sh
rocm-smi
/opt/venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

PyTorch calls the ROCm device `cuda`; `torch.version.hip` proves it is a ROCm
build.

> **Do not run `pip install torch`.** The AMD docs explicitly warn that this
> replaces the image’s ROCm build with a CUDA build and PyTorch then cannot see
> the GPU. Install only the extra libraries:
>
> ```sh
> /opt/venv/bin/pip install transformers accelerate sentencepiece huggingface_hub
> ```

If the GitHub clone is blocked, use the **AMD-Stable-GH-Proxy** image variant
or set your proxy before cloning.

## 1. Get the repo and model

```sh
cd /workspace
git clone https://github.com/shemshallah/NaNLM.git
cd NaNLM

/opt/venv/bin/huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /workspace/models/qwen2.5-7b-instruct
```

Qwen2.5-7B-Instruct is Apache-2.0. The download is about 15 GB in fp16. Put it
on persistent `/workspace`, not the ephemeral container disk.

If the GPU has less than ~20 GB, use 4-bit (`--load-in-4bit`) after installing
a ROCm-compatible `bitsandbytes`, or use a smaller teacher for the smoke test.
Do not start with a long corpus.

## 2. Smoke test on Alice

```sh
cd /workspace/NaNLM

/opt/venv/bin/python tools/hf_teacher.py \
  --model /workspace/models/qwen2.5-7b-instruct \
  --corpus data/alice_excerpt.txt \
  --out build/qwen7b_alice.ntr \
  --card build/qwen7b_alice.card.json \
  --max-tokens 2048 \
  --top-k 16 \
  --chunk 128 \
  --device cuda \
  --dtype float16
```

If fp16 OOMs:

```sh
/opt/venv/bin/pip install bitsandbytes
/opt/venv/bin/python tools/hf_teacher.py ... --load-in-4bit --chunk 64
```

Read the card before doing anything else:

```sh
cat build/qwen7b_alice.card.json
```

It contains:

- teacher top-1 and BPC on held-out token boundaries;
- a 3-gram BPC on the same byte positions;
- model, revision, token count, top-k, and context size.

Do not distill if the teacher does not beat the 3-gram baseline.

## 3. Compile students

```sh
make build/nanlm_teacher build/nanlm_slot8_train \
     build/nanlm_teacher_slot build/nanlm_asm build/nanlm_vm

./build/nanlm_teacher stats build/qwen7b_alice.ntr

TEACHER=build/qwen7b_alice.ntr \
CORPUS=data/alice_excerpt.txt \
MAX_KEYS=2048 \
sh tools/build_slot8.sh
```

For full-stream logs:

```sh
./build/nanlm_slot8_train \
  --teacher build/qwen7b_alice.ntr \
  --corpus data/alice_excerpt.txt \
  --max-keys 2048 \
  --eval-log build/qwen7b_slot8.tsv \
  --out build/qwen7b_slot8.nas \
  --expected build/qwen7b_slot8.expected \
  --manifest build/qwen7b_slot8.manifest

./build/nanlm_teacher_slot \
  --teacher build/qwen7b_alice.ntr \
  --corpus data/alice_excerpt.txt \
  --eval-log build/qwen7b_slot3.tsv \
  --out build/qwen7b_slot3.nas \
  --expected build/qwen7b_slot3.expected \
  --manifest build/qwen7b_slot3.manifest
```

Score:

```sh
python3 experiments/eval_stream.py \
  --run slot8=build/qwen7b_slot8.tsv=build/qwen7b_slot8.nas \
  --run slot3=build/qwen7b_slot3.tsv=build/qwen7b_slot3.nas \
  --boot 5000
```

## 4. Send back the small artifacts

```sh
sha256sum build/qwen7b_alice.ntr build/qwen7b_alice.card.json \
  build/qwen7b_slot8.nas build/qwen7b_slot3.nas \
  build/qwen7b_slot8.manifest build/qwen7b_slot3.manifest

tar czf /workspace/qwen7b_run.tgz \
  build/qwen7b_alice.ntr build/qwen7b_alice.card.json \
  build/qwen7b_slot8.tsv build/qwen7b_slot3.tsv \
  build/qwen7b_slot8.nas build/qwen7b_slot8.expected \
  build/qwen7b_slot8.manifest build/qwen7b_slot3.nas \
  build/qwen7b_slot3.expected build/qwen7b_slot3.manifest
```

Download `qwen7b_run.tgz` from JupyterLab or `scp` it. Do not upload the 7B
model itself; the repo needs the teacher shard and the card.

## 5. If Alice passes, run a larger stream

Use a PG-19 book or a pinned code corpus, not all of GitHub:

```sh
# Example code corpus from the continuous pipeline:
make continuous-check
/opt/venv/bin/python tools/hf_teacher.py \
  --model /workspace/models/qwen2.5-7b-instruct \
  --corpus build/continuous/corpus.txt \
  --out build/qwen7b_code.ntr \
  --card build/qwen7b_code.card.json \
  --max-tokens 8192 \
  --top-k 16 \
  --chunk 128 \
  --device cuda
```

## 6. Token Factory note

AMD’s Token Factory is an OpenAI-compatible **hosted chat** endpoint. It is
useful for data generation and sanity checks, but it does not expose the
next-token logits needed for NTR teacher shards. Use a GPU instance with the
Hugging Face teacher for distillation.

## 7. Free-hour rules

- One model load, many tokens; batch chunks.
- Cache the model in `/workspace`.
- Run Alice first; only then spend hours on a bigger corpus.
- Prefer 4k–16k tokens of a curated, license-checked stream over 1M random
  tokens.
- Do not fine-tune 7B on free hours. Teacher inference + CPU counting is the
  path.
- If ROCm/4-bit is unstable, use Modal A10G (`modal_teacher.py`) instead.
