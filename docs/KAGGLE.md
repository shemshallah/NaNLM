# Kaggle runbook: Qwen 7B teacher

Kaggle gives free T4 GPU hours and is the easiest fallback when Modal/AMD is
not available. Use the script, not a multi-line shell cell.

## 1. Create the notebook

- New Notebook.
- **Settings → Accelerator → GPU T4 x2** (T4 is the safe choice).
- **Settings → Internet → On**.
- Phone-verify the account if GPU/internet is disabled.

Check:

```python
!nvidia-smi
```

## 2. Run the script

In one code cell:

```python
!git clone --depth 1 https://github.com/shemshallah/NaNLM.git
!python NaNLM/kaggle_qwen7b.py
```

That is the entire wrapper. The script:

- keeps Kaggle’s CUDA PyTorch; it only installs Transformers, Accelerate,
  bitsandbytes, SentencePiece, and Hugging Face Hub;
- clones/updates the repo;
- loads Qwen2.5-7B-Instruct in 4-bit on one GPU;
- writes `qwen7b_alice.ntr` and `qwen7b_alice.card.json`;
- compiles the 8-slot and 3-slot students;
- runs `nanlm_vm`;
- scores both students with full-stream bootstrap;
- packages `qwen7b_run.tgz` in `/kaggle/working`.

Download it from the notebook **Output** tab.

If you cannot clone from inside Kaggle, upload this repository as a Kaggle
Dataset, copy it to `/kaggle/working/NaNLM`, and run:

```python
!python /kaggle/working/NaNLM/kaggle_qwen7b.py
```

## 3. Check the teacher gate

The script prints the card and:

```
TEACHER GATE: PASS
```

or

```
TEACHER GATE: FAIL
```

Pass requires `teacher_bpc < ngram_bpc` and a non-empty eval set. Do not
promote a failing teacher.

## 4. Memory and time

- The script forces one GPU (`CUDA_VISIBLE_DEVICES=0`) because 4-bit 7B fits on
  a single T4 and multi-GPU `device_map=auto` is more fragile.
- Default: 2,048 tokens, 128-token chunks, top-k 16.
- If OOM:
  ```python
  !python NaNLM/kaggle_qwen7b.py --max-tokens 1024 --chunk 32
  ```
- If the C compile fails, get the teacher artifacts only:
  ```python
  !python NaNLM/kaggle_qwen7b.py --no-compile
  ```
- The 7B weights are about 15 GB; Kaggle `/kaggle/working` is about 20 GB. Do
  not download multiple teacher models in the same session.

## 5. What to do with the result

Bring back:

- `qwen7b_alice.card.json`
- `qwen7b_alice.ntr`
- `qwen7b_slot8.tsv` / `qwen7b_slot3.tsv`
- the two `.nas` files or their manifests and hashes.

Do not bring back the 7B model. The teacher shard is additive and small.

If the teacher passes but the student loses, the next move is a larger teacher
or a better corpus, not a larger student.

## 6. Kaggle-specific gotchas

- Internet must be enabled; `git clone` and Hugging Face downloads use it.
- T4 is recommended. P100 has older compute capability and 4-bit support is
  less reliable.
- Sessions are time-limited. Model download + teacher pass + compile is
  designed to fit one session.
- The script does not install `torch`; Kaggle’s CUDA build must stay intact.
- If `bitsandbytes` fails, run with `--no-4bit` only on a GPU with enough
  memory; otherwise use a smaller Apache teacher.
