# Linking Kaggle datasets

Kaggle notebooks are ephemeral. A token shard must be stored as a Kaggle
Dataset (or downloaded/committed to a notebook version) to survive a reset.

## UI: create a dataset

1. Kaggle → **Datasets** → **New Dataset**.
2. Upload:
   - `qwen7b_tokens.ntt`
   - `qwen7b_tokens.card.json`
3. Title: `NaNLM Qwen7B Token Teacher`
4. Visibility: private unless you intentionally publish it.
5. Create.

The mount path becomes:

```text
/kaggle/input/<dataset-slug>/qwen7b_tokens.ntt
```

Example:

```text
/kaggle/input/nanlm-qwen7b-tokens/qwen7b_tokens.ntt
```

## UI: link it into a notebook

1. Open the notebook.
2. Right sidebar → **Add Data**.
3. **Your Datasets** → search `NaNLM Qwen7B Token Teacher`.
4. Add it.
5. The files appear under `/kaggle/input/...` immediately.

Find the exact path:

```python
!python /kaggle/working/NaNLM/tools/kaggle_find.py --pattern '*.ntt'
```

Use it:

```python
!python /kaggle/working/NaNLM/experiments/token_hdc_ntt.py \
  --ntt /kaggle/input/nanlm-qwen7b-tokens/qwen7b_tokens.ntt \
  --corpus /kaggle/working/NaNLM/data/alice_excerpt.txt \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dim 2048 --mode regime --top-counts 8
```

## CLI: create/update a dataset

Install/configure once:

```sh
pip install kaggle
mkdir -p ~/.kaggle
# Kaggle -> Account -> Create New API Token -> kaggle.json
mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Create a dataset directory:

```sh
mkdir -p /kaggle/working/nanlm-token-dataset
cp /kaggle/working/qwen7b_tokens.ntt \
   /kaggle/working/qwen7b_tokens.card.json \
   /kaggle/working/nanlm-token-dataset/
cat > /kaggle/working/nanlm-token-dataset/dataset-metadata.json <<'JSON'
{
  "title": "NaNLM Qwen7B Token Teacher",
  "id": "YOUR_USERNAME/nanlm-qwen7b-tokens",
  "licenses": [{"name": "CC0-1.0"}],
  "isPrivate": true
}
JSON
kaggle datasets create -p /kaggle/working/nanlm-token-dataset --dir-mode zip
```

Update it later:

```sh
cp /kaggle/working/qwen7b_tokens.ntt \
   /kaggle/working/nanlm-token-dataset/
kaggle datasets version -p /kaggle/working/nanlm-token-dataset \
  -m "refresh token teacher shard"
```

## Linking from a notebook output

If the run was **Save Version**-ed:

1. Open the saved version.
2. **Output** tab → download `qwen7b_tokens.tgz`.
3. Unpack locally.
4. Create a Kaggle Dataset with the `.ntt` and card.

Do not rely on `/kaggle/working` after the session ends.

## What not to link

- Do not upload the 7B model weights just to reuse the tokenizer; the tokenizer
  is tiny and downloadable.
- Do not commit the `.ntt` to git unless it is small and deliberately versioned.
- Do not put Kaggle credentials in the repository.
