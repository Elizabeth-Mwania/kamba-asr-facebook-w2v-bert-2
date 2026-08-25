# Kamba ASR — Fine-tuning ASR models for Kamba (Kikamba)

> **Kamba (Kikamba)** is a Bantu language spoken by approximately 4 million people in Kenya.  
> This repository contains a Wav2Vec-BERT 2.0 baseline and a reusable Whisper Small workflow for Kamba, fine-tuned on 10 hours of speech from the [Kamba-ASR-Data-Subset-484H](https://huggingface.co/datasets/Digital-Divide-Data/Kamba-ASR-Data-Subset-484H) corpus.

---

## Results

| Split | WER ↓ | CER ↓ | Score ↑ |
|---|---|---|---|
| Validation | 28.66% | 8.27% | 81.54% |

**Model:** `facebook/w2v-bert-2.0` · **Training data:** 10 hours · **Epochs:** 10 · **GPU:** T4

---

## Dataset

| Split | Clips | Duration |
|---|---|---|
| Train | 3,673 | 10.00 hours |
| Validation | 500 | ~1.5 hours |
| Test | 500 | ~1.5 hours |

Splits are created by streaming the first 500 clips as test, the next 500 as validation, and the remainder (up to 10 hours) as training — ensuring no overlap. The dataset has only a `train` split on HuggingFace; splits are created locally.

## Reusable benchmark data

Materialize the selected subset once, then point every model at it. This downloads and stores only the selected raw audio and transcripts; it does **not** cache the complete 484-hour corpus or model-specific features.

```shell
python scripts/materialize_kamba_subset.py --output_dir data/kamba_10h_v1
```

The directory contains `dataset/` (self-contained 16 kHz WAV audio and metadata) and `benchmark_manifest.json` (the exact source row indices and selection policy). Keep it outside Git—`data/` is ignored—and back it up to Drive or durable local storage. It is the fixed benchmark for all experiments.

Do not persist Whisper log-Mel features with `Dataset.map`: 30-second Whisper features are nearly 1 MB per clip and create a large model-specific disk cache. The Whisper workflow extracts them only for the current batch.

The sequential split deliberately reproduces the existing Wav2Vec-BERT setup. The source has six speakers, so it is not speaker-disjoint; report that limitation and create a separate, explicitly versioned speaker-disjoint benchmark if the research question needs speaker generalisation. Never silently change the manifest between models.

### Whisper Small

```shell
python scripts/train_whisper.py \
  --data_dir data/kamba_10h_v1 \
  --output_dir outputs/kamba-whisper-small-10h
```

This saves `metrics.json`, `predictions_validation.json`, and `predictions_test.json`; the research metrics are WER and CER only. The experiment record is [config_files/kamba_whisper_small_10h.yaml](config_files/kamba_whisper_small_10h.yaml). Whisper has no Kamba language token, so the current experiment uses its Swahili (`sw`) prompt as an explicit, recorded proxy—not a claim that Kamba is Swahili.

### Wav2Vec-BERT 2.0 on the same benchmark

Use [config_files/kamba_asr_w2v_bert_10h_benchmark.yaml](config_files/kamba_asr_w2v_bert_10h_benchmark.yaml). Unlike the original streaming configuration, this reads the materialized dataset directly and does not resplit or reselect clips.

```shell
python scripts/train_model.py --config config_files/kamba_asr_w2v_bert_10h_benchmark.yaml
```

---

## Setup

### 1. Clone the repository

```shell
git clone https://github.com/Elizabeth-Mwania/kamba-asr-facebook-w2v-bert-2.git
cd kamba-asr-facebook-w2v-bert-2
```

### 2. Install dependencies

```shell
pip install transformers==4.44.2 datasets==2.20.0 accelerate==0.33.0 \
            jiwer evaluate wandb pyyaml
```

### 3. Configure `.env`

Create a `.env` file in the project root:

```env
# Weights & Biases — set WANDB_DISABLED=true to disable logging
WANDB_API_KEY=""
WANDB_DISABLED="true"

# Hugging Face — leave empty for public datasets
HF_API_KEY=""

# Cache directories
HF_HOME="./huggingface_cache"
NUMBA_CACHE_DIR="/tmp/numba_cache"
LIBROSA_CACHE_DIR="/tmp/librosa_cache"
MPLCONFIGDIR="/tmp/matplotlib_cache"
```

### 4. Configure the YAML

The experiment config is at `config_files/kamba_asr_w2v_bert_10h.yaml`.  
Key settings:

```yaml
# Project
project: "kamba-asr"
output_dir: "outputs/kamba-asr-facebook-w2v-bert-2"
seed: 42

# Model — w2v-bert-2.0 requires add_final_layer_adapter: true
pretrained_model: "facebook/w2v-bert-2.0"
freeze_feature_encoder: true
add_final_layer_adapter: true

# Training
batch_size: 8
gradient_accumulation_steps: 4   # effective batch = 32
num_epochs: 10
learning_rate: 0.00003
warmup_ratio: 0.1
fp16: true                        # requires GPU
gradient_checkpointing: true
save_steps: 500
eval_steps: 500
logging_steps: 10
save_total_limit: 2
report_to: "none"

# Dataset — DDD-Kenya corpus, streamed (no full download)
use_custom_dataset: false
dataset_path: "DDD-Kenya/Kamba-ASR-Data-Subset-484H"
train_split: "train"
create_splits_if_missing: true
validation_sample_size: 500       # fixed clip count for val
test_sample_size: 500             # fixed clip count for test
max_train_hours: 10.0             # cap training data to 10 hours

# Kamba character set
add_language_tokens: false
apply_accent_replacements: false
character_set: "abcdefghijklmnoprstuvwyzĩũ' 0123456789"
```

> **Important (Windows users):** The YAML must be saved as UTF-8 and loaded with `encoding='utf-8'`  
> to prevent `ĩ` and `ũ` from being silently corrupted. This is handled in `src/utils/config.py`.

### 5. Run training

```shell
python scripts/train_model.py --config config_files/kamba_asr_w2v_bert_10h.yaml
```

On first run, the script streams ~10 hours of audio from HuggingFace (takes ~20 minutes).  
On subsequent runs it loads from the local cache.

### 6. Run evaluation

```shell
python scripts/evaluate_kamba_asr.py \
  --model_path outputs/kamba-asr-facebook-w2v-bert-2/<experiment_folder> \
  --experiment_name kamba_w2v_bert_10h \
  --split test \
  --batch_size 4
```

---

## Running on Google Colab

A self-contained Colab notebook is available at `Kamba_ASR_Colab.ipynb`.  
It handles Drive mounting, streaming with caching, training, and evaluation in a single notebook.

**Quick start:**
1. Open in Colab
2. `Runtime → Change runtime type → T4 GPU`
3. Run all cells — outputs save automatically to `My Drive/KambaASR/`

---

## Project Structure

```
kamba-asr-facebook-w2v-bert-2/
├── config_files/
│   └── kamba_asr_w2v_bert_10h.yaml     
├── post_processing/
│   └── normalization.py                
├── scripts/
│   ├── train_model.py                  
│   └── evaluate_kamba_asr.py           
├── src/
│   ├── data/
│   │   ├── dataset.py                  
│   │   ├── dataset_encoders.py
│   │   └── preprocessing.py
│   ├── models/
│   │   └── factory.py
│   ├── training/
│   │   ├── collator.py
│   │   ├── metrics.py
│   │   └── trainer.py
│   └── utils/
│       └── config.py
├── Kamba_ASR_Colab.ipynb              
├── .env                                
└── README.md
```

---

## Linguistic Notes on Kamba Orthography

Standard text normalisation tools are not designed for Kamba and will incorrectly strip characters that are orthographically and phonemically essential:

| Character | Role | Example |
|---|---|---|
| `'` (apostrophe) | Marks the velar nasal in `ng'`  | `ng'ombe` (cow) |
| `ĩ` (i-tilde) | Nasalized high front vowel — distinct from plain `i` | `nĩ` (is/are) |
| `ũ` (u-tilde) | Nasalized high back vowel — distinct from plain `u` | `ũla` (that) |

---

## Supervision

This project was conducted under the supervision of **Prof. David Ifeoluwa Adelani**.


## References 

> ```
> @misc{abdullah2024ethioasr,
>   title={Ethio-ASR: ...},
>   author={Badr M. Abdullah et al.},
>   year={2024}
> }
> ```
