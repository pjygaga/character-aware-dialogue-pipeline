# Setup Guide

Step-by-step installation for the character-aware dialogue pipeline. Maps to rubric item **#109**.

There are two supported environments: **local** (any OS, GPU recommended for stages 1–3) and **Google Colab** (the original development environment, T4 GPU sufficient).

---

## Option A — Google Colab (recommended, matches dev environment)

The four notebooks under `notebooks/` are written to run in Colab with a T4 GPU. Each notebook contains its own `!pip install` cell as the first executable cell, so package installation is automatic on first run.

1. Open Google Colab → `File` → `Upload notebook` → pick the desired notebook from `notebooks/`.
2. `Runtime` → `Change runtime type` → set Hardware accelerator to **T4 GPU**.
3. Mount Google Drive if the notebook expects it:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Run cells top to bottom.

### Required Google Drive layout for notebooks 01 and 03

Notebooks 01 (LoRA training + checkpoint backup) and 03 (pipeline inference) read and write to fixed Drive paths. Before running, ensure the following layout exists under your Drive:

| Path | Used by | Role |
|---|---|---|
| `/content/drive/MyDrive/cs372_final/lora_final_model/adapter/` | notebook 03 (read) | trained LoRA adapter — output of notebook 01 |
| `/content/drive/MyDrive/cs372_final/classifier_v2_best/` | notebook 03 (read) | trained BERT classifier — output of notebook 02 |
| `/content/drive/MyDrive/cs372_final/pipeline_outputs/` | notebook 03 (write) | demo image + JSON output destination |
| `/content/drive/MyDrive/cs372_final/` | notebook 03 (read) | also serves as `IMAGE_PROMPTS_DIR` for the Stage 5 prompt builder |
| `/content/drive/MyDrive/hermione_project_backups/phase3_4/` | notebook 01 (write) | mid-training checkpoint backup |

To use a different layout, edit the path constants near the top of each notebook (`LORA_MODEL_PATH`, `CLASSIFIER_PATH`, `OUTPUT_DIR`, `IMAGE_PROMPTS_DIR`, `DRIVE_BASE`).

## Option B — Local installation

### Prerequisites
- **Python**: 3.12 (the version this project was developed and tested on; older 3.10/3.11 may work but were not validated).
- **CUDA**: 12.x with a recent NVIDIA driver if running stages 1–3 on GPU.
- **GPU memory**: ≥ 10 GB (RTX 3060 / 4060 Ti / Colab T4 all work).
- **Disk**: ~15 GB for cached HuggingFace + SD weights.

### Step-by-step

```bash
# 1. Clone
git clone <repo-url>
cd character-aware-dialogue-pipeline

# 2. (Recommended) Create a virtualenv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# edit .env and fill in OPENAI_API_KEY (used by Stage 1 labeling and Stage 2 baseline)

# 5. (Optional) Pre-download model weights to avoid first-run delay
#    Phi-2 (~5 GB) and Stable Diffusion 1.5 (~4 GB) auto-download on first use.

# 6. Launch notebooks
jupyter notebook
# then open notebooks/00_..03_*.ipynb
```

---

## Pre-trained Artifacts

The repository **does not** ship trained model weights (too large for git). To reproduce inference without retraining:

| Artifact | Where to put it | How to obtain |
|---|---|---|
| LoRA adapter for Phi-2 | `outputs/lora/models/lora_final_model/` | Trained weights are not redistributed in this repository (data licensing + git LFS budget). To reproduce, run `notebooks/01_hermione_lora_training.ipynb` on a T4 GPU (≈90 minutes). |
| Fine-tuned BERT classifier | `outputs/classifier/v2/best_model/` | Same constraint as above. To reproduce, run `notebooks/02_hermione_emotion_classifier.ipynb` (≈20 minutes on T4). |
| Stable Diffusion 1.5 base | auto-downloaded by `diffusers` | no manual step |

To retrain from scratch, run the notebooks in order: 01 → 02. Stage 0 and 3 are inference-only.

---

## Verification

After install, sanity-check the environment:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import transformers, peft, diffusers; print('OK')"
```

If both print without error, you're ready to run the notebooks.
