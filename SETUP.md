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

<!-- TODO: 如果你的 notebook 依赖某个具体的 Drive 路径或者预下载好的模型,在这里说清楚.
     例如:
     - LoRA 训练需要预先把 hermione_dataset_lora.json 放到 /content/drive/MyDrive/cs372/data/
     - LoRA inference 需要把训练好的 adapter_model 放到 /content/drive/MyDrive/cs372/models/lora_final_model/ -->

## Option B — Local installation

### Prerequisites
- **Python**: 3.10–3.12 (project was developed on 3.12)
- **CUDA**: 12.x with a recent NVIDIA driver if running stages 1–3 on GPU
- **GPU memory**: ≥ 10 GB (RTX 3060 / 4060 Ti / Colab T4 all work)
- **Disk**: ~15 GB for cached HuggingFace + SD weights

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

### Known local-vs-Colab gotchas

<!-- TODO: 如果你跑本地的时候踩了坑,在这里记一下,例如:
     - bitsandbytes 在 Windows 上需要装预编译 wheel
     - compel 0.x 和 diffusers 某版本不兼容
     - peft 加载 LoRA adapter 时如果 transformers 版本不对会 silently 退化
     如果没踩坑就删掉这一节 -->

---

## Pre-trained Artifacts

The repository **does not** ship trained model weights (too large for git). To reproduce inference without retraining:

| Artifact | Where to put it | How to obtain |
|---|---|---|
| LoRA adapter for Phi-2 | `outputs/lora/models/lora_final_model/` | <!-- TODO: 上传到 HuggingFace Hub / Drive 然后写下载链接 --> |
| Fine-tuned BERT classifier | `outputs/classifier/v2/best_model/` | <!-- TODO: 同上 --> |
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
