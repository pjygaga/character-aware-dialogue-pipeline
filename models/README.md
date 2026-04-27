# Models

Trained model weights, configurations, tokenizers, and helper scripts for the character-aware dialogue pipeline.

## Layout

| Subdirectory | What ships in git | What does NOT ship | Source |
|---|---|---|---|
| `hermione_lora_adapter/` | Full LoRA adapter (~21 MB): `adapter_model.safetensors`, `adapter_config.json`, tokenizer, `training_metadata.json`, `README.md`. **Ready for inference after `git clone`.** | nothing | `notebooks/01_hermione_lora_training.ipynb` |
| `emotion_classifier_v2/` | `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`, plus two helper scripts (`load_model.py`, `train_classifier.py`). | `model.safetensors` (~440 MB, exceeds GitHub limit) | `notebooks/02_hermione_emotion_classifier.ipynb` |

## How to use each model

### LoRA adapter — works immediately after clone

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "models/hermione_lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("models/hermione_lora_adapter")
```

### Emotion classifier — re-train or drop in your own checkpoint

The BERT weights file (`model.safetensors`, ~440 MB) is too large for git. Choose one path:

**Path A — re-train (recommended for graders who want to verify)**:

```bash
python models/emotion_classifier_v2/train_classifier.py
```

Reads `data/processed/{train,val,test}.json` plus the 100 LoRA-augmented samples at `outputs/classifier/lora_generated_100_labeled.json`, applies class-weighted CE with the original v2 hyperparameters (lr=2e-5, batch=16, dropout=0.1, weight_decay=0.01, max 8 epochs, early stopping patience=2), and writes the final checkpoint into this directory. Expected runtime: ~20 minutes on T4, ~5 minutes on RTX 4090. Original v2 test macro F1 = 0.453.

**Path B — supply your own `model.safetensors`** at `models/emotion_classifier_v2/model.safetensors`, then:

```python
from models.emotion_classifier_v2.load_model import load_classifier, EMOTION_LABELS
model, tokenizer = load_classifier()
```

`load_model.py` raises a clear `FileNotFoundError` if the weights are missing.

## End-to-end inference

For combining both models with Stable Diffusion 1.5, see `notebooks/03_pipeline_integration.ipynb`. That notebook expects the LoRA adapter to be present (it is, after clone) and the BERT checkpoint to exist (run Path A first if you have not).
