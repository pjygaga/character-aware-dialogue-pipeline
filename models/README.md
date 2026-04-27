# Models

Trained model weights, configurations, and tokenizers for the character-aware dialogue pipeline.

## Layout

| Subdirectory | What it is | Produced by |
|---|---|---|
| `hermione_lora_adapter/` | LoRA adapter for Phi-2, fine-tuned on 537 Hermione movie lines (rank=16, target `q_proj`/`v_proj`). Contains `adapter_config.json`, `adapter_model.safetensors`, tokenizer files, and `training_metadata.json`. | `notebooks/01_hermione_lora_training.ipynb` |
| `emotion_classifier_v2/` | Fine-tuned `bert-base-uncased` 6-class emotion classifier (final v2 model with augmented training set). Contains `config.json`, `model.safetensors`, tokenizer files, and `training_args.bin`. | `notebooks/02_hermione_emotion_classifier.ipynb` |

## Reproducing the weights

The actual weight files (`*.safetensors`, `*.bin`) are excluded from git via `.gitignore` (the BERT classifier alone is ~440 MB). To regenerate them:

1. Run `notebooks/01_hermione_lora_training.ipynb` on a T4 GPU (≈90 minutes) → produces `hermione_lora_adapter/`.
2. Run `notebooks/02_hermione_emotion_classifier.ipynb` on a T4 GPU (≈20 minutes) → produces `emotion_classifier_v2/`.

See [`../SETUP.md`](../SETUP.md) for full setup and Colab path mapping.

## Loading

```python
# LoRA adapter on top of Phi-2
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
model = PeftModel.from_pretrained(base, "models/hermione_lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("models/hermione_lora_adapter")

# BERT emotion classifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer
clf = AutoModelForSequenceClassification.from_pretrained("models/emotion_classifier_v2")
clf_tokenizer = AutoTokenizer.from_pretrained("models/emotion_classifier_v2")
```

For end-to-end inference combining both models with Stable Diffusion 1.5, see `notebooks/03_pipeline_integration.ipynb`.
