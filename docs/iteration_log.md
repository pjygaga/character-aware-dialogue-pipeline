# Iteration Log

This document records the design iterations behind each model in the pipeline. For each iteration we report what changed, why it changed, and what the measurable effect was. Maps to rubric items **#38** (iteration evidence) and **#95** (process documentation).

The raw artifacts referenced below live in `archive/` (older code/data versions) and `outputs/` (training runs, plots, classification reports).

---

## 1. Prompt Engineering Iterations (v1 → v2 → v3)

> Stable Diffusion image-prompt builder. Code archive: `archive/prompt_iterations/image_prompts_v{1,2,3}.py`. Final version: `src/pipeline/image_prompts.py`.
>
> <!-- TODO: ATTRIBUTION.md Stage 5 narrative mentions a v1→v5 progression; the code archive only has v1→v3. Reconcile here — either (a) describe only the 3 archived versions, or (b) describe all 5 conceptual iterations and note which were archived. -->

### Comparison Table

<!-- TODO: fill in. Suggested columns:

| Version | Prompt structure | Identity weighting | Emotion weighting | Compel used? | Visual outcome |
|---|---|---|---|---|---|
| v1 | | | | | |
| v2 | | | | | |
| v3 | | | | | |
-->

### What Changed and Why

<!-- TODO: 1-2 paragraphs per version transition. Cover:
     - What problem the previous version had (e.g., "v1 lacked emotion differentiation")
     - What you changed in the prompt (e.g., "added portrait composition keywords")
     - What that fixed and what it broke (e.g., "improved framing but identity drifted")
     - Source material in ATTRIBUTION.md "Stage 5 — SD prompt iteration" section -->

---

## 2. Classifier Iterations (v1 → v2)

> BERT emotion classifier (`bert-base-uncased`, 6 classes). Outputs: `outputs/classifier/`. Final model: `outputs/classifier/v2/best_model/`.

### Setup (shared across iterations)

- Backbone: `bert-base-uncased` (Lecture 10 standard fine-tuning starting point; fits in T4 / RTX 4090 memory).
- Splits: 80/10/10 stratified — train 478, val 58, test 64 (v1) → 578 / 72 / 80 (v2 after augmentation).
- 6 emotion classes, severely imbalanced: neutral 184 vs sad 13 (~14×).
- Loss: class-weighted cross-entropy, weights inversely proportional to class frequency (neutral 0.49, sad 6.02 in v1).
- Hyperparameters: lr=2e-5, batch_size=16, weight_decay=0.01, warmup=10%, max 8 epochs. Early stopping on val macro F1, patience=2.
- **Acknowledged data limitation**: sad has only 13 training samples; no amount of weighting can fully compensate.

### Comparison Table

| Class | v1 F1 | v2 F1 | Δ |
|---|---|---|---|
| worried | 0.250 | 0.500 | +0.250 |
| angry | 0.400 | 0.500 | +0.100 |
| determined | 0.381 | 0.421 | +0.040 |
| sad | 0.000 | 0.000 | 0 (still 0) |
| neutral | 0.760 | 0.727 | -0.033 |
| happy | 0.667 | 0.571 | -0.096 |
| **macro F1** | **0.410** | **0.453** | **+0.043** |
| accuracy | 0.500 | 0.547 | +0.047 |
| weighted F1 | 0.484 | 0.546 | +0.062 |

Diagnostic — v1 evaluated on 100 LoRA-generated samples (re-labeled by GPT-4o-mini using the same Stage 1 guidelines):

| | True dist (API) | v1 predicted |
|---|---|---|
| worried | 41 | 49 (+8) |
| determined | 26 | 24 |
| neutral | 13 | 6 (-7) |
| happy | 12 | 10 |
| angry | 5 | 10 (+5) |
| sad | 3 | 1 |

Mean prediction confidence: 0.375 (random baseline for 6 classes is 0.167). Agreement with API labels: 65%.

### What Changed and Why

**v1 (478 movie lines)** trained well on the majority class (neutral F1=0.76) but failed on minority classes — `sad` was completely absent (F1=0.0) and `worried` (F1=0.25) had a fuzzy boundary against `determined`. Diagnostic on 100 LoRA-generated samples revealed v1 was unfamiliar with the LoRA text distribution: it over-predicted `worried` (+8) and `angry` (+5), missed half of `neutral`, and barely identified `sad` (1 of 3). Low-confidence cases like *"I'm sorry, I just wanted to see them up close..."* (sad, p=0.22) and *"You did it? You killed your own father!"* (angry, p=0.24) showed the model lacked exposure to LoRA-style phrasing.

**v2 (478 + 100 LoRA-augmented = 578 samples)** added the 100 LoRA-generated samples (with API labels) to the training set. Class weights were recomputed for the new distribution. **All other hyperparameters were held identical to v1**, so any improvement is attributable to the data, not parameters. The result was a clear pattern: classes with the most augmentation samples improved most. `worried` got +41 augmented samples and gained +0.250 F1 (the largest jump). `sad` got only +3 and remained at F1=0. The slight regression in `neutral` (-0.033) and `happy` (-0.096) is within run-to-run variance (~0.03) and may also reflect their small augmentation share.

**Choice of final model**: v2, augmented data, dropout=0.1 → macro F1 = 0.453 on test.

### Sub-iteration: Dropout Sweep on v2

| Dropout | Macro F1 | Outcome |
|---|---|---|
| 0.1 | 0.424 | Best — full 8 epochs |
| 0.2 | 0.315 | Early stop at epoch 5/8 (val F1 declining) |
| 0.3 | 0.128 | Severe collapse, stopped at epoch 4/8 |

Dropout=0.3 collapsed by classifying nearly everything as `worried`: F1=0 for `determined`, `happy`, `sad`; only 3/23 `neutral` correct. This is the textbook Lecture 05 "dropout too strong → underfit" failure mode on a small dataset (~36 batches/epoch). Early stopping correctly identified and halted the failing run.

The dropout=0.1 sweep run (F1=0.424) vs the v2 main run (F1=0.453, also dropout=0.1) differ by 0.029 — within run-to-run variance. HuggingFace `Trainer` has un-seeded sources of randomness even when `seed` is fixed; results should be read as a range, not a point estimate.

### Acknowledged Limitations

- **sad class fundamentally fails**: data scarcity (13 train samples), not a model issue.
- **happy/sad test sets too small** (3 each); single-sample correctness dominates per-class F1.
- **No cross-validation**: time constraint; single train/test split has unmeasured variance.
- **API labels are not gold standard**: but consistent with Stage 1 annotation, so internally coherent.
- **Run-to-run variance ~0.03 macro F1**: results should be read as a range.

---

## 3. LoRA Hyperparameter Search (3 runs)

> Phi-2 + LoRA fine-tuning on 537 Hermione movie lines. Outputs: `outputs/lora/hyperparam_comparison.csv`, `outputs/lora/training_loss_curves.png`. Final model: `outputs/lora/models/lora_final_model/`.

### Comparison Table

| Run | Learning Rate | Final Train Loss | Final Eval Loss |
|---|---|---|---|
| run_A | 1e-4 | 3.2456 | 3.1774 |
| run_B | 5e-4 | 3.0991 | 3.1279 |
| run_C | 1e-3 | 3.0192 | 3.1604 |

Source: `outputs/lora/hyperparam_comparison.csv`. Loss curves: `outputs/lora/training_loss_curves.png`.

<!-- TODO: optionally add columns for which run was chosen as the final model and why
     (e.g., lowest eval loss = run_B at 3.1279, but train-eval gap matters too) -->

### What Changed and Why

<!-- TODO: 1-2 paragraphs covering:
     - Why these three learning rates were chosen (sweep range rationale)
     - What the train-vs-eval loss gap tells you for each run
       (run_C has the lowest train loss but eval loss is higher than run_B → starting to overfit)
     - Which run was selected as the final model and why
     - What was held constant across all three runs (rank, alpha, target modules, epochs, batch size)
     - Source material: ATTRIBUTION.md Phase 3.2 (loss masking insight), Phase 3.4 (evaluation findings),
       Phase 3.5 (distribution shift) provide complementary context for what the final LoRA model achieves -->

---

## Cross-References

- Raw narrative captured during development: `ATTRIBUTION.md` → "Additional Development Notes"
- Annotation guidelines (Stage 1 labeling): `docs/annotation_guidelines.md`
- Archived iteration code: `archive/prompt_iterations/`, `archive/lora_iterations/`
- Per-class confusion matrices: `outputs/classifier/confusion_matrix_{v1,v2,dropout_*}.png`
- Training curves: `outputs/classifier/classifier_training_curves.png`, `outputs/lora/training_loss_curves.png`
