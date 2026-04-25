# Attribution

Maps to course rubric items **#110** (data sources) and **#98** (AI tool usage).

## Data Sources

- **Harry Potter Movies Dataset** (Kaggle) — primary source of dialogue lines
  https://www.kaggle.com/datasets/maricinnamon/harry-potter-movies-dataset
- **Harry Potter Data** (gastonstat / GitHub) — supplementary character/scene metadata consulted during dataset preparation
  https://github.com/gastonstat/harry-potter-data

The raw `Dialogue.csv`, `Characters.csv`, and `Data_Dictionary.csv` under `data/raw/` are taken directly from the Kaggle dataset (see its `LICENSE` in the same directory).

## Course Lecture References

- **Lecture 05** — dropout regularization theory; cited in the classifier hyperparameter analysis when explaining why `dropout=0.3` underfit on a 578-sample dataset.
- **Lecture 10** — BERT fine-tuning baseline; rationale for choosing `bert-base-uncased` as the classifier backbone.

## AI Tool Usage

Per rubric #98, this section documents where AI tools (Claude, GPT-4o-mini API) assisted the project. Items are grouped by stage; "AI-generated" means the code was largely produced by AI, "AI-assisted" means I wrote the bulk and AI helped with one or more parts.

### Stage 1 — Dataset Preparation
- Dataset filtering / column selection / row selection: AI-assisted
- `.gitignore`: AI-drafted
- `src/data/prepare_datasets.py` `print_distribution()` helper: AI-drafted

### Stage 1 — Emotion Labeling
- Used **GPT-4o-mini** API to generate initial emotion labels for the 600-sample dataset, following the annotation guidelines in `docs/annotation_guidelines.md`. Iterated prompts across two versions with manual accuracy checks on a 100-sample validation set; accuracy improved from 87/100 (v1) to 93/100 (v2). Iteration details in [`docs/iteration_log.md`](docs/iteration_log.md#1-emotion-annotation-stage-1).

### Stage 2 — Baseline (GPT-4o-mini prompt comparison)
- Analysis of the 30-question × 3-prompt-variant baseline run was AI-assisted (Claude wrote the comparison narrative; I selected the questions and ran the experiment). Findings in [`docs/iteration_log.md` §5 "Baseline analysis"](docs/iteration_log.md#baseline-analysis-prompt-c-wins-on-style).

### Stage 3 — LoRA Training (notebook `01_hermione_lora_training.ipynb`)
- `format_sample()` helper and dataset formatting code: AI-generated
- Phase 3.3 tokenization code: AI-modified (reused from Phase 3.2 with adjustments)
- "Active token distribution check" code: AI-generated
- Comparison-table builder code: AI-generated
- Training-loss plot code: AI-assisted
- Phase 3.5: of 100 diverse prompts for Hermione response generation, ~40 written by hand and the remainder were AI-suggested
- Phase 3.5 distribution-shift analysis write-up: AI-drafted

### Stage 4 — Emotion Classifier (notebook `02_hermione_emotion_classifier.ipynb`)
- File-import / loader functions: AI-assisted
- v1-vs-v2 prediction agreement comparison: AI-assisted
- `plot_training_curves()`: AI-assisted

### Stage 5 — Pipeline Integration (notebook `03_pipeline_integration.ipynb`)
- LoRA + BERT joint model loader: AI-assisted
- `compel` package (used for SD prompt weighting): introduced via AI suggestion

### Substantial Debug / Rework Cases

The following non-trivial issues required diagnosis and intervention beyond what AI tools produced on first pass:

- **Phase 3.2 — Loss masking bug**: initial training loss started at 5.16 because labels included instruction template tokens. Diagnosed by inspecting per-token loss; fixed by setting prefix and padding tokens to -100, dropping initial loss to 3.8.
- **Phase 3.5 — Emotion conditioning failure on 'angry' class**: observed only 12.5% intent-match rate; root cause traced to training-set imbalance (10.4% angry samples) compounded by Phi-2's prior toward cautious phrasing.
- **Stage 5 — SD identity drift under emotion weighting**: Compel weight syntax (`++`) initially destabilized character identity; resolved by applying matching weights to identity tokens.
