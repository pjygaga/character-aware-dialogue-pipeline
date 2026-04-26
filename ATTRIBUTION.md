# Attribution

Maps to course rubric items **#110** (data sources) and **#98** (AI tool usage).

## Data Sources

- **Harry Potter Movies Dataset** (Kaggle) — primary source of dialogue lines
  https://www.kaggle.com/datasets/maricinnamon/harry-potter-movies-dataset

The raw `Dialogue.csv`, `Characters.csv`, and `Data_Dictionary.csv` under `data/raw/` are taken directly from the Kaggle dataset (see its `LICENSE` in the same directory).

## Course Lecture References

The following course lectures directly informed the project's technical decisions:

- **Lecture 05 (Training Networks)** — Dropout regularization and early stopping. Used in the BERT classifier (notebook 02); hyperparameter sweep over dropout ∈ {0.1, 0.2, 0.3} discussed in docs/iteration_log.md §2.
- **Lecture 09 (Language Models)** — Decoder-only transformer architecture. Background for choosing Phi-2 (2.7B decoder-only) as the LoRA base model.
- **Lecture 10 (Fine-Tuning)** — Parameter-efficient fine-tuning, LoRA decomposition (W = W₀ + BA). Direct reference for the LoRA configuration in notebook 01 (rank=16, target q_proj/v_proj).
- **Lecture 11 (In-Context Learning)** — Few-shot prompting. Basis for the Stage 2 baseline prompt C, which used 3 in-context Hermione exemplars and outperformed zero-shot and instruction-only variants.
- **Lecture 13 (Diffusion)** — Forward/reverse process, classifier-free guidance. Background for SD 1.5 inference in notebooks 00 and 03.
- **Lecture 14 (Guided Diffusion)** — Text-conditional generation. Direct reference for the emotion-to-prompt mapping that conditions SD on classifier output.

## AI Tool Usage

Per rubric #98, this section documents where AI tools (Claude, GPT-4o-mini API) assisted the project. Items are grouped by stage; "AI-generated" means the code was largely produced by AI, "AI-assisted" means I wrote the bulk and AI helped with one or more parts.

### Stage 1 — Dataset Preparation
- Dataset filtering / column selection / row selection: AI-assisted
- `.gitignore`: AI-drafted
- `src/data/prepare_datasets.py` `print_distribution()` helper: AI-drafted

### Stage 1 — Emotion Labeling
- Used **GPT-4o-mini** API to generate initial emotion labels for the 600-sample dataset, following the annotation guidelines in `docs/annotation_guidelines.md`. Iterated prompts across two versions with manual accuracy checks on a 100-sample validation set; accuracy improved from 87/100 (v1) to 93/100 (v2). Annotation guidelines for v1 and v2 are preserved as `docs/annotation_guidelines_v1.md` and `docs/annotation_guidelines.md` respectively, allowing direct comparison of the rule changes that drove the 87→93 improvement. Iteration details in [`docs/iteration_log.md`](docs/iteration_log.md#1-emotion-annotation-stage-1).

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
- **Phase 3.5 — Emotion conditioning failure on 'angry' class**: observed only 12.5% intent-match rate; root cause traced to training-set imbalance (10.4% angry samples) compounded by Phi-2's prior toward cautious phrasing. *(Acknowledged limitation, not resolved within project scope — discussed further in error analysis.)*
- **Stage 5 — SD identity drift under emotion weighting**: Compel weight syntax (`++`) initially destabilized character identity; resolved by applying matching weights to identity tokens.

### Documentation & Repository Structure

The repository's markdown documentation (README.md, SETUP.md, ATTRIBUTION.md, docs/iteration_log.md) was structured and edited with Claude Code (April 2026). I provided the source content — experiment results, design decisions, and my own development notes and descriptions — and Claude organized it into the section layouts, surfaced internal inconsistencies, and rewrote scattered notes into coherent narrative form. The associated repository cleanup (archive/ creation, requirements.txt consolidation from notebook pip-install cells, removal of stale files) was also Claude-assisted.
