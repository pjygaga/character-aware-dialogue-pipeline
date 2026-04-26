# Character-Aware Dialogue Pipeline

## What It Does

This project builds a three-model pipeline that produces in-character Hermione Granger dialogue with matching emotional portraits. A user query enters the pipeline, a LoRA-fine-tuned Phi-2 generates a response in Hermione's style, a BERT classifier labels the emotion of that response, and a Stable Diffusion 1.5 model conditioned on that emotion produces a portrait of the character delivering the line.

Off-the-shelf LLMs answer in a generic assistant voice, and off-the-shelf text-to-image models produce visually inconsistent characters. The pipeline addresses both: parameter-efficient fine-tuning shifts the LLM from assistant-voice to Hermione-voice (rubric items #42, #45, #46), and emotion-derived prompt conditioning grounds the SD output in the actual content of the generated text rather than a static character template.

Character-grounded multimodal generation is an active research area. Chen et al. (Findings of EMNLP 2023, "Large Language Models Meet Harry Potter: A Bilingual Dataset for Aligning Dialogue Agents with Characters") explores LLM character alignment using the Harry Potter universe; Lu et al. (ACL 2024, "Large Language Models are Superpositions of All Characters: Attaining Arbitrary Role-play via Self-Alignment") demonstrates that style fine-tuning alone is necessary but insufficient without grounding signals. This pipeline tests one specific grounding signal — emotion classification — on a small, controlled scope.

The pipeline is split into four stages, each implemented in a Colab notebook under `notebooks/`:

| Stage | Notebook | What it does |
|---|---|---|
| 0 | `00_sd_baseline_test.ipynb` | Sanity-check Stable Diffusion 1.5 on the target character |
| 1 | `01_hermione_lora_training.ipynb` | Fine-tune Phi-2 with LoRA on 537 Hermione movie lines |
| 2 | `02_hermione_emotion_classifier.ipynb` | Train a BERT emotion classifier (6 classes) |
| 3 | `03_pipeline_integration.ipynb` | End-to-end: text → classifier → emotion-conditioned SD prompt → image |

Reusable utilities (data loading, prompt templates, classifier inference helpers) live in `src/`. The four notebooks orchestrate the pipeline stages end-to-end and produce all artifacts under `outputs/`.

## Quick Start

```bash
# 1. Clone and enter
git clone <repo-url>
cd character-aware-dialogue-pipeline

# 2. Create a virtualenv (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
source .venv/bin/activate

# 3. Install dependencies (see SETUP.md for full setup)
pip install -r requirements.txt

# 4. Copy env template and fill in your OpenAI key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# 5. Run a stage notebook
jupyter notebook notebooks/03_pipeline_integration.ipynb
```

For Colab / GPU / model-weight setup see [SETUP.md](SETUP.md).

## Video Links

Videos will be added prior to final submission deadline.

- **Demo video**: _to be added_
- **Walkthrough video**: _to be added_

## Evaluation

| Component | Metric | Result |
|---|---|---|
| Emotion annotation (GPT-4o-mini) | accuracy on 100-sample manual review | 87% → 93% (v1 → v2) |
| LoRA emotion conditioning | intent–output match rate | 41% (vs 16.7% random baseline) |
| LoRA style transfer | qualitative inspection (30 test questions) | strong on imperative voice, weak on signature phrases — [see LoRA training notes](docs/iteration_log.md#3-lora-training-stage-3) |
| Emotion classifier | macro F1 on test set | 0.453 (accuracy 0.547, weighted F1 0.546) |
| Pipeline factual reliability | hallucination rate (30 LoRA responses) | ~50% contained at least one factual deviation |

Evaluation methodology. The classifier is evaluated on a held-out 10% test split (stratified by emotion class). LoRA emotion control is evaluated by re-classifying 100 LoRA-generated samples with the same GPT-4o-mini annotator used for the training labels — this measures whether the conditioning intent survived generation. Style-transfer and factual reliability are assessed qualitatively on a fixed 30-question test set covering academic, conflict, and moral-dilemma scenarios.

Known limitations. The "sad" emotion class has only 17 samples in the source dialogue (Hermione is a resilient character), leaving 1 sample in val and 3 in test — per-class metrics for sad are not statistically meaningful. The Stable Diffusion 1.5 base model exhibits mild identity drift on emotion-weighted prompts; see [SD prompt iteration history](docs/iteration_log.md#4-sd-prompt-engineering-stage-5).

Detailed iteration analysis (per-class F1, distribution-shift findings, hyperparameter sweep, classifier v1→v2 improvement deltas) is in [`docs/iteration_log.md`](docs/iteration_log.md).

Demo outputs: 16 PNG images covering the 6 emotion conditions are in `outputs/pipeline/pipeline_demo_v1/`, alongside `demo_results.json` with the prompt + classifier-emotion + filename mapping.

## Repository Layout

```
.
├── README.md                # this file
├── SETUP.md                 # full installation & GPU setup
├── ATTRIBUTION.md           # data sources, lecture refs, AI tool usage
├── LICENSE
├── requirements.txt
├── .env.example
├── notebooks/               # 4 Colab notebooks (stages 0–3)
├── src/                     # extracted utilities (data prep, baseline, classifier, pipeline)
├── data/                    # raw + extracted + processed datasets
├── outputs/                 # CSVs, plots, classifier reports, demo PNGs
├── docs/                    # annotation guidelines, iteration log
└── archive/                 # iteration evidence (earlier prompt and LoRA versions)
```

## License

See [LICENSE](LICENSE). Dataset license terms in `data/raw/LICENSE`.
