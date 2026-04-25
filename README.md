# Character-Aware Dialogue Pipeline

> CS372 final project — generates emotion-conditioned, character-styled dialogue and matching Stable Diffusion portraits for Hermione Granger (Harry Potter).

<!-- TODO: replace this one-line description with your final framing if it changes -->

## What It Does

<!-- TODO: 用 3-5 句话讲清楚 (rubric #112)
     建议覆盖:
     - 输入是什么 (用户提问 / 对话上下文)
     - 中间发生了什么 (LoRA 生成台词 → BERT 分类情绪 → SD 出图)
     - 输出是什么 (一段 Hermione 风格台词 + 一张匹配情绪的人物肖像)
     - 它解决了什么问题 / 比裸 SD 或裸 LLM 好在哪 -->

The pipeline is split into four stages, each implemented in a Colab notebook under `notebooks/`:

| Stage | Notebook | What it does |
|---|---|---|
| 0 | `00_sd_baseline_test.ipynb` | Sanity-check Stable Diffusion 1.5 on the target character |
| 1 | `01_hermione_lora_training.ipynb` | Fine-tune Phi-2 with LoRA on 537 Hermione movie lines |
| 2 | `02_hermione_emotion_classifier.ipynb` | Train a BERT emotion classifier (6 classes) |
| 3 | `03_pipeline_integration.ipynb` | End-to-end: text → classifier → emotion-conditioned SD prompt → image |

Source modules under `src/` are extracted utilities; the notebooks are the canonical implementation.

## Quick Start

```bash
# 1. Clone and enter
git clone <repo-url>
cd character-aware-dialogue-pipeline

# 2. Install dependencies (see SETUP.md for full setup)
pip install -r requirements.txt

# 3. Copy env template and fill in your OpenAI key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# 4. Run a stage notebook
jupyter notebook notebooks/03_pipeline_integration.ipynb
```

For Colab / GPU / model-weight setup see [SETUP.md](SETUP.md).

## Video Links

<!-- TODO (rubric #114): paste your video URLs here.
     建议放:
     - 项目演示视频 (3-5 min, end-to-end demo)
     - 个人贡献说明视频 (如果是组队作业) -->

- **Demo video**: <!-- TODO: YouTube / Bilibili link -->
- **Walkthrough video**: <!-- TODO: optional second link -->

## Evaluation

<!-- TODO (rubric #115): summarize quantitative results.
     可以从 docs/iteration_log.md 里搬最终数字过来,例如:

     | Component | Metric | Score |
     |---|---|---|
     | Emotion classifier (final) | macro F1 (test) | 0.453 |
     | LoRA emotion control | intent–output match | 41% (vs 16.7% random) |
     | LoRA style transfer | qualitative — see Phase 3.4 in iteration_log | — |
     | SD pipeline | qualitative — see archive/demo_outputs_full/ | — |

     另外加 1-2 段说明:
     - 评估方法 (test split / API re-labeling / human inspection)
     - 已知 limitation (sad 类样本少, SD 1.5 身份漂移等) -->

Detailed iteration analysis (per-class F1, distribution-shift findings, hyperparameter sweep) is in [docs/iteration_log.md](docs/iteration_log.md).

Demo outputs: see `outputs/pipeline/pipeline_demo_v1/` for representative images. Full 16-image demo set is preserved in `archive/demo_outputs_full/`.

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
└── archive/                 # iteration evidence (prompt/lora versions, full demo set)
```

## License

See [LICENSE](LICENSE). Dataset license terms in `data/raw/LICENSE`.
