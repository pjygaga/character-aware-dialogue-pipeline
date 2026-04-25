# Attribution

Maps to course rubric items **#110** (data sources) and **#98** (AI tool usage).

## Data Sources

- **Harry Potter Movies Dataset** (Kaggle) — primary source of dialogue lines
  https://www.kaggle.com/datasets/maricinnamon/harry-potter-movies-dataset
- **Harry Potter Datasets** (gastonstat / GitHub) — supplementary character/scene metadata
  https://github.com/gastonstat/harry-potter-datas

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
- Used **GPT-4o-mini** API to generate initial emotion labels for the 600-sample dataset, following the annotation guidelines in `docs/annotation_guidelines.md`. Iterated prompts across three versions (v1 → v2 → final) with manual accuracy checks on a 50-sample validation set; accuracy improved from 87/100 (v1) to 93/100 (v2).

### Stage 2 — Baseline (GPT-4o-mini prompt comparison)
- Analysis of the 30-question × 3-prompt-variant baseline run was AI-assisted (Claude wrote the comparison narrative; I selected the questions and ran the experiment).

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

---

## Additional Development Notes

> The notes below were captured during development in the original `来源.txt`. They describe iteration decisions, technical trade-offs, and observations. They are preserved here verbatim and will be migrated into `docs/iteration_log.md` as Step 5 of the cleanup.

### Stage 1 — Emotion-labeling iteration log

**v1: Initial Labeling**
- System: GPT-4o-mini, temperature=0
- Guidelines: `annotation_guidelines_v1.md` (1500 tokens)
- Time: 5:50 minutes
- Label distribution: neutral 39.2%, worried 23.8%, determined 20.5%, angry 10.3%, happy 3.3%, sad 2.8%

**Observation**
Manual review of 100 random samples (stratified by label) showed 87% accuracy. Key issue: 'neutral' class was over-predicted, likely due to overly permissive definition in guidelines. Mis-labeled 'neutral' samples often belonged to mild 'determined' or 'worried' categories.

**v2: Refined Guidelines**
1. Tightened 'neutral' definition to exclude any detectable emotional tone
2. Added disambiguation rules for 'angry' vs 'worried'
3. Added 2 more examples per class for 'happy' and 'sad'

Re-labeled same 100 samples with v2 guidelines. Accuracy improved to 93/100.

**Decision**
Re-ran all 600 lines with v2 guidelines. Final dataset saved to `data/processed/hermione_labeled_v2.json`.

### Stage 1 — Sad-class data scarcity (technical decision record)

仔细看 sad 在三个集合里的数量：

- Train: 13 条
- Val: 1 条
- Test: 3 条

Val 里只有 1 条 sad 意味着分类器训练完后,sad 这一类的 val 准确率是 100% 或 0%,不是有意义的数字。这是数据本身的问题(赫敏角色坚强,悲伤台词少),不是代码错。要在报告里写明:由于 sad 原始样本仅 17 条,切分后 val 集只有 1 条、test 集只有 3 条,该类别评估统计意义有限。后续分类器训练采用加权损失(weighted loss)缓解。

### Stage 2 — Baseline analysis summary

Baseline phase: ran 30 test questions across three prompt versions (zero-shot A, detailed instruction B, few-shot with real movie lines C) on gpt-4o-mini, 90 responses total. A and B still sound like a chatbot in costume, C actually sounds like a character.

The clearest evidence is formatting. B writes bold markdown and numbered lists when asked things like how to prep for OWLs, which real Hermione would never do — she would snap at you. The underlying model still thinks it is answering a query, not reacting as a friend. C has zero bold and zero lists across all 30 responses.

Length is striking once you quantify it. A averages 95 words, B 113, C only 48. C is half the length and that is exactly why it feels human — Hermione snaps things like "Are you completely mad? Skip Potions?" instead of lecturing for 80 words.

Something weird with B: it has "Honestly" in all 30 responses and "I read" in 25, because the prompt literally told it to use those words. It now overdoes them to the point of parody. C uses "Honestly" 11 times which is more natural, and "I read" only once but still feels more like Hermione. So telling the model what words to use is worse than showing it examples — examples carry rhythm and context not just vocabulary.

Category matters a lot. On academic questions all three are OK because lecturing fits. The gap explodes on friendship and conflict questions (Q3, Q11, Q17), where A and B give polite assistant advice and C actually snaps back like a friend. For the final report the best comparison is a conflict question, not an academic one.

Going into LoRA, the real baseline to beat is C, not A. What C still misses is the very short clipped responses (real Hermione often has 4-15 word lines in the dataset) and specific book citations with chapter numbers — C makes up generic-sounding titles. LoRA should hopefully fix both.

### Stage 3 — LoRA limitation (knowledge vs style)

Realized an important limitation of LoRA fine-tuning. LoRA on dialogue data only teaches the model **how Hermione speaks, not what she knows**. World knowledge about the Harry Potter universe (spell mechanics, character history, SPEW details) comes entirely from the Phi-2 base model, and LoRA cannot add new knowledge — only restyle existing outputs. About 40% of the 30 test questions require world knowledge (Q1 wand movement, Q21 SPEW, Q26 mudblood context), and on these the LoRA output depends heavily on Phi-2's prior knowledge, which is limited and sometimes wrong. Keeping these questions as stress tests for the error-analysis section, since they reveal the knowledge-style separation that is a known characteristic of style-focused fine-tuning.

### Phase 3.2 — Loss masking key insight

**Issue detected**: Initial loss started at 5.16 (too high) because labels included the instruction/input template text, forcing the model to waste capacity memorizing fixed prompt tokens instead of learning Hermione's response style.

**Fix applied**: Added loss masking to set prefix tokens and padding tokens to `-100`, so the model only computes loss on the `### Response:` content, concentrating all 5.2M trainable LoRA parameters on learning the actual dialogue style.

**Expected improvement**: Loss curve now starts at 3.8 and converges more meaningfully, giving Phase 3.3's full training a much cleaner signal and better final model quality.

### Phase 3.4 — Evaluation findings

The fine-tuned LoRA model successfully captures several of Hermione Granger's linguistic patterns: imperative voice ("You have to study, Harry!"), emphasis on rules and study, and protective concern for friends. Emotion conditioning is observable: SAD-conditioned outputs use longing language ("Where did all the joy go..."), while WORRIED outputs emphasize protective vigilance.

However, the model exhibits notable limitations:

1. **Factual hallucinations**: The model invents non-canon details, including fictional characters ("Lily Lovegood"), incorrect spell effects ("Wingardium Leviosa lets you fly"), and historical errors. Out of 30 test responses, ~15 contained at least one factual deviation.
2. **Character role confusion**: In some responses, the model attributes Hermione's voice to other characters or has Hermione speak to wrong conversational partners (e.g., responding to Hagrid when no Hagrid was mentioned in the prompt).
3. **Missing signature phrases**: Despite training, the model rarely produces Hermione's most iconic phrases ("Honestly, Ronald," references to "Hogwarts: A History"). This likely reflects scarcity in training data.

These limitations are consistent with the chosen approach's known constraints: small base model (Phi-2 at 2.7B parameters), limited fine-tuning data (537 samples), and PEFT-LoRA modifying only 0.19% of parameters. Documented as expected trade-offs, not execution failures.

### Phase 3.5 — Distribution shift analysis

To assess the generalization gap between the original Hermione dialogue and the LoRA model's outputs, generated 100 responses with explicit emotion conditioning, then re-classified them with GPT-4o-mini using the identical annotation guidelines applied to the training data.

**Findings**:

1. **Overall intent-output mismatch (41% match rate)**: Across 100 samples, only 41% had a predicted emotion matching the intended conditioning. While 2.5× above random baseline (16.7%), this confirms that LoRA's emotion control on Phi-2 is partial.
2. **"Worried" bias** (41 predicted vs 18 intended, +23): The model over-produces hedging language ("What if...", "I'm not sure...") regardless of intended emotion. Likely due to imbalanced training data (worried = 24% of samples) and Phi-2's pretrained tendency toward cautious phrasing.
3. **"Angry" near-collapse (12.5% match rate)**: Of 16 intended-angry prompts, 7 were classified as worried, 5 as determined, only 2 as angry. Suggests LoRA failed to learn Hermione's anger patterns, likely due to insufficient angry training samples (56 of 537 = 10.4%).

**Implications for Phase 4**: This distribution shift sets up a meaningful test for the emotion classifier — the v1 classifier (trained on original movie dialogue) is expected to underperform on LoRA-generated text, and augmenting the training set with these 100 labeled samples (v2) should improve performance on this distribution.

### Stage 4 — Classifier iteration story (v1 → v2 → dropout sweep)

Migrated to [`docs/iteration_log.md`](docs/iteration_log.md#2-classifier-iterations-v1--v2). Includes setup, v1 baseline, the 100-sample LoRA diagnostic, v2 augmented-data results with per-class F1 deltas, the dropout sweep, and acknowledged limitations.

### Stage 5 — SD prompt iteration (v1 → v5)

SD 1.5 在小脸构图下面部细节不稳定。尝试了 5 版 prompt 调整:

- **v1**: 缺乏情绪区分
- **v2**: 简化 prompt 后情绪表达太弱
- **v3**: 加入肖像构图改善画质,但情绪仍模糊
- **v4**: 用 Compel 权重语法 `(++)+++` 强化情绪后表情清晰,但身份漂移(人种、年龄、发型不一致)
- **v5**: 给身份特征同样加权后达到平衡,但仍有微妙的"诡异感"

这反映了 SD 1.5 在小模型容量下,情绪与身份特征的注意力竞争。未来工作可换用 SDXL 或加 face restoration 后处理。
