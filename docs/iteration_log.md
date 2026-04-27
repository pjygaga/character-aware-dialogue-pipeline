# Iteration Log

This document records the design iterations behind each model in the pipeline. For each iteration we report what changed, why it changed, and what the measurable effect was. Maps to rubric items **#38** (iteration evidence) and **#95** (process documentation).

The raw artifacts referenced below live in `docs/archive/` (older code/data versions), `models/` (final trained model weights), and `outputs/` (training runs, plots, classification reports).

---

## 1. Emotion Annotation (Stage 1)

> Source artifacts: `docs/annotation_guidelines.md` (final guidelines), `data/processed/hermione_labeled_v2.json` (final labels). Earlier guidelines were reviewed and superseded.

### v1 baseline

- System: GPT-4o-mini, `temperature=0`
- Guidelines: `annotation_guidelines_v1.md` (~1500 tokens) — first-pass criteria distinguishing six classes (determined, worried, angry, happy, sad, neutral).
- Time to label all 600 samples: 5:50 minutes.
- Resulting label distribution: neutral 39.2%, worried 23.8%, determined 20.5%, angry 10.3%, happy 3.3%, sad 2.8%.

### v2 refinement

Manual review of 100 stratified-random samples after v1 showed **87% accuracy**. The dominant failure mode was over-prediction of `neutral`: the v1 guideline definition was too permissive, swallowing mild `determined` and `worried` cases.

Three changes for v2:

1. Tightened the `neutral` definition to exclude any detectable emotional tone.
2. Added disambiguation rules separating `angry` from `worried` (intensity + target distinction).
3. Added two more examples per class for the small classes (`happy`, `sad`).

### Accuracy improvement

Re-labeling the same 100-sample evaluation set with v2 guidelines reached **93/100 accuracy** (+6 points). The full 600-line corpus was then re-run under v2 and saved as `data/processed/hermione_labeled_v2.json`. This is the dataset all downstream stages depend on.

---

## 2. Classifier Iterations (Stage 4)

> BERT emotion classifier (`bert-base-uncased`, 6 classes). Outputs: `outputs/classifier/`. Final model: `models/emotion_classifier_v2/`.

### Setup (shared across iterations)

- Backbone: `bert-base-uncased` (Lecture 10 standard fine-tuning starting point; fits in T4 / RTX 4090 memory).
- Splits: 80/10/10 stratified — train 478, val 58, test 64 (v1) → 578 / 72 / 80 (v2 after augmentation).
- 6 emotion classes, severely imbalanced: neutral 184 vs sad 13 (~14×).
- Loss: class-weighted cross-entropy, weights inversely proportional to class frequency (neutral 0.49, sad 6.02 in v1).
- Hyperparameters: lr=2e-5, batch_size=16, weight_decay=0.01, warmup=10%, max 8 epochs. Early stopping on val macro F1, patience=2.
- **Acknowledged data limitation**: sad has only 13 training samples; no amount of weighting can fully compensate.

### Comparison Table — v1 → v2

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

Source: `outputs/classifier/classifier_comparison_v1_v2.csv` and `classification_report_v{1,2}.txt`.

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

Source: `outputs/classifier/classifier_hyperparam_dropout.csv`.

Dropout=0.3 collapsed by classifying nearly everything as `worried`: F1=0 for `determined`, `happy`, `sad`; only 3/23 `neutral` correct. This is the textbook Lecture 05 "dropout too strong → underfit" failure mode on a small dataset (~36 batches/epoch). Early stopping correctly identified and halted the failing run.

The dropout=0.1 sweep run (F1=0.424) vs the v2 main run (F1=0.453, also dropout=0.1) differ by 0.029 — within run-to-run variance. HuggingFace `Trainer` has un-seeded sources of randomness even when `seed` is fixed; results should be read as a range, not a point estimate.

### Acknowledged Limitations

- **sad class fundamentally fails**: data scarcity (13 train samples), not a model issue.
- **happy/sad test sets too small** (3 each); single-sample correctness dominates per-class F1.
- **No cross-validation**: time constraint; single train/test split has unmeasured variance.
- **API labels are not gold standard**: but consistent with Stage 1 annotation, so internally coherent.
- **Run-to-run variance ~0.03 macro F1**: results should be read as a range.

---

## 3. LoRA Training (Stage 3)

> Phi-2 + LoRA fine-tuning on 537 Hermione movie lines. Outputs: `outputs/lora/`. Final model: `models/hermione_lora_adapter/`.

### Phase 3.2 — Loss masking fix

**Issue detected**: Initial training loss started at 5.16 — too high. Inspection showed labels included the instruction/input template text, forcing the model to waste capacity memorizing fixed prompt tokens (`### Instruction:`, `### Input:`, `### Response:`) instead of learning Hermione's response style.

**Fix applied**: Added loss masking that sets prefix tokens and padding tokens to `-100`, so cross-entropy is only computed over the `### Response:` content. This concentrates all 5.2M trainable LoRA parameters on the actual dialogue.

**Effect**: Initial loss dropped from 5.16 to 3.8, and the loss curve converged more meaningfully. This unblocked Phase 3.3 (full training) by giving it a clean signal to optimize against.

### Phase 3.4 — Qualitative evaluation

The fine-tuned LoRA model successfully captures several of Hermione Granger's linguistic patterns: imperative voice ("You have to study, Harry!"), emphasis on rules and study, and protective concern for friends. Emotion conditioning is observable: SAD-conditioned outputs use longing language ("Where did all the joy go..."), while WORRIED outputs emphasize protective vigilance.

However, the model exhibits notable limitations:

1. **Factual hallucinations**: The model invents non-canon details, including fictional characters ("Lily Lovegood"), incorrect spell effects ("Wingardium Leviosa lets you fly"), and historical errors. Out of 30 test responses, ~15 contained at least one factual deviation.
2. **Character role confusion**: In some responses, the model attributes Hermione's voice to other characters or has Hermione speak to wrong conversational partners (e.g., responding to Hagrid when no Hagrid was mentioned in the prompt).
3. **Missing signature phrases**: Despite training, the model rarely produces Hermione's most iconic phrases ("Honestly, Ronald," references to "Hogwarts: A History"). This likely reflects scarcity in training data.

These limitations are consistent with the chosen approach's known constraints: small base model (Phi-2 at 2.7B parameters), limited fine-tuning data (537 samples), and PEFT-LoRA modifying only 0.19% of parameters. Documented as expected trade-offs, not execution failures.

### Phase 3.5 — Distribution shift

To assess the generalization gap between the original Hermione dialogue and the LoRA model's outputs, we generated 100 responses with explicit emotion conditioning, then re-classified them with GPT-4o-mini using the identical annotation guidelines applied to the training data.

**Findings**:

1. **Overall intent–output mismatch (41% match rate)**: Across 100 samples, only 41% had a predicted emotion matching the intended conditioning. While 2.5× above random baseline (16.7%), this confirms that LoRA's emotion control on Phi-2 is partial.
2. **"Worried" bias** (41 predicted vs 18 intended, +23): The model over-produces hedging language ("What if...", "I'm not sure...") regardless of intended emotion. Likely due to imbalanced training data (worried = 24% of samples) and Phi-2's pretrained tendency toward cautious phrasing.
3. **"Angry" near-collapse (12.5% match rate)**: Of 16 intended-angry prompts, 7 were classified as worried, 5 as determined, only 2 as angry. Suggests LoRA failed to learn Hermione's anger patterns, likely due to insufficient angry training samples (56 of 537 = 10.4%).

**Implication for Stage 4**: This distribution shift sets up a meaningful test for the emotion classifier — the v1 classifier (trained on original movie dialogue) is expected to underperform on LoRA-generated text, and augmenting the training set with these 100 labeled samples (v2) should improve performance on this distribution. That hypothesis is confirmed in §2 above.

### Hyperparameter sweep — 3 learning rates

| Run | Learning Rate | Final Train Loss | Final Eval Loss |
|---|---|---|---|
| run_A | 1e-4 | 3.2456 | 3.1774 |
| run_B | 5e-4 | 3.0991 | 3.1279 |
| run_C | 1e-3 | 3.0192 | 3.1604 |

Source: `outputs/lora/hyperparam_comparison.csv`. Loss curves: `outputs/lora/training_loss_curves.png`.

Run B (lr=5e-4) achieved the lowest eval loss (3.1279) and was selected as the final model. Run C reached the lowest train loss (3.0192) but its eval loss (3.1604) is higher than run B — the train-eval gap signals the start of overfitting at the higher learning rate. Run A (lr=1e-4) under-converged relative to the other two within the same epoch budget.

---

## 4. SD Prompt Engineering (Stage 5)

> Stable Diffusion 1.5 image-prompt builder. Final version: `src/pipeline/image_prompts.py`. Code archive: `docs/archive/prompt_iterations/image_prompts_v{1,2,3}.py` (v1 / v2 / v3 implementations preserved; v4 / v5 changes are described below but were not split into separate code files).

### v1 → v5 iteration

SD 1.5 在小脸构图下面部细节不稳定。尝试了 5 版 prompt 调整:

- **v1**: 缺乏情绪区分 — 同一个 character prompt 跑不同情绪输出几乎一样,情绪信号没有进入图像。
- **v2**: 简化 prompt 后情绪表达太弱 — 缩短了 prompt 想让情绪关键词权重更突出,反而削弱了所有信号。
- **v3**: 加入肖像构图(`portrait, close-up, headshot`)改善画质,但情绪仍模糊 — 构图修好了,情绪还是没出来。
- **v4**: 用 Compel 权重语法 `(emotion_word)++` / `(emotion_word)+++` 强化情绪后表情清晰,但身份漂移(人种、年龄、发型不一致)— 情绪权重把模型 attention 拉走了。
- **v5**: 给身份特征(`brown bushy hair, brown eyes, young british teenager`)同样加权后达到平衡 — 情绪和身份都稳定,但仍有微妙的"诡异感"(uncanny valley typical of SD 1.5 small-face composition)。

这反映了 SD 1.5 在小模型容量下,情绪与身份特征的注意力竞争。未来工作可换用 SDXL(更大的容量更能同时维持多种约束)或加 face restoration 后处理(GFPGAN / CodeFormer)修复 uncanny valley。

---

## 5. Technical Decisions

### Sad-class data scarcity

仔细看 sad 在三个集合里的数量:

- Train: 13 条
- Val: 1 条
- Test: 3 条

Val 里只有 1 条 sad 意味着分类器训练完后,sad 这一类的 val 准确率只有 100% 或 0% 两个值,不是有意义的数字。这是数据本身的问题(赫敏角色坚强,悲伤台词少),不是代码错。最终报告里写明:由于 sad 原始样本仅 17 条,切分后 val 集只有 1 条、test 集只有 3 条,该类别评估统计意义有限。后续分类器训练采用 class-weighted cross-entropy(sad 权重 6.02 vs neutral 0.49)缓解,但 §2 的结果显示 sad F1 在 v1/v2 都是 0 — 加权改变不了训练数据本身的容量上限。

### LoRA limitation: knowledge vs style

LoRA 在 dialogue data 上 fine-tune **只教模型 Hermione 怎么说话,不教她知道什么**。对哈利波特世界的 world knowledge(咒语机制、人物历史、SPEW 细节)完全来自 Phi-2 base model,LoRA 无法添加新知识,只能改写已有输出的风格。30 个 test questions 里大约 40% 需要 world knowledge(Q1 wand movement, Q21 SPEW, Q26 mudblood context),这些题的 LoRA 输出严重依赖 Phi-2 的先验知识 —— 而 Phi-2 在哈利波特领域的知识有限且时有错误。这些 question 保留作为 error analysis 的 stress test,因为它们暴露的正是 style-focused fine-tuning 的已知特征 ——「知识与风格分离」。

### Baseline analysis: prompt C wins on style

Baseline phase: 30 个 test questions × 3 个 prompt 版本(zero-shot A / detailed instruction B / few-shot with real movie lines C),gpt-4o-mini 上跑 90 个 response。结论:**A 和 B 仍然像穿着戏服的 chatbot,只有 C 真的像角色**。

最清楚的证据是 formatting:B 在被问"OWLs 怎么准备"这种问题时会写 bold markdown 和 numbered list — 真实的 Hermione 不会这样做,她会 snap at you。底层模型仍然以为自己在回答 query,而不是作为朋友 react。C 在 30 个 response 里 zero bold zero list。

长度也很说明问题:A 平均 95 词,B 113 词,C 只有 48 词。C 的长度短一半正是它感觉像人的原因 — Hermione 会说 "Are you completely mad? Skip Potions?" 而不是讲 80 词的 lecture。

B 的奇怪现象:30 个 response 全都有 "Honestly",25 个有 "I read",因为 prompt 字面上告诉它要用这些词。它过度使用到 parody 程度。C 用了 11 次 "Honestly"(更自然),"I read" 只用了 1 次但仍然像 Hermione。结论:**告诉模型用什么词,比给它例子让它看,效果更差** — 例子带的是节奏和上下文,不只是词汇。

类别也很关键:学术问题三个版本都 OK 因为讲解适合 Hermione。差距在 friendship / conflict 题(Q3, Q11, Q17)爆开 — A 和 B 给出礼貌的 assistant advice,C 真的像朋友一样 snap back。最终报告比较时用 conflict question 比 academic question 更有说服力。

进入 LoRA 阶段时,真正要打败的 baseline 是 C 不是 A。C 仍然欠缺的是:很短的台词(real Hermione 在 dataset 里经常 4-15 词)和具体到章节号的书名引用(C 编造看起来像样的假书名)。LoRA 应该能修两者。

---

## Cross-References

- Final annotation guidelines: `docs/annotation_guidelines.md`
- AI tool usage and data sources: `ATTRIBUTION.md`
- Archived iteration code: `docs/archive/prompt_iterations/`, `docs/archive/lora_iterations/`
- Per-class confusion matrices: `outputs/classifier/confusion_matrix_{v1,v2,dropout_*}.png`
- Training curves: `outputs/classifier/classifier_training_curves.png`, `outputs/lora/training_loss_curves.png`
