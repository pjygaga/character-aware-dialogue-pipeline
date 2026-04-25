# Archive — Iteration Evidence

This directory preserves intermediate artifacts that document the iterative development process. They are kept out of the active source tree to reduce noise but committed to the repository because they constitute graded deliverables.

> Maps to grading rubric items **#38** (iteration evidence) and **#95** (process documentation).

## Layout

| Path | What it contains | Why it's here |
|---|---|---|
| `prompt_iterations/` | Earlier versions of the SD image-prompt builder: `image_prompts_v1.py`, `_v2.py`, `_v3.py` | Shows the prompt-engineering trajectory before settling on the final `src/pipeline/image_prompts.py` |
| `lora_iterations/` | First-round LoRA generation outputs: `lora_responses_comparison_v1.csv`, `lora_responses_emotional_v1.json`, `lora_responses_neutral_v1.json` | Side-by-side comparison against the v2 outputs in `outputs/lora/` to demonstrate hyperparameter improvements |

## How to read this archive

For the narrative behind each iteration (what changed between versions and why), see `docs/iteration_log.md`. The files here are the raw evidence; the log is the explanation.
