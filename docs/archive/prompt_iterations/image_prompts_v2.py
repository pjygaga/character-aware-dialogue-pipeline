# Hermione SD prompt builder, redesigned to fit within CLIP's 77-token limit.
# Key change: emotion description goes FIRST (CLIP weights early tokens more heavily),
# and the appearance description is trimmed to the essentials.

# Compact base appearance (about 25 tokens)
BASE_APPEARANCE = (
    "close-up portrait, head and shoulders, "
    "young teenage girl, bushy brown hair, brown eyes, "
    "Hogwarts uniform, Gryffindor red gold tie, "
    "soft studio lighting, photorealistic, detailed face"
)

# Compact emotion descriptors (each about 15-20 tokens)
# Emotion goes first so CLIP definitely sees it
EMOTION_TO_VISUAL = {
    "determined": "confident face, sharp focused eyes, chin raised, eyebrows tightened",
    "worried":    "anxious face, slight frown, biting lower lip, eyes glancing aside",
    "angry":      "frowning face, furrowed brows, tense jaw, narrowed eyes",
    "happy":      "bright genuine smile, sparkling eyes, raised cheeks, relaxed face",
    "sad":        "downcast eyes, slight frown, soft sorrowful look, lips parted",
    "neutral":    "calm face, neutral mouth, attentive eyes, focused look",
}

# Trimmed negative prompt (still effective, fewer tokens)
NEGATIVE_PROMPT = (
    "blurry, low quality, distorted face, deformed face, "
    "asymmetric face, bad eyes, crossed eyes, ugly face, "
    "extra fingers, deformed hands, bad anatomy, "
    "cartoon, anime, text, watermark"
)


def build_prompt(emotion):
    # Emotion first, then appearance.
    # CLIP processes left-to-right and weights early tokens more.
    emotion_desc = EMOTION_TO_VISUAL[emotion]
    return f"{emotion_desc}, {BASE_APPEARANCE}"


# Token count check (rough word count as proxy)
if __name__ == "__main__":
    for emotion in EMOTION_TO_VISUAL.keys():
        prompt = build_prompt(emotion)
        word_count = len(prompt.split())
        print(f"\n=== {emotion.upper()} ({word_count} words) ===")
        print(prompt)