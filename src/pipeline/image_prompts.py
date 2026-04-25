# Hermione SD prompt builder v5
# Changes from v4:
#   - Anchored core identity features (age, hair, skin) with Compel weights (++)
#   - This counterbalances the emotion weights so the character looks consistent
#   - Both emotion AND identity get attention, instead of emotion drowning out identity

# Base appearance with weighted identity anchors
# (white teenage girl)++ locks ethnicity to a specific look
# (about 14 years old)++ locks age range
# These three anchors keep the character recognizable across emotions
BASE_APPEARANCE = (
    "(white teenage girl, about 14 years old)++, "
    "(long bushy curly brown hair)++, "
    "(brown eyes, fair skin)++, "
    "wearing Hogwarts school robes with Gryffindor red and gold tie, "
    "white shirt underneath, "
    "warm interior at Hogwarts, soft cinematic lighting, "
    "highly detailed face, photorealistic, film still"
)

# Emotion descriptors - keep weights but slightly reduced from v4
EMOTION_TO_VISUAL = {
    "determined": (
        "(stern serious expression)++, firm closed mouth, "
        "intense focused gaze, eyebrows pulled together, "
        "chin slightly raised, confident posture"
    ),

    "worried": (
        "(visibly anxious face)++, deeply furrowed brow, "
        "biting lower lip, eyes wide with concern, "
        "shoulders tense"
    ),

    "angry": (
        "(angry face)++, deeply frowning, "
        "furrowed eyebrows, tightly pressed lips, "
        "narrowed glaring eyes, jaw clenched"
    ),

    "happy": (
        "(broad genuine smile)++, laughing joyfully, "
        "bright cheerful eyes, raised cheeks, "
        "open relaxed posture"
    ),

    "sad": (
        "(visibly sad face)++, tears in eyes, "
        "pouting frown, downcast eyes, "
        "shoulders slumped"
    ),

    "neutral": (
        "calm neutral expression, relaxed mouth, "
        "attentive eyes, reading a book, "
        "thoughtful absorbed look"
    ),
}

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted face, deformed face, "
    "asymmetric face, bad eyes, crossed eyes, ugly face, "
    "extra fingers, deformed hands, bad anatomy, "
    "smiling, happy, cheerful, "
    "dark skin, black skin, asian features, "  # IMPORTANT: lock ethnicity for consistency
    "old woman, child, "                       # lock age range
    "cartoon, anime, text, watermark"
)


def build_prompt(emotion):
    # Identity anchors come FIRST, then emotion, then rest of appearance
    # This ordering helps CLIP lock identity before adding emotional variation
    emotion_desc = EMOTION_TO_VISUAL[emotion]
    return f"{BASE_APPEARANCE}, {emotion_desc}"


def build_negative_prompt(emotion):
    """Emotion-aware negative prompt.
    
    For happy emotion, drop the anti-smile terms so the model can express joy freely.
    """
    if emotion == "happy":
        return (
            "blurry, low quality, distorted face, deformed face, "
            "asymmetric face, bad eyes, crossed eyes, ugly face, "
            "extra fingers, deformed hands, bad anatomy, "
            "dark skin, black skin, asian features, "
            "old woman, child, "
            "cartoon, anime, text, watermark"
        )
    return NEGATIVE_PROMPT


if __name__ == "__main__":
    for emotion in EMOTION_TO_VISUAL.keys():
        prompt = build_prompt(emotion)
        word_count = len(prompt.split())
        print(f"\n=== {emotion.upper()} ({word_count} words) ===")
        print(prompt)