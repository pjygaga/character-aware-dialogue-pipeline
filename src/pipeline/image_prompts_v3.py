# Hermione SD prompt builder v4
# Changes from v3:
#   - Emotion descriptions use direct, unambiguous verbs SD recognizes well
#     (e.g. "smiling broadly" instead of "slight smile")
#   - Compel weight syntax (key++ or (phrase)++) emphasizes the emotion words
#     so SD allocates more attention to the facial expression

# Base appearance, kept rich since Compel handles long prompts now
BASE_APPEARANCE = (
    "young teenage girl, bushy curly brown hair, brown eyes, "
    "wearing Hogwarts school robes with Gryffindor red and gold tie, "
    "white shirt underneath, "
    "warm interior at Hogwarts, soft cinematic lighting, "
    "highly detailed face, photorealistic, film still"
)

# Emotion descriptors with Compel weight emphasis
# (phrase)++ multiplies attention weight on that phrase by ~1.21
# (phrase)+++ multiplies by ~1.33 (use sparingly, can distort)
EMOTION_TO_VISUAL = {
    "determined": (
        "(stern serious expression)++, (firm closed mouth)++, "
        "(intense focused gaze)++, eyebrows pulled together, "
        "chin slightly raised, confident posture"
    ),

    "worried": (
        "(visibly anxious face)++, (deeply furrowed brow)++, "
        "(biting lower lip)++, eyes wide with concern, "
        "shoulders tense, hands clutched near chest"
    ),

    "angry": (
        "(angry face)+++, (deeply frowning)++, "
        "(furrowed eyebrows)++, (tightly pressed lips)++, "
        "narrowed glaring eyes, jaw clenched, leaning forward aggressively"
    ),

    "happy": (
        "(broad genuine smile)+++, (laughing joyfully)++, "
        "(bright cheerful eyes)++, raised cheeks, "
        "open relaxed posture, warm expression"
    ),

    "sad": (
        "(visibly sad face)+++, (crying with tears in eyes)++, "
        "(pouting frown)++, (downcast eyes)++, "
        "shoulders slumped low, fragile expression"
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
    "smiling, happy, cheerful, "  # IMPORTANT: discourage default smile bias
    "cartoon, anime, text, watermark"
)


def build_prompt(emotion):
    # Emotion goes first (CLIP weights early tokens more heavily)
    emotion_desc = EMOTION_TO_VISUAL[emotion]
    return f"{emotion_desc}, {BASE_APPEARANCE}"


# When emotion IS happy, drop the anti-smile negative prompt
def build_negative_prompt(emotion):
    """Build emotion-aware negative prompt.
    
    Default SD has strong bias toward smiling faces. We negate smile-related words
    EXCEPT when the target emotion actually is happy.
    """
    if emotion == "happy":
        # Don't fight the smile when we want one
        return (
            "blurry, low quality, distorted face, deformed face, "
            "asymmetric face, bad eyes, crossed eyes, ugly face, "
            "extra fingers, deformed hands, bad anatomy, "
            "cartoon, anime, text, watermark"
        )
    return NEGATIVE_PROMPT


if __name__ == "__main__":
    for emotion in EMOTION_TO_VISUAL.keys():
        prompt = build_prompt(emotion)
        word_count = len(prompt.split())
        print(f"\n=== {emotion.upper()} ({word_count} words) ===")
        print(prompt)