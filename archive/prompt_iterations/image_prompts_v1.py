# Hermione base appearance (fixed core identity, reused everywhere)

BASE_APPEARANCE = (
    "portrait of a young teenage girl, messy bushy brown hair (a bit frizzy, not too neat), "
    "natural brown eyes, slightly serious academic vibe, "
    "wearing Hogwarts school robes with Gryffindor tie (red and gold stripes), "
    "white shirt underneath, maybe slightly wrinkled like she’s been studying too long, "
    "at Hogwarts castle interior, old stone walls, warm candle light, "
    "cinematic lighting, soft shadows, subtle depth of field, "
    "highly detailed skin texture, realistic hair strands, "
    "photorealistic, 4k, ultra detailed, film still feeling, not overly stylized"
)

# emotion → expression + body language (kinda loose mapping, not super strict)
# trying to include face + pose + small behavior details

EMOTION_TO_VISUAL = {
    "determined": (
        "confident expression, focused sharp eyes, chin slightly raised, "
        "eyebrows slightly tightened, holding a wand firmly in front, "
        "upright posture, like ready to act or argue"
    ),

    "worried": (
        "anxious expression, slight frown, biting lower lip unconsciously, "
        "eyes glancing sideways as if thinking too much, "
        "shoulders slightly tense, hands close to chest"
    ),

    "angry": (
        "frowning, furrowed brows, tense jaw, lips pressed together, "
        "eyes narrowed with frustration, "
        "body leaning slightly forward, gripping wand tighter than usual"
    ),

    "happy": (
        "bright genuine smile (not fake), sparkling eyes, "
        "cheeks slightly raised, relaxed face muscles, "
        "open posture, maybe holding a book casually"
    ),

    "sad": (
        "downcast eyes, unfocused gaze, slight frown, "
        "soft sorrowful expression, lips slightly parted, "
        "shoulders slumped a bit, low energy posture"
    ),

    "neutral": (
        "calm expression, neutral mouth, attentive eyes, "
        "reading a book, slightly absorbed, "
        "relaxed but not smiling, just focused"
    )
}

# negative prompt 
# mainly for SD artifacts + anatomy issues
NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, pixelated, "
    "distorted face, asymmetrical face, bad eyes, cross-eyed, "
    "extra fingers, missing fingers, deformed hands, bad anatomy, "
    "weird proportions, unnatural pose, "
    "ugly, overexposed, oversaturated, harsh lighting, "
    "cartoon, anime, illustration style, "
    "text, watermark, logo, signature"
)


def build_prompt(emotion):
    #Combine base appearance and emotion variation into a full SD prompt.
    
    emotion_desc = EMOTION_TO_VISUAL[emotion]
    full_prompt = f"{BASE_APPEARANCE}, {emotion_desc}"
    return full_prompt


#test

if __name__ == "__main__":
    for emotion in EMOTION_TO_VISUAL.keys():
        print(f"\n=== {emotion.upper()} ===")
        print(build_prompt(emotion))