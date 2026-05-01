from transformers import pipeline
import pandas as pd
from torch.utils.data import Dataset

# ── Binary candidate labels — lebih sesuai dengan skema harmful/neutral ──
CANDIDATE_LABELS = [
    "This text is neutral, positive, or contains no harmful content",
    "This text contains hate speech, offensive language, or insults toward a person or group"
]

LABEL_MAP = {
    "This text is neutral, positive, or contains no harmful content": 0,          # neutral
    "This text contains hate speech, offensive language, or insults toward a person or group": 1   # harmful
}

LABEL_NAMES = {0: "neutral", 1: "harmful"}


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    device=0
)

def run_labeling(df, text_col="clean_tweet", batch_size=32):  # ← sesuai nama kolom dari preprocessing
    texts = df[text_col].fillna("").tolist()

    valid_idx = [i for i, t in enumerate(texts) if t.strip()]
    valid_texts = [texts[i] for i in valid_idx]

    labels = [-1] * len(texts)           # ← int, bukan string "unknown"
    confidences = [0.0] * len(texts)

    dataset = TextDataset(valid_texts)

    results = []
    for out in classifier(
        dataset,
        candidate_labels=CANDIDATE_LABELS,
        multi_label=False,
        batch_size=batch_size
    ):
        results.append(out)

    for i, result in zip(valid_idx, results):
        top_label_full = result["labels"][0]
        top_score      = result["scores"][0]
        second_score   = result["scores"][1]

        if top_score - second_score < 0.15:
            labels[i] = -1                # uncertain → akan difilter
        else:
            labels[i] = LABEL_MAP[top_label_full]

        confidences[i] = round(top_score, 4)

    df["zsl_label"]      = labels
    df["zsl_confidence"] = confidences

    # ── Filter uncertain sebelum simpan ──────────────────────────
    before = len(df)
    df_clean = df[df["zsl_label"] != -1].copy()
    after    = len(df_clean)

    print(f"\nFiltered uncertain: {before - after} rows ({(before-after)/before*100:.1f}%)")
    print("\nLabel distribution:")
    print(df_clean["zsl_label"].map(LABEL_NAMES).value_counts())

    return df_clean

# if __name__ == "__main__":
#     df_test = pd.DataFrame({
#         "comment_clean": [
#             # --- Clear Neutral ---
#             "Have a nice day :)",
#             "This is spam, buy now!",
#             "I just woke up and had coffee",
#             "Check out this link for more info",

#             # --- Clear Offensive (personal attack, no group target) ---
#             "You are stupid and rude!",
#             "I hate this person!",
#             "What an idiot you are",
#             "You're a complete waste of time",

#             # --- Clear Hateful (group-targeted) ---
#             "All [group] people should be banned",
#             "Those people are subhuman trash",
#             "I hate every single one of them because of their religion",
#             "People like that don't deserve rights",

#             # --- Medium / Ambiguous ---
#             "This guy is the worst, I can't stand him",      
#             "These politicians are all corrupt idiots",        
#             "I don't like people from that country",         
#             "You're so annoying, just stop talking",         
#             "Some groups just can't follow rules",           
#             "He's such a clown, nobody likes him",          
#             "This content is trash and so are you",          
#             "They always cause trouble wherever they go",    
#         ]
#     })

#     df_test = run_labeling(df_test)

#     # Pretty print with category grouping
#     pd.set_option("display.max_colwidth", 50)
#     pd.set_option("display.width", 120)
#     print("\n=== RESULTS ===")
#     print(df_test[["comment_clean", "label", "confidence"]].to_string(index=False))

#     print("\n=== LABEL DISTRIBUTION ===")
#     print(df_test["label"].value_counts())

#     print("\n=== UNCERTAIN / LOW CONFIDENCE ===")
#     uncertain = df_test[df_test["label"] == "uncertain"]
#     print(uncertain[["comment_clean", "confidence"]] if not uncertain.empty else "None")