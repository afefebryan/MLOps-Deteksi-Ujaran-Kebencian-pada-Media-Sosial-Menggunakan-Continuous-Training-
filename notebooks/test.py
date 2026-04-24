from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
df = pd.read_csv('hate_speech_clean.csv')

df = df[["text_clean", "label"]]

print(df["label"].value_counts())
df.head()

import re

def clean_text(text):
    text = text.lower()                                      # lowercase
    text = re.sub(r'@\w+', '', text)                        # hapus @mention
    text = re.sub(r'http\S+|www\S+', '', text)              # hapus URL
    text = re.sub(r'\bamp\b', '', text)                     # hapus 'amp' (sisa HTML &amp;)
    text = re.sub(r'\b\d{4,}\b', '', text)                  # hapus angka panjang (unicode artifacts)
    text = re.sub(r'[^\w\s]', '', text)                     # hapus punctuation (termasuk !!!)
    text = re.sub(r'\s+', ' ', text).strip()                # normalisasi spasi
    return text

df['clean_tweet'] = df['text_clean'].apply(clean_text)

df = df.drop_duplicates(subset=["clean_tweet"])

label_map = {0: "hate_speech", 1: "offensive", 2: "neither"}


X = df["clean_tweet"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import os

# ============================================================
# 1. CONFIG
# ============================================================
MODEL_NAME   = "Hate-speech-CNERG/dehatebert-mono-english"
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 4
DROPOUT      = 0.1
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR     = "./dehatebert-finetuned"

LABEL2ID = {"hate_speech": 0, "offensive": 1, "neither": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

print(f"Using device : {DEVICE}")
print(f"Num labels   : {NUM_LABELS}  → {ID2LABEL}")
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. DATASET CLASS
# ============================================================
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids"     : encoding["input_ids"].squeeze(0),       # (max_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_len,)
            "labels"        : torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ============================================================
# 3. MODEL — Custom Classifier on top of DehатeBERT backbone
# ============================================================
class HateSpeechClassifier(nn.Module):
    """
    Menggunakan DehатeBERT sebagai backbone (feature extractor),
    lalu menambahkan custom classification head sendiri.
    Keuntungan: bebas modifikasi arsitektur, tidak konflik num_labels.
    """

    def __init__(self, model_name, num_labels=3, dropout=0.1):
        super().__init__()
        self.bert        = AutoModel.from_pretrained(model_name)
        hidden_size      = self.bert.config.hidden_size  # 768 untuk BERT-base

        self.classifier  = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),   # 768 → 384
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)     # 384 → 3
        )

        # Inisialisasi bobot linear baru
        self._init_weights()

    def _init_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]   # ambil token [CLS]
        logits     = self.classifier(cls_output)
        return logits


# ============================================================
# 4. LOAD TOKENIZER & MODEL
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = HateSpeechClassifier(
    model_name=MODEL_NAME,
    num_labels=NUM_LABELS,
    dropout=DROPOUT
)
model.to(DEVICE)

# Print ringkasan parameter
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params     : {total_params:,}")
print(f"Trainable params : {trainable_params:,}")


# ============================================================
# 5. HANDLE CLASS IMBALANCE — Weighted CrossEntropyLoss
# ============================================================
# y_train harus berupa list/array integer (0, 1, 2)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=y_train
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

print(f"\nClass weights:")
for i, w in enumerate(class_weights):
    print(f"  {ID2LABEL[i]:12s} → {w:.4f}")


# ============================================================
# 6. DATALOADER
# ============================================================
train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, MAX_LEN)
test_dataset  = HateSpeechDataset(X_test,  y_test,  tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"\nTrain batches : {len(train_loader)}")
print(f"Test  batches : {len(test_loader)}")


# ============================================================
# 7. OPTIMIZER + SCHEDULER
# ============================================================
# Differential learning rate:
# backbone (bert)     → LR kecil agar pretrained weights tidak rusak
# classifier head     → LR lebih besar karena diinisialisasi random
optimizer = AdamW([
    {"params": model.bert.parameters(),        "lr": 2e-5, "weight_decay": 0.01},
    {"params": model.classifier.parameters(),  "lr": 1e-4, "weight_decay": 0.01},
])

total_steps   = len(train_loader) * EPOCHS
warmup_steps  = total_steps // 10   # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"\nTotal steps  : {total_steps}")
print(f"Warmup steps : {warmup_steps}")


# ============================================================
# 8. TRAINING & EVAL FUNCTIONS
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss   = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # gradient clipping
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (step + 1) % 100 == 0:
            print(f"  Step [{step+1}/{len(loader)}] | Loss: {loss.item():.4f}")

    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds  = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


# ============================================================
# 9. TRAINING LOOP
# ============================================================
best_macro_f1    = 0.0
best_model_state = None
history          = []

print("\n" + "="*60)
print("START TRAINING")
print("="*60)

for epoch in range(EPOCHS):
    print(f"\n📌 Epoch {epoch+1}/{EPOCHS}")
    print("-" * 40)

    train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE)
    preds, labels_true = eval_epoch(model, test_loader, DEVICE)

    report_dict = classification_report(
        labels_true, preds,
        target_names=list(LABEL2ID.keys()),
        output_dict=True
    )
    macro_f1 = report_dict["macro avg"]["f1-score"]
    hate_f1  = report_dict["hate_speech"]["f1-score"]

    history.append({
        "epoch"     : epoch + 1,
        "train_loss": train_loss,
        "macro_f1"  : macro_f1,
        "hate_f1"   : hate_f1,
    })

    print(f"\nTrain Loss : {train_loss:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}  |  hate_speech F1: {hate_f1:.4f}")
    print("\n" + classification_report(
        labels_true, preds,
        target_names=list(LABEL2ID.keys())
    ))

    # Simpan model terbaik berdasarkan macro F1
    if macro_f1 > best_macro_f1:
        best_macro_f1    = macro_f1
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  ✅ Best model updated! Macro F1: {best_macro_f1:.4f}")

print("\n" + "="*60)
print(f"TRAINING DONE | Best Macro F1: {best_macro_f1:.4f}")
print("="*60)


# ============================================================
# 10. SAVE MODEL
# ============================================================
model.load_state_dict(best_model_state)

# Simpan weights model
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))

# Simpan tokenizer
tokenizer.save_pretrained(SAVE_DIR)

# Simpan metadata (label map, config)
import json
metadata = {
    "model_name" : MODEL_NAME,
    "num_labels" : NUM_LABELS,
    "label2id"   : LABEL2ID,
    "id2label"   : ID2LABEL,
    "max_len"    : MAX_LEN,
    "dropout"    : DROPOUT,
    "best_macro_f1": best_macro_f1,
}
with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nModel saved to: {SAVE_DIR}/")
print(f"  ├── model.pt")
print(f"  ├── metadata.json")
print(f"  └── tokenizer files")

# Simpan training history
df_history = pd.DataFrame(history)
df_history.to_csv(os.path.join(SAVE_DIR, "training_history.csv"), index=False)
print(f"  └── training_history.csv")


# ============================================================
# 11. INFERENCE — Load & Predict
# ============================================================
def load_model(save_dir, device):
    """Load model yang sudah di-fine-tune."""
    with open(os.path.join(save_dir, "metadata.json")) as f:
        meta = json.load(f)

    tokenizer_loaded = AutoTokenizer.from_pretrained(save_dir)
    model_loaded     = HateSpeechClassifier(
        model_name=meta["model_name"],
        num_labels=meta["num_labels"],
        dropout=meta["dropout"]
    )
    model_loaded.load_state_dict(
        torch.load(os.path.join(save_dir, "model.pt"), map_location=device)
    )
    model_loaded.to(device)
    model_loaded.eval()
    return model_loaded, tokenizer_loaded, meta


def predict(text, model, tokenizer, meta, device, max_len=128):
    """Predict single teks."""
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs  = torch.softmax(logits, dim=1).squeeze(0)
        pred   = torch.argmax(probs).item()

    label = meta["id2label"][str(pred)]
    return {
        "label"      : label,
        "confidence" : round(probs[pred].item(), 4),
        "scores"     : {
            meta["id2label"][str(i)]: round(p.item(), 4)
            for i, p in enumerate(probs)
        }
    }


def moderate_comment(text, model, tokenizer, meta, device):
    """
    Wrapper untuk simulasi Reddit moderation.
    Returns action, label, confidence, explanation.
    """
    result = predict(text, model, tokenizer, meta, device)
    label  = result["label"]

    actions = {
        "hate_speech": {
            "action"     : "🚫 BLOCKED",
            "explanation": "Comment mengandung hate speech yang menargetkan kelompok tertentu."
        },
        "offensive": {
            "action"     : "⚠️ HIDDEN",
            "explanation": "Comment bersifat kasar/offensive dan disembunyikan dari publik."
        },
        "neither": {
            "action"     : "✅ VISIBLE",
            "explanation": "Comment aman dan dapat ditampilkan."
        },
    }

    return {
        "text"       : text,
        "action"     : actions[label]["action"],
        "label"      : label,
        "confidence" : f"{result['confidence']:.2%}",
        "explanation": actions[label]["explanation"],
        "scores"     : result["scores"],
    }


# ============================================================
# 12. TEST INFERENCE
# ============================================================
print("\n" + "="*60)
print("TEST INFERENCE")
print("="*60)

model_inf, tokenizer_inf, meta_inf = load_model(SAVE_DIR, DEVICE)

test_comments = [
    "I hate all people from that religion, they should be removed.",
    "You're such a stupid idiot, nobody asked for your opinion.",
    "Great post! Thanks for sharing this information.",
    "These immigrants are destroying our country.",
    "I disagree with your point but I respect your view.",
]

for comment in test_comments:
    result = moderate_comment(comment, model_inf, tokenizer_inf, meta_inf, DEVICE)
    print(f"\nText       : {result['text'][:70]}...")
    print(f"Action     : {result['action']}")
    print(f"Label      : {result['label']}  ({result['confidence']})")
    print(f"Explanation: {result['explanation']}")
    print(f"Scores     : {result['scores']}")