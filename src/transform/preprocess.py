import pandas as pd
import re
from sentence_transformers import SentenceTransformer


BAD_USERS = {"AutoModerator", "[deleted]", "[removed]"}


def clean_text(text):

    if pd.isna(text):
        return ""

    text = text.lower()

    text = text.replace("\n", " ")

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def transform_data(input_path, output_path, embedding_model_name="all-MiniLM-L6-v2"):
    # === Load data ===
    df = pd.read_csv(input_path)
    print("Sebelum cleaning:", len(df))
    
    # === Clean text ===
    df["comment_clean"] = df["text"].apply(clean_text)
    
    # === Remove empty rows ===
    df = df[df["comment_clean"] != ""]
    
    # === Drop duplicates ===
    df = df.drop_duplicates(subset=["comment_clean"])
    
    # === Filter bad users ===
    df = df[~df["user_id"].isin(BAD_USERS)]
    
    df = df.reset_index(drop=True)
    
    # === Generate embeddings ===
    print("Generating embeddings ...")
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(df["comment_clean"].tolist(), batch_size=32, show_progress_bar=True)
    df["embedding"] = embeddings.tolist()
    
    # === Save CSV ===
    df.to_csv(output_path, index=False)
    
    print("Transform selesai")
    print("Jumlah data:", len(df))