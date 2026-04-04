import pandas as pd
import re
from sentence_transformers import SentenceTransformer


BAD_USERS = {"AutoModerator", "[deleted]", "[removed]"}


def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()

    # normalize whitespace dulu
    text = re.sub(r"\s+", " ", text)

    # hapus URL
    text = re.sub(r"http\S+", "", text)

    # hapus simbol
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # normalize lagi setelah cleaning
    text = re.sub(r"\s+", " ", text).strip()

    return text


def transform_data(input_path, output_path, embedding_model_name="all-MiniLM-L6-v2"):
    df = pd.read_csv(input_path)
    print("Sebelum cleaning:", len(df))
    
    df["comment_clean"] = df["text"].apply(clean_text)
    
    df = df[df["comment_clean"] != ""]
    
    df = df.drop_duplicates(subset=["comment_clean"])
    
    # Filter bad users
    df = df[~df["user_id"].isin(BAD_USERS)]
    
    df = df.reset_index(drop=True)
    
    # Generate embeddings
    print("Generating embeddings ...")
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(df["comment_clean"].tolist(), batch_size=32, show_progress_bar=True)
    df["embedding"] = embeddings.tolist()
    
    # Save CSV
    df.to_csv(output_path, index=False)
    
    print("Transform Done")
    print("Total Data:", len(df))