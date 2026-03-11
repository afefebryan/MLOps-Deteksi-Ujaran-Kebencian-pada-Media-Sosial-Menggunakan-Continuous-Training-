import pandas as pd
import re


def clean_text(text):

    if pd.isna(text):
        return ""

    text = text.lower()

    text = text.replace("\n", " ")

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def transform_data(input_path, output_path):

    df = pd.read_csv(input_path)

    df["comment_clean"] = df["text"].apply(clean_text)

    df = df[df["comment_clean"] != ""]

    df = df.drop_duplicates(subset=["comment_clean"])

    df = df[
        (df["user_id"] != "AutoModerator") &
        (df["user_id"] != "[deleted]") &
        (df["user_id"] != "[removed]")
    ]

    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)

    print("Transform selesai")
    print("Jumlah data:", len(df))