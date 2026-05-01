import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

# Download resource yang dibutuhkan
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

BAD_USERS = {"AutoModerator", "[deleted]", "[removed]"}

CONTRACTIONS = {
    "ain't": "are not", "aren't": "are not", "can't": "cannot",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "i'd": "i would", "i'll": "i will",
    "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it's": "it is", "let's": "let us", "mightn't": "might not",
    "mustn't": "must not", "shan't": "shall not", "she'd": "she would",
    "she'll": "she will", "she's": "she is", "shouldn't": "should not",
    "that's": "that is", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have",
    "wasn't": "was not", "we'd": "we would", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what's": "what is",
    "won't": "will not", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "kinda": "kind of", "sorta": "sort of", "lemme": "let me",
    "gimme": "give me", "tryna": "trying to",
}

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Pertahankan negasi — penting untuk konteks hate speech
NEGATION_WORDS = {"no", "not", "nor", "never", "neither", "nobody", "nothing", "nowhere", "cannot"}
stop_words = stop_words - NEGATION_WORDS

def expand_contractions(text):
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)
    return text

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wordnet.ADJ
    elif treebank_tag.startswith('V'): return wordnet.VERB
    elif treebank_tag.startswith('N'): return wordnet.NOUN
    elif treebank_tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def lemmatize_text(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    return " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in pos_tags])

def remove_stopwords(text):
    return " ".join([t for t in text.split() if t not in stop_words])

def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = expand_contractions(text)          # expand "don't" → "do not" dll
    text = re.sub(r"http\S+", "", text)       # hapus URL
    text = re.sub(r"\bamp\b", "", text)       # hapus artefak HTML &amp;
    text = re.sub(r"\b\d{4,}\b", "", text)   # hapus angka panjang
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = lemmatize_text(text)               # lemmatisasi dengan POS-aware
    text = remove_stopwords(text)             # hapus stopwords (minus negasi)
    return text


def transform_data(input_path, output_path):
    df = pd.read_csv(input_path)
    print("Sebelum cleaning:", len(df))

    # ── Reddit-specific filters ───────────────────────────────────
    df = df[~df["user_id"].isin(BAD_USERS)]

    df["clean_tweet"] = df["text"].apply(clean_text)

    df = df[df["clean_tweet"].str.strip() != ""]
    df = df[df["clean_tweet"].str.len() > 5]   # filter teks terlalu pendek
    df = df.drop_duplicates(subset=["clean_tweet"])
    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)

    print("Transform done")
    print("Total data:", len(df))