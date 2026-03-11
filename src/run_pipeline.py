from extract.reddit_extractor import run_extract
from transform.transform import transform_data
from load.loader import load_dataset


RAW_PATH = "../data/raw/reddit_raw_comments.csv"
CLEAN_PATH = "../data/processed/reddit_clean_comments.csv"


print("===== EXTRACT =====")
run_extract(RAW_PATH)

print("===== TRANSFORM =====")
transform_data(RAW_PATH, CLEAN_PATH)

print("===== LOAD =====")
df = load_dataset(CLEAN_PATH)