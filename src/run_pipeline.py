from extract.ingest_data import run_extract
from transform.preprocess import transform_data
from load.loader import load_dataset
from labeling.labeling import run_labeling

from datetime import datetime
import os

# === BASE DIR (IMPORTANT FIX) ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === CONFIG ===
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "processed")

# ensure directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

BASE_RAW_NAME = "reddit_raw_comments"
BASE_CLEAN_NAME = "reddit_clean_comments"

# === GENERATE VERSION + DATE ===
date_str = datetime.now().strftime("%Y-%m-%d")

def get_versioned_filename(directory, base_name, date_str):
    version = 1
    
    while True:
        filename = f"{base_name}_V{version}_{date_str}.csv"
        full_path = os.path.join(directory, filename)
        
        if not os.path.exists(full_path):
            return full_path
        
        version += 1

# === CREATE FILE PATHS ===
RAW_PATH = get_versioned_filename(RAW_DIR, BASE_RAW_NAME, date_str)
CLEAN_PATH = get_versioned_filename(CLEAN_DIR, BASE_CLEAN_NAME, date_str)

# print("RAW PATH:", RAW_PATH)
# print("CLEAN PATH:", CLEAN_PATH)

# === PIPELINE ===
print("===== EXTRACT =====")
run_extract(RAW_PATH)

print("===== TRANSFORM =====")
transform_data(RAW_PATH, CLEAN_PATH)

print("===== LOAD =====")
df = load_dataset(CLEAN_PATH)

print("===== LABELING =====")
df = run_labeling(df)
df.to_csv(CLEAN_PATH, index=False)


print(f"\nSaved RAW  -> {RAW_PATH}")
print(f"Saved CLEAN -> {CLEAN_PATH}")
