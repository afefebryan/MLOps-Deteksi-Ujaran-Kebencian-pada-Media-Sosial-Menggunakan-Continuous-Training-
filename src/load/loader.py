import pandas as pd


def load_dataset(path):

    df = pd.read_csv(path)

    print("Dataset berhasil dimuat")

    print(df.head())

    print("Total data:", len(df))

    return df