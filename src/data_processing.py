import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="../data/HIPPA.csv"):
    df = pd.read_csv(path)
    # Combine PHI fields into one text column
    df["text"] = df.astype(str).agg(" ".join, axis=1)
    return df

def split_data(df):
    X = df["text"]
    y = [1]*len(df)  # All contain PHI (supervised)
    return train_test_split(X, y, test_size=0.2, random_state=42)
