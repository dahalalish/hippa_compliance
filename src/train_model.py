import sys, os, pickle, random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HIPPA.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "phi_model.pkl")


def load_data(path=DATA_PATH):
    """Load HIPPA.csv and add synthetic NON-PHI data for training."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)

    # PHI class = 1
    df["text"] = df.astype(str).agg(" ".join, axis=1)
    df["label"] = 1

    # Generate synthetic NON-PHI examples
    non_phi_samples = [
        "The weather is nice today.",
        "This is a company policy document.",
        "Quarterly revenue increased by 12 percent.",
        "Meeting agenda for next week.",
        "Project roadmap and strategy notes.",
        "Marketing plan draft version 2.",
        "Generic placeholder text for testing.",
    ]
    df_nonphi = pd.DataFrame({"text": non_phi_samples, "label": [0]*len(non_phi_samples)})

    # Combine PHI + NON-PHI
    df_all = pd.concat([df[["text","label"]], df_nonphi], ignore_index=True)
    return train_test_split(df_all["text"], df_all["label"], test_size=0.2, random_state=42)


def train_phi_model():
    X_train, X_test, y_train, y_test = load_data()

    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    best_acc, best_model, best_iter = 0, None, 0
    print("🔄 Starting training...")

    for epoch in [50, 100, 200, 300]:
        model = LogisticRegression(max_iter=epoch, solver="lbfgs")
        model.fit(X_train_tfidf, y_train)
        acc = model.score(X_test_tfidf, y_test)
        print(f"Epochs={epoch:>3} | Accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc, best_model, best_iter = acc, model, epoch

    print("\n✅ Best accuracy: {:.4f} at {} epochs".format(best_acc, best_iter))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((vectorizer, best_model), f)
    print(f"📦 Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    train_phi_model()
