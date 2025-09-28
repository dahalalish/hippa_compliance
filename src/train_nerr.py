import os
import pandas as pd
import spacy
from spacy.tokens import DocBin

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HIPPA.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "phi_ner")

# --- Utility to create training examples ---
def create_training_data(df):
    training_examples = []

    for _, row in df.iterrows():
        text = " ".join(map(str, row.values))
        entities = []

        if isinstance(row.get("first_name"), str) and isinstance(row.get("last_name"), str):
            name = f"{row['first_name']} {row['last_name']}"
            start = text.find(name)
            if start != -1:
                entities.append((start, start+len(name), "PERSON"))

        if isinstance(row.get("dob"), str):
            dob = row["dob"]
            start = text.find(dob)
            if start != -1:
                entities.append((start, start+len(dob), "DATE"))

        if isinstance(row.get("ssn"), str):
            ssn = row["ssn"]
            start = text.find(ssn)
            if start != -1:
                entities.append((start, start+len(ssn), "SSN"))

        if isinstance(row.get("phone"), str):
            phone = row["phone"]
            start = text.find(phone)
            if start != -1:
                entities.append((start, start+len(phone), "PHONE"))

        if isinstance(row.get("email"), str):
            email = row["email"]
            start = text.find(email)
            if start != -1:
                entities.append((start, start+len(email), "EMAIL"))

        if isinstance(row.get("address"), str):
            addr = row["address"]
            start = text.find(addr)
            if start != -1:
                entities.append((start, start+len(addr), "ADDRESS"))

        training_examples.append((text, {"entities": entities}))
    return training_examples


def train_ner():
    df = pd.read_csv(DATA_PATH)
    train_data = create_training_data(df)

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    labels = ["PERSON", "DATE", "SSN", "PHONE", "EMAIL", "ADDRESS"]
    for label in labels:
        ner.add_label(label)

    # Convert training data into spaCy DocBin
    db = DocBin()
    for text, annot in train_data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            if span:
                ents.append(span)
        doc.ents = ents
        db.add(doc)

    # Save to disk for spaCy training CLI
    os.makedirs("training_data", exist_ok=True)
    db.to_disk("training_data/train.spacy")

    print("✅ Training data prepared → run training next")


if __name__ == "__main__":
    train_ner()
