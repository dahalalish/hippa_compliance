import os
import random
import re
import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HIPPA.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "training_data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "phi_ner")

LABELS = ["PERSON", "DATE", "SSN", "PHONE", "EMAIL", "ADDRESS", "GENDER"]

# --- Templates for synthetic PHI sentences ---
TEMPLATES = [
    "{name} was born on {dob}. SSN: {ssn}, phone: {phone}, email: {email}, address: {address}. Gender: {gender}.",
    "Patient {name}, DOB {dob}, reachable at {phone} or {email}, lives at {address}. Gender: {gender}.",
    "Record: {name}, {gender}, {dob}, {ssn}, {phone}, {email}, {address}.",
    "Contact info: {name}, {phone}, {email}, Address: {address}, DOB: {dob}, Gender: {gender}, SSN: {ssn}.",
    "Refer {name}, gender: {gender}, born {dob}, SSN {ssn}, email {email}.",
    "Admission record: {name}, {gender}, DOB {dob}, contact {phone}, address {address}.",
    "Medical report for {name} ({gender}), born {dob}, ID {ssn}, email {email}.",
    "Patient details → Name: {name}, DOB: {dob}, Gender: {gender}, SSN: {ssn}, Phone: {phone}, Email: {email}, Address: {address}.",
    "Emergency contact: {name} ({gender}), reachable via {phone}, SSN {ssn}.",
    "{name}, residing at {address}, DOB {dob}, contact: {phone}, email: {email}, gender {gender}."
]


def randomize_formats(row):
    """Randomize formats for DOB, phone, and SSN safely (handle NaN as '')."""
    dob = str(row.get("dob") or "")
    phone = str(row.get("phone") or "")
    ssn = str(row.get("ssn") or "")

    if dob and dob.lower() != "nan":
        dob_variants = [dob, dob.replace("-", "/"), dob.replace("/", "-")]
        dob = random.choice(dob_variants)

    if phone and phone.lower() != "nan":
        phone_variants = [
            phone,
            phone.replace("-", "."),
            phone.replace("-", " "),
            f"({phone[:3]}) {phone[4:7]}-{phone[8:]}" if len(phone) >= 12 else phone
        ]
        phone = random.choice(phone_variants)

    if ssn and ssn.lower() != "nan":
        ssn_variants = [ssn, ssn.replace("-", ""), ssn.replace("-", " ")]
        ssn = random.choice(ssn_variants)

    return dob, phone, ssn


def find_all_spans(text, substring):
    """Find all occurrences of substring in text."""
    return [(m.start(), m.end()) for m in re.finditer(re.escape(substring), text)]


def create_training_data(df, drop_prob=0.3):
    """Generate training examples with text + entity spans."""
    training_examples = []

    for _, row in df.iterrows():
        if not row.get("first_name") or not row.get("last_name"):
            continue

        name = f"{row['first_name']} {row['last_name']}"
        dob, phone, ssn = randomize_formats(row)

        email = str(row.get("email") or "")
        address = str(row.get("address") or "")
        gender = str(row.get("gender") or "")

        # Randomly drop fields to simulate missing data (negative examples)
        if random.random() < drop_prob: dob = ""
        if random.random() < drop_prob: phone = ""
        if random.random() < drop_prob: ssn = ""
        if random.random() < drop_prob: email = ""
        if random.random() < drop_prob: address = ""
        if random.random() < drop_prob: gender = ""

        template = random.choice(TEMPLATES)
        text = template.format(
            name=name, dob=dob, ssn=ssn, phone=phone, email=email,
            address=address, gender=gender
        )

        entities = []
        used_spans = set()

        for label, value in [
            ("PERSON", name),
            ("DATE", dob),
            ("SSN", ssn),
            ("PHONE", phone),
            ("EMAIL", email),
            ("ADDRESS", address),
            ("GENDER", gender)
        ]:
            if value and value.lower() != "nan":
                for start, end in find_all_spans(text, value):
                    if (start, end) not in used_spans:
                        entities.append((start, end, label))
                        used_spans.add((start, end))

        if entities:  # only keep valid samples
            training_examples.append((text, {"entities": entities}))

    return training_examples


def save_spacy_data(train_data, dev_size=0.2):
    """Save training and dev data as .spacy files."""
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for label in LABELS:
        ner.add_label(label)

    train_examples, dev_examples = train_test_split(
        train_data, test_size=dev_size, random_state=42
    )

    for split_name, examples in zip(["train.spacy", "dev.spacy"], [train_examples, dev_examples]):
        db = DocBin()
        for text, annot in examples:
            doc = nlp.make_doc(text)
            try:
                example = Example.from_dict(doc, annot)
                db.add(example.reference)
            except Exception as e:
                print(f"⚠️ Skipping bad example: {text}\nError: {e}")
                continue

        os.makedirs(TRAIN_DIR, exist_ok=True)
        db.to_disk(os.path.join(TRAIN_DIR, split_name))

    print(f"✅ Training data saved to {TRAIN_DIR}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    train_data = create_training_data(df)

    print("\n🔎 Preview of generated training data:")
    for text, ann in train_data[:5]:
        print(text)
        print(ann)
        print("-----")

    save_spacy_data(train_data)
