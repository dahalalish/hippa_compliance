import os
import random
import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from faker import Faker
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HIPPA.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "training_data")

LABELS = ["PERSON", "DATE", "SSN", "PHONE", "EMAIL", "ADDRESS", "GENDER"]

fake = Faker()
nlp = spacy.blank("en")

# Load embeddings for PERSON filtering
embedder = SentenceTransformer("all-MiniLM-L6-v2")
REFERENCE_NAMES = ["John Smith", "Emily Johnson", "Dr. Alan Green", "Maria Lopez", "Sarah Thompson"]
REFERENCE_NAME_VECS = embedder.encode(REFERENCE_NAMES, convert_to_tensor=True)

# -------------------
# Helpers
# -------------------
def is_name_like(candidate: str, threshold: float = 0.6) -> bool:
    cand_vec = embedder.encode(candidate, convert_to_tensor=True)
    sim = float(util.cos_sim(cand_vec, REFERENCE_NAME_VECS).max())
    return sim >= threshold


def randomize_formats(dob, phone, ssn):
    """Randomize DOB, phone, SSN formats for diversity."""
    if dob and str(dob).lower() != "nan":
        dob_variants = [dob, dob.replace("-", "/"), dob.replace("/", "-")]
        dob = random.choice(dob_variants)

    if phone and "-" in phone and str(phone).lower() != "nan":
        phone_variants = [phone, phone.replace("-", "."), phone.replace("-", " ")]
        phone = random.choice(phone_variants)

    if ssn and "-" in ssn and str(ssn).lower() != "nan":
        ssn_variants = [ssn, ssn.replace("-", ""), ssn.replace("-", " ")]
        ssn = random.choice(ssn_variants)

    return dob, phone, ssn

# -------------------
# Real Data (HIPPA.csv)
# -------------------
TEMPLATES = [
    "{name} was born on {dob}. SSN: {ssn}, phone: {phone}, email: {email}, address: {address}. Gender: {gender}.",
    "Patient {name}, DOB {dob}, reachable at {phone} or {email}, lives at {address}. Gender: {gender}.",
    "Record: {name}, {gender}, {dob}, {ssn}, {phone}, {email}, {address}.",
    "Contact info: {name}, {phone}, {email}, Address: {address}, DOB: {dob}, Gender: {gender}, SSN: {ssn}.",
]

def create_training_data_from_csv(df):
    training_examples = []

    for _, row in df.iterrows():
        if not row.get("first_name") or not row.get("last_name"):
            continue

        name = f"{row['first_name']} {row['last_name']}"
        dob, phone, ssn = randomize_formats(str(row.get("dob") or ""),
                                            str(row.get("phone") or ""),
                                            str(row.get("ssn") or ""))
        email = str(row.get("email") or "")
        address = str(row.get("address") or "")
        gender = str(row.get("gender") or "")

        template = random.choice(TEMPLATES)
        text = template.format(
            name=name, dob=dob, ssn=ssn, phone=phone, email=email,
            address=address, gender=gender
        )

        entities = []
        for label, value in [
            ("PERSON", name),
            ("DATE", dob),
            ("SSN", ssn),
            ("PHONE", phone),
            ("EMAIL", email),
            ("ADDRESS", address),
            ("GENDER", gender),
        ]:
            if value and value.lower() != "nan":
                start = text.find(value)
                if start != -1:
                    end = start + len(value)
                    entities.append((start, end, label))

        training_examples.append((text, {"entities": entities}))
    return training_examples

# -------------------
# Synthetic Faker Data
# -------------------
FAKE_TEMPLATES = [
    "Patient {PERSON} visited the clinic on {DATE}.",
    "Doctor {PERSON} prescribed medication on {DATE}.",
    "{PERSON} can be contacted at {PHONE} or {EMAIL}.",
    "The SSN of {PERSON} is {SSN}.",
    "{PERSON} lives at {ADDRESS}. Gender: {GENDER}.",
]

def create_synthetic_example():
    template = random.choice(FAKE_TEMPLATES)
    annotations = []
    filled = template

    if "{PERSON}" in filled:
        name = None
        while not name:
            candidate = fake.name()
            if is_name_like(candidate):
                name = candidate
        start = filled.index("{PERSON}")
        filled = filled.replace("{PERSON}", name, 1)
        annotations.append((start, start + len(name), "PERSON"))

    if "{DATE}" in filled:
        date = fake.date(pattern="%m/%d/%Y")
        start = filled.index("{DATE}")
        filled = filled.replace("{DATE}", date, 1)
        annotations.append((start, start + len(date), "DATE"))

    if "{SSN}" in filled:
        ssn = fake.ssn()
        start = filled.index("{SSN}")
        filled = filled.replace("{SSN}", ssn, 1)
        annotations.append((start, start + len(ssn), "SSN"))

    if "{PHONE}" in filled:
        phone = fake.phone_number()
        start = filled.index("{PHONE}")
        filled = filled.replace("{PHONE}", phone, 1)
        annotations.append((start, start + len(phone), "PHONE"))

    if "{EMAIL}" in filled:
        email = fake.email()
        start = filled.index("{EMAIL}")
        filled = filled.replace("{EMAIL}", email, 1)
        annotations.append((start, start + len(email), "EMAIL"))

    if "{ADDRESS}" in filled:
        addr = fake.address().replace("\n", ", ")
        start = filled.index("{ADDRESS}")
        filled = filled.replace("{ADDRESS}", addr, 1)
        annotations.append((start, start + len(addr), "ADDRESS"))

    if "{GENDER}" in filled:
        gender = random.choice(["Male", "Female"])
        start = filled.index("{GENDER}")
        filled = filled.replace("{GENDER}", gender, 1)
        annotations.append((start, start + len(gender), "GENDER"))

    return filled, {"entities": annotations}

def create_synthetic_training_data(n=300):
    return [create_synthetic_example() for _ in range(n)]

# -------------------
# Save SpaCy Data
# -------------------
def save_spacy_data(train_data, dev_size=0.2):
    train_examples, dev_examples = train_test_split(train_data, test_size=dev_size, random_state=42)

    for split_name, examples in zip(["train.spacy", "dev.spacy"], [train_examples, dev_examples]):
        db = DocBin()
        for text, ann in examples:
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in ann["entities"]:
                span = doc.char_span(start, end, label=label)
                if span:
                    ents.append(span)
            doc.ents = ents
            db.add(doc)
        os.makedirs(TRAIN_DIR, exist_ok=True)
        db.to_disk(os.path.join(TRAIN_DIR, split_name))
    print(f"✅ Training data saved to {TRAIN_DIR}")

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    csv_data = create_training_data_from_csv(df)
    synthetic_data = create_synthetic_training_data(300)

    combined_data = csv_data + synthetic_data
    print(f"🔎 Generated {len(combined_data)} total examples")

    save_spacy_data(combined_data)
