import spacy
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "phi_ner")  # path to your trained model

# Load your pretrained PHI NER model
nlp = spacy.load(MODEL_DIR)

# Detect PHI using the trained model
def detect_phi(text):
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return entities

# Regex-based detection for structured PHI
import re

def detect_phi_regex(text):
    regex_patterns = {
        "DATE": r"\b\d{2}/\d{2}/\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "PHONE": r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b",
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "ADDRESS": r"\b\d{1,5}\s\w+(\s\w+){0,4}\b",  # basic street pattern
    }

    violations = {}
    for label, pattern in regex_patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            violations[label] = matches
    return violations
