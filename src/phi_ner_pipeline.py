import spacy
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "phi_ner", "model-best")

def load_phi_model():
    return spacy.load(MODEL_DIR)

def detect_phi_entities(text):
    nlp = load_phi_model()
    doc = nlp(text)
    results = []
    for ent in doc.ents:
        results.append({"text": ent.text, "label": ent.label_})
    return results

if __name__ == "__main__":
    test_text = "John Doe born on 01/02/1990 lives at 123 Main St. SSN: 123-45-6789 Email: john@example.com Phone: 555-123-4567"
    entities = detect_phi_entities(test_text)
    print("🔎 Detected PHI Entities:")
    for e in entities:
        print(f"{e['label']}: {e['text']}")
