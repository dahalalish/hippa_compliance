import os
import re
import spacy

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "phi_ner", "model-best")

# Load trained NER model
try:
    nlp = spacy.load(MODEL_DIR)
    print(f"✅ Loaded PHI NER model from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load PHI model: {e}")
    nlp = None


# ---------------- HELPER FILTER ----------------
def is_valid_entity(label: str, text: str) -> bool:
    """Rule-based validation for NER predictions"""
    tokens = text.split()
    clean_text = text.strip()

    if len(tokens) == 0 or len(clean_text) < 2:
        return False

    # PERSON filter
    if label == "PERSON":
        if len(tokens) > 4:  # too long for a name
            return False

        # Allow common titles (Dr., Mr., Ms., Mrs., Prof.)
        titles = {"mr", "ms", "mrs", "dr", "prof"}
        if tokens[0].lower().rstrip(".") in titles:
            return True

        # Require at least one capitalized word (name-like)
        if not any(t[0].isupper() for t in tokens if t.isalpha()):
            return False

        # Reject common section/medical keywords
        bad_keywords = {
            "diagnosis", "treatment", "history", "introduction", "evaluation",
            "angiography", "medication", "rehabilitation", "adjustment",
            "compliance", "adherence", "symptoms", "copyright", "reports",
            "information", "purpose", "revealed", "advise", "birth", "leave"
        }
        if any(word.lower() in bad_keywords for word in tokens):
            return False

    # GENDER filter
    if label == "GENDER":
        valid_genders = {"male", "female", "other", "unknown", "m", "f"}
        if clean_text.lower() not in valid_genders:
            return False

    # ADDRESS filter
    if label == "ADDRESS":
        if re.fullmatch(r"\d{9}", clean_text):  # looks like SSN
            return False
        if not any(c.isdigit() for c in clean_text):
            return False
        if not re.search(r"(Street|St\.|Avenue|Ave|Road|Rd|Blvd|Boulevard|Lane|Ln|Drive|Dr|Suite|Apt)",
                         clean_text, re.IGNORECASE):
            return False

    return True


# ---------------- DETECTION FUNCTIONS ----------------
def detect_phi(text):
    """NER-based PHI detection"""
    if nlp is None:
        return []
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text.strip(),
            "label": ent.label_,
            "score": 0.85  # pseudo-confidence
        })
    return entities


def detect_phi_regex(text):
    """Regex-based PHI detection (trusted for structured entities)"""
    patterns = {
        "DATE": r"\b(?:\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2})\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b",
        "PHONE": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    }
    results = {}
    for label, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            results[label] = [{"text": m, "label": label, "score": 1.0} for m in matches]
    return results


def detect_phi_clean(text, threshold=0.9):
    """Combined PHI detection with filtering + scores"""
    regex_entities = detect_phi_regex(text)  # high confidence
    ner_entities = detect_phi(text)  # weaker, needs filtering

    filtered = []
    for e in ner_entities:
        if e["score"] < threshold:
            continue
        if is_valid_entity(e["label"], e["text"]):
            filtered.append(e)

    grouped = {}
    # Add regex results first
    for label, values in regex_entities.items():
        grouped[label] = values

    # Add filtered NER results
    for e in filtered:
        grouped.setdefault(e["label"], []).append(
            {"text": e["text"], "score": e["score"]}
        )

    return grouped
