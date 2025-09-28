import os
from pathlib import Path
from ocr_engine import extract_text_from_pdf
from phi_detection import detect_phi, detect_phi_regex
from alerting_system import log_violation

SAMPLE_DOCS_DIR = Path(__file__).parent.parent / "data" / "sampledocs"

def clean_text(text):
    import re
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'^[A-Z ]{3,50}\n', '', text, flags=re.MULTILINE)
    return text.strip()

def analyze_document(doc_path):
    print(f"\nAnalyzing: {doc_path.name}")
    text = extract_text_from_pdf(doc_path)
    text = clean_text(text)

    ner_entities = detect_phi(text)
    regex_violations = detect_phi_regex(text)

    # Group entities
    VALID_LABELS = {"PERSON", "DATE", "ADDRESS", "PHONE", "EMAIL", "SSN"}
    filtered_entities = [e for e in ner_entities if e["label"] in VALID_LABELS and len(e["text"].strip()) > 2]
    unique_entities = {(e["text"], e["label"]) for e in filtered_entities}
    grouped_entities = {}
    for text_val, label in unique_entities:
        grouped_entities.setdefault(label, []).append(text_val)

    report = {
        "binary_phi": bool(grouped_entities or regex_violations),
        "ner_entities": filtered_entities,
        "regex_violations": regex_violations,
        "grouped_entities": grouped_entities
    }

    log_violation(doc_path.name, report)

    print(f"⚠️ Violations found in {doc_path.name}:")
    print(f"Grouped PHI Entities:\n{grouped_entities}")
    print(f"Regex Violations:\n{regex_violations}")

def main():
    if not SAMPLE_DOCS_DIR.exists():
        print(f"Sample documents directory does not exist: {SAMPLE_DOCS_DIR}")
        return

    for file in SAMPLE_DOCS_DIR.iterdir():
        if file.suffix.lower() == ".pdf":
            analyze_document(file)

if __name__ == "__main__":
    main()
