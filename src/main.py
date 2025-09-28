import os
import json
from pathlib import Path
from phi_detection import detect_phi_clean
from ocr_engine import extract_text_from_pdf

BASE_DIR = Path(__file__).parent.parent
SAMPLE_DOCS_DIR = BASE_DIR / "data" / "sampledocs"
RESULTS_FILE = BASE_DIR / "phi_results.json"


def clean_text(text):
    import re
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"^[A-Z ]{3,50}\n", "", text, flags=re.MULTILINE)
    return text.strip()


def analyze_document(doc_path):
    print(f"\nINFO: Analyzing: {doc_path.name}")
    text = extract_text_from_pdf(doc_path)
    text = clean_text(text)

    grouped_entities = detect_phi_clean(text)

    if grouped_entities:
        print(f"WARNING: ⚠️ Violations found in {doc_path.name}")
        for label, ents in grouped_entities.items():
            print(f"  {label}:")
            for e in ents:
                print(f"    - {e['text']} (score={e['score']:.2f})")
    else:
        print(f"INFO: ✅ No PHI violations detected in {doc_path.name}")

    return {doc_path.name: grouped_entities}


def main():
    if not SAMPLE_DOCS_DIR.exists():
        print(f"ERROR: Sample documents directory does not exist: {SAMPLE_DOCS_DIR}")
        return

    results = []
    for file in SAMPLE_DOCS_DIR.iterdir():
        if file.suffix.lower() == ".pdf":
            results.append(analyze_document(file))

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nINFO: 📄 Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
