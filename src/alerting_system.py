import json
from pathlib import Path

LOG_FILE = Path(__file__).parent.parent / "logs" / "violations.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log_violation(doc_name, report):
    """
    Logs PHI violations to a JSON lines file.
    Ensures all data is JSON serializable.
    """
    # Convert any bool or non-serializable items to strings if needed
    safe_report = {}
    for key, value in report.items():
        if isinstance(value, (bool, int, float, str, dict, list)):
            safe_report[key] = value
        else:
            safe_report[key] = str(value)

    # Include document name
    safe_report['document'] = doc_name

    # Append to log file
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe_report, ensure_ascii=False) + "\n")
