"""Field formatter for the TTS readback pipeline.

Loads extraction results CSV, parses predicted JSON per row, converts
structured fields into natural spoken-language summaries for ElevenLabs TTS,
handles null/missing fields, and saves to readback_formatted.csv.
"""

import csv
import json
from pathlib import Path

# Config
EXTRACTION_RESULTS = Path("results/extraction_results_few_shot_fuzz_2.csv")
RESULTS_DIR        = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FORMATTED_PATH     = RESULTS_DIR / "readback_formatted.csv"

# Deal status to human-readable spoken form
DEAL_STATUS_LABELS = {
    "closed_won":      "closed and won",
    "closed_lost":     "closed and lost",
    "in_progress":     "in progress",
    "at_risk":         "at risk",
    "upsell":          "an upsell opportunity",
    "renewal":         "up for renewal",
    "new_lead":        "a new lead",
    "active_customer": "an active customer",
}

# Client concern to human-readable spoken form
CONCERN_LABELS = {
    "pricing": "pricing",
    "price": "pricing",
    "gdpr_compliance": "GDPR compliance",
    "technical_issue": "a technical issue",
    "missing_decision_maker": "a missing decision maker",
    "competitor": "competition from another vendor",
    "integration": "system integration",
    "reliability": "reliability",
    "budget": "budget constraints",
    "missing_features": "missing product features",
    "commitment_risk": "commitment risk",
    "asr_quality": "ASR transcription quality",
    "timing":                 "timing",
    "trust_validation":       "trust and validation",
    "budget_approval":        "budget approval",
}


def load_extraction_results(path: Path) -> list[dict]:
    """Load the extraction results CSV and return a list of row dicts.

    Args:
        path: Path to the extraction results CSV.

    Returns:
        List of row dicts from the CSV.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows)} rows from {path}")
    return rows


def format_deal_status(value: str | None) -> str | None:
    """Convert a deal_status key to a spoken-language label.

    Args:
        value: Raw deal_status string or None.

    Returns:
        Human-readable label or None.
    """
    if not value:
        return None
    return DEAL_STATUS_LABELS.get(value, value.replace("_", " "))


def format_client_concern(value: str | None) -> str | None:
    """Convert a client_concern key to a spoken-language label.

    Args:
        value: Raw client_concern string or None.

    Returns:
        Human-readable label or None.
    """
    if not value:
        return None
    return CONCERN_LABELS.get(value, value.replace("_", " "))


def format_fields_to_speech(predicted: dict) -> str:
    """Convert a predicted fields dict into a natural spoken-language summary.

    Includes only present, non-null fields; action items are comma-separated.

    Args:
        predicted: Dict with deal_status, sentiment, follow_up_date,
            client_concern, action_items.

    Returns:
        A single string ready for ElevenLabs TTS.
    """
    sentences = []

    # Deal status
    deal_status = format_deal_status(predicted.get("deal_status"))
    if deal_status:
        sentences.append(f"Deal status is {deal_status}.")

    # Sentiment
    sentiment = predicted.get("sentiment")
    if sentiment:
        sentences.append(f"Overall sentiment is {sentiment}.")

    # Follow-up date
    follow_up = predicted.get("follow_up_date")
    if follow_up:
        sentences.append(f"Follow-up is scheduled for {follow_up}.")

    # Client concern
    concern = format_client_concern(predicted.get("client_concern"))
    if concern:
        sentences.append(f"The main client concern is {concern}.")

    # Action items
    action_items = predicted.get("action_items", [])
    if action_items:
        if len(action_items) == 1:
            sentences.append(f"Action item: {action_items[0]}.")
        else:
            items_text = ", ".join(action_items[:-1]) + f", and {action_items[-1]}"
            sentences.append(f"Action items include: {items_text}.")

    # Fallback if everything was null
    if not sentences:
        return "No structured information could be extracted from this visit."

    return " ".join(sentences)


def build_formatted_rows(rows: list[dict]) -> list[dict]:
    """Process all extraction result rows and build formatted summary rows.

    Args:
        rows: List of dicts from the extraction results CSV.

    Returns:
        List of dicts with script_id, audio_type, audio_file, formatted_summary.
    """
    formatted = []

    for row in rows:
        script_id  = row["script_id"]
        audio_type = row["audio_type"]
        audio_file = row["audio_file"]

        # Parse the predicted JSON string
        try:
            predicted = json.loads(row["predicted"])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse predicted JSON for {audio_file} — {e}")
            continue

        summary = format_fields_to_speech(predicted)

        formatted.append({
            "script_id": script_id,
            "audio_type": audio_type,
            "audio_file": audio_file,
            "formatted_summary": summary,
        })

    return formatted


def save_formatted_results(rows: list[dict], path: Path) -> None:
    """Save formatted summaries to CSV.

    Args:
        rows: List of dicts with script_id, audio_type, audio_file, formatted_summary.
        path: Output CSV path.
    """
    fieldnames = ["script_id", "audio_type", "audio_file", "formatted_summary"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFormatted summaries saved to {path}")


def print_preview(rows: list[dict], n: int = 5) -> None:
    """Print a preview of the first n formatted summaries.

    Args:
        rows: List of formatted row dicts.
        n: Number of rows to preview (default 5).
    """
    print(f"\n{'='*60}")
    print(f"FORMATTED SUMMARY PREVIEW (first {n} rows)")
    print(f"{'='*60}")
    for row in rows[:n]:
        print(f"\n[{row['audio_file']}]")
        print(f"  {row['formatted_summary']}")
    print(f"\n{'='*60}")


def main():
    """Orchestrate the full field formatting pipeline.

    Loads extraction results, formats each row to spoken summary, saves to
    readback_formatted.csv, and prints a preview of the first 5 rows.
    """
    rows = load_extraction_results(EXTRACTION_RESULTS)
    formatted = build_formatted_rows(rows)

    print(f"Successfully formatted {len(formatted)} summaries")

    save_formatted_results(formatted, FORMATTED_PATH)
    print_preview(formatted)


if __name__ == "__main__":
    main()
