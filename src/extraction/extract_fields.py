"""
extract_fields.py
=================
LLM-based structured field extraction and evaluation pipeline (few-shot + fuzzy matching).

This module:
- Loads transcription results (hypotheses) from the ASR pipeline CSV.
- Sends each hypothesis to a local Ollama LLM (default: llama3.1:8b) using a
few-shot prompt to extract structured fields from the sales visit transcript:
    - deal_status: current stage of the deal.
    - sentiment: overall tone of the conversation.
    - action_items: list of next steps identified in the transcript.
    - follow_up_date: date of the next scheduled follow-up.
    - client_concern: primary concern raised by the client.
- Evaluates extracted fields against ground truth using field-level scoring:
    - Exact match for deal_status, sentiment, client_concern.
    - Fuzzy match (rapidfuzz) for follow_up_date and action_items.
    - Overall accuracy as the average of all field scores.
- Saves per-file results to results/extraction_results_few_shot_fuzz.csv.
- Prints a summary table of average overall accuracy per audio type
(clean, slight_noise, heavy_noise).
"""

import json
from pathlib import Path
from collections import defaultdict
import csv
import ollama
from rapidfuzz import fuzz

# Config
TRANSCRIPTION_RESULTS = Path("results/transcription_results.csv")
GROUND_TRUTH_PATH = Path("data/ground_truth/ground_truth.json")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "extraction_results_few_shot_fuzz_2.csv"

MODEL = "llama3.1:8b"

# PROMPT_TEMPLATE = """
# You are an AI assistant that extracts structured information from sales visit transcripts.

# Given the following sales visit transcript, extract these fields and return ONLY a valid JSON object with no explanation, no preamble, no markdown:

# {{
# "deal_status": one of [closed_won, closed_lost, in_progress, at_risk, upsell, renewal, new_lead, active_customer],
# "sentiment": one of [positive, negative, neutral, mixed],
# "action_items": list of strings describing next steps,
# "follow_up_date": string or null,
# "client_concern": one of [price, gdpr_compliance, technical_issue, missing_decision_maker, competitor, integration, reliability, budget, missing_features, commitment_risk, asr_quality, timing, trust_validation, budget_approval] or null
# }}

# Transcript:
# {transcript}

# Return ONLY the JSON object. No explanation. No markdown. No extra text.
# """

PROMPT_TEMPLATE = """
You are an AI assistant that extracts structured information from sales visit transcripts.

Given the following sales visit transcript, extract these fields and return ONLY a valid JSON object with no explanation, no preamble, no markdown backticks.

Fields to extract:
- "deal_status": one of [closed_won, closed_lost, in_progress, at_risk, upsell, renewal, new_lead, active_customer]
- "sentiment": one of [positive, negative, neutral, mixed]
- "action_items": list of concrete next steps. Always include specific details mentioned such as email addresses, dates, days, names, and amounts.
- "follow_up_date": the day, date, month or any specific part of the day mentioned for follow up, or null if none
- "client_concern": one of [pricing, gdpr_compliance, technical_issue, missing_decision_maker, competitor, integration, reliability, budget, missing_features, commitment_risk, asr_quality, timing, trust_validation, budget_approval] or null

---

EXAMPLE 1:
Transcript:
"So based on your team size, I'd recommend the professional plan at 400 euros per month. That's more than we expected, honestly. I understand. What budget range were you thinking? Somewhere around 250. Let me check if we can offer a quarterly discount. I'll come back to you by Wednesday."

Output:
{{
"deal_status": "in_progress",
"sentiment": "neutral",
"action_items": ["Check if quarterly discount is possible", "Follow up by Wednesday with discount proposal"],
"follow_up_date": "Wednesday",
"client_concern": "price"
}}

---

EXAMPLE 2:
Transcript:
"Thanks for making time. Have you had a chance to review the proposal? We have, and we've decided to go in a different direction. Can I ask what made the difference? Mostly internal budget cuts. Nothing to do with your product. I understand. Can I check back in next quarter? Sure, that's fine."

Output:
{{
"deal_status": "closed_lost",
"sentiment": "neutral",
"action_items": ["Follow up next quarter"],
"follow_up_date": "next quarter",
"client_concern": "budget"
}}

---

EXAMPLE 3:
Transcript:
"Good morning, I'm here to follow up on our proposal from last week. Yes, come in. We had a chance to review it. Great. Any questions on the pricing structure? Not really. We'd like to move forward with the starter package. Perfect. I'll send the contract by Thursday. We can onboard you by end of month. That works for us."

Output:
{{
"deal_status": "closed_won",
"sentiment": "positive",
"action_items": ["Send contract by Thursday", "Onboard client by end of month"],
"follow_up_date": "Thursday",
"client_concern": null
}}

---

Now extract from this transcript:
{transcript}

Return ONLY the JSON object. No explanation. No markdown. No extra text.
"""


def load_ground_truth(path):
    """Load ground truth JSON and return a dict keyed by script id."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def load_transcriptions(path):
    """Load transcription results CSV."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_fields(transcript):
    """Send transcript to Ollama and extract structured fields."""
    prompt = PROMPT_TEMPLATE.format(transcript=transcript)
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response["message"]["content"].strip()

    # Clean up response in case model adds markdown
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to fix incomplete JSON by closing it
        try:
            fixed = raw.strip()
            if not fixed.endswith("}"):
                fixed = fixed + "\n}"
            return json.loads(fixed)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON response:\n  {raw}")
            return None


def evaluate_extraction(predicted, ground_truth):
    """
    Compare predicted fields against ground truth.
    Exact match for fixed label fields.
    Fuzzy match for natural language fields.
    """
    scores = {}

    # deal_status — exact match
    scores["deal_status_correct"] = int(
        predicted.get("deal_status") == ground_truth.get("deal_status")
    )

    # sentiment — exact match
    scores["sentiment_correct"] = int(
        predicted.get("sentiment") == ground_truth.get("sentiment")
    )

    # client_concern — exact match
    scores["client_concern_correct"] = int(
        predicted.get("client_concern") == ground_truth.get("client_concern")
    )

# follow_up_date — fuzzy match (natural language varies)
    gt_date   = ground_truth.get("follow_up_date") or ""
    pred_date = predicted.get("follow_up_date") or ""
    if gt_date == "" and pred_date == "":
        scores["follow_up_date_correct"] = 1
    elif gt_date == "" or pred_date == "":
        scores["follow_up_date_correct"] = 0
    else:
        similarity = fuzz.partial_ratio(gt_date.lower(), pred_date.lower())
        scores["follow_up_date_correct"] = int(similarity >= 80)



    # action_items — fuzzy partial match (natural language varies)
    gt_actions   = ground_truth.get("action_items", [])
    pred_actions = predicted.get("action_items", [])
    if gt_actions:
        matched = sum(
            1 for gt_item in gt_actions
            if any(
                fuzz.partial_ratio(gt_item.lower(), pred.lower()) >= 70
                for pred in pred_actions
            )
        )
        scores["action_items_accuracy"] = round(matched / len(gt_actions), 4)
    else:
        scores["action_items_accuracy"] = 1.0

    # Overall score — average of all field scores
    scores["overall_accuracy"] = round(
        (scores["deal_status_correct"] +
        scores["sentiment_correct"] +
        scores["follow_up_date_correct"] +
        scores["client_concern_correct"] +
        scores["action_items_accuracy"]) / 5, 4
    )

    return scores


def print_summary(results):
    """Print average extraction accuracy per audio type."""
    accuracy_by_type = defaultdict(list)
    for r in results:
        accuracy_by_type[r["audio_type"]].append(float(r["overall_accuracy"]))

    print("\n========== Extraction Summary ==========")
    print(f"{'Audio Type':<20} {'Overall Accuracy':>18}")
    print("-" * 40)
    for audio_type in ["clean", "slight_noise", "heavy_noise"]:
        vals = accuracy_by_type[audio_type]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"{audio_type:<20} {avg:>17.2%}")
    print("=========================================")


def main():
    """
    Orchestrate the full extraction and evaluation pipeline.

    Steps:
        1. Load ground truth from JSON and transcriptions from CSV.
        2. For each transcription, send the hypothesis text to Ollama (llama3.1:8b)
        to extract structured fields (deal_status, sentiment, action_items,
        follow_up_date, client_concern).
        3. Evaluate extracted fields against ground truth using field-level scoring.
        4. Save all results (scores + predicted/ground truth JSON) to
        results/extraction_results.csv.
        5. Print a summary table of average overall accuracy per audio type.
    """
    ground_truth   = load_ground_truth(GROUND_TRUTH_PATH)
    transcriptions = load_transcriptions(TRANSCRIPTION_RESULTS)
    print(f"Loaded {len(transcriptions)} transcriptions")

    results    = []
    fieldnames = [
        "script_id", "audio_type", "audio_file",
        "deal_status_correct", "sentiment_correct",
        "follow_up_date_correct", "client_concern_correct",
        "action_items_accuracy", "overall_accuracy",
        "predicted", "ground_truth"
    ]

    for row in transcriptions:
        script_id  = row["script_id"]
        audio_type = row["audio_type"]
        hypothesis = row["hypothesis"]
        gt = ground_truth.get(script_id)

        if not gt:
            print(f"No ground truth for {script_id} — skipping")
            continue

        print(f"Extracting: {row['audio_file']}")
        predicted = extract_fields(hypothesis)
        # print(f"Predicted: {predicted}")
        # break

        if predicted is None:
            print(f"Skipping {script_id} due to parse error")
            continue

        scores = evaluate_extraction(predicted, gt)
        print(f"Overall accuracy: {scores['overall_accuracy']:.2%}")

        results.append({
            "script_id": script_id,
            "audio_type": audio_type,
            "audio_file": row["audio_file"],
            "deal_status_correct": scores["deal_status_correct"],
            "sentiment_correct": scores["sentiment_correct"],
            "follow_up_date_correct": scores["follow_up_date_correct"],
            "client_concern_correct": scores["client_concern_correct"],
            "action_items_accuracy": scores["action_items_accuracy"],
            "overall_accuracy": scores["overall_accuracy"],
            "predicted": json.dumps(predicted),
            "ground_truth": json.dumps({
                "deal_status": gt.get("deal_status"),
                "sentiment": gt.get("sentiment"),
                "follow_up_date": gt.get("follow_up_date"),
                "client_concern": gt.get("client_concern"),
                "action_items": gt.get("action_items"),
            })
        })

    # Save results
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {RESULTS_PATH}")

    print_summary(results)


if __name__ == "__main__":
    main()
