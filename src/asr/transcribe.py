"""
Batch ASR transcription and evaluation pipeline using OpenAI Whisper.

This module:
- Loads clean and noisy audio files from the data directory.
- Transcribes each file using the Whisper model (default: medium) on MPS/CPU.
- Evaluates transcriptions against ground truth using two metrics:
    - Word Error Rate (WER): measures word-level transcription accuracy.
    - Keyword Accuracy: measures how many expected keywords appear in the transcript.
- Saves per-file results to a CSV file.
- Prints a summary table of average WER and keyword accuracy per audio type
(clean, slight_noise, heavy_noise).
"""

import json
from pathlib import Path
from collections import defaultdict
import csv
import torch
import whisper
from jiwer import wer

# Config
AUDIO_CLEAN_DIR = Path("data/audio_clean")
AUDIO_NOISY_DIR = Path("data/audio_noisy")
GROUND_TRUTH_PATH = Path("data/ground_truth/ground_truth.json")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "transcription_results.csv"

# Use MPS for M4 Pro Mac
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_SIZE = "medium"


def load_ground_truth(path):
    """Load ground truth JSON and return a dict keyed by script id."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def compute_wer(reference, hypothesis):
    """Compute Word Error Rate using jiwer library."""
    return round(wer(reference.lower(), hypothesis.lower()), 4)


def compute_keyword_accuracy(keywords, hypothesis):
    """
    Compute keyword accuracy:
    how many of the expected keywords appear in the hypothesis transcript.
    Score = matched keywords / total keywords
    """
    if not keywords:
        return None
    hyp_lower = hypothesis.lower()
    matched = sum(1 for kw in keywords if kw.lower() in hyp_lower)
    return round(matched / len(keywords), 4)


def get_script_id_from_filename(filename):
    """Extract script_id from audio filename. e.g. script_01_clean -> script_01"""
    stem = Path(filename).stem
    parts = stem.split("_")
    return f"{parts[0]}_{parts[1]}"


def get_audio_type(filename):
    """Determine audio type from filename."""
    if "slight_noise" in filename:
        return "slight_noise"
    elif "heavy_noise" in filename:
        return "heavy_noise"
    return "clean"


def transcribe_all(model, audio_files, ground_truth):
    """Transcribe all audio files and compute both metrics for each."""
    results = []

    for audio_path in sorted(audio_files):
        script_id = get_script_id_from_filename(audio_path.name)
        gt = ground_truth.get(script_id)

        if not gt:
            print(f"No ground truth found for {script_id} — skipping")
            continue

        print(f"Transcribing: {audio_path.name}")
        result = model.transcribe(str(audio_path), fp16=False)
        hypothesis = result["text"].strip()

        # Metric 1 — WER against full transcript
        reference = gt["full_transcript"]
        wer_score = compute_wer(reference, hypothesis)

        # Metric 2 — Keyword accuracy
        keywords = gt.get("keywords", [])
        keyword_accuracy = compute_keyword_accuracy(keywords, hypothesis)

        audio_type = get_audio_type(audio_path.name)

        results.append({
            "script_id": script_id,
            "audio_type": audio_type,
            "audio_file": audio_path.name,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": wer_score,
            "keyword_accuracy": keyword_accuracy,
        })

        print(f"WER: {wer_score:.2%} | Keyword Accuracy: {keyword_accuracy:.2%} | Type: {audio_type}")

    return results


def save_results(results, path):
    """Save transcription results to CSV."""
    fieldnames = [
        "script_id", "audio_type", "audio_file",
        "reference", "hypothesis", "wer", "keyword_accuracy"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {path}")


def print_summary(results):
    """Print average WER and keyword accuracy per audio type."""
    wer_by_type = defaultdict(list)
    keyword_by_type = defaultdict(list)

    for r in results:
        wer_by_type[r["audio_type"]].append(r["wer"])
        if r["keyword_accuracy"] is not None:
            keyword_by_type[r["audio_type"]].append(r["keyword_accuracy"])

    print("\n========== Evaluation Summary ==========")
    print(f"{'Audio Type':<20} {'Avg WER':>10} {'Keyword Acc':>15}")
    print("-" * 47)
    for audio_type in ["clean", "slight_noise", "heavy_noise"]:
        avg_wer = sum(wer_by_type[audio_type]) / len(wer_by_type[audio_type])
        avg_keyword = sum(keyword_by_type[audio_type]) / len(keyword_by_type[audio_type])
        print(f"{audio_type:<20} {avg_wer:>9.2%} {avg_keyword:>14.2%}")
    print("=========================================")


def main():
    """Main function to run the transcription and evaluation pipeline."""
    print(f"Using device : {DEVICE}")
    print(f"Whisper model: {MODEL_SIZE}")
    model = whisper.load_model(MODEL_SIZE, device=DEVICE)

    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    clean_files = list(AUDIO_CLEAN_DIR.glob("*.mp3"))
    noisy_files = list(AUDIO_NOISY_DIR.glob("*.mp3"))
    all_files = clean_files + noisy_files
    print(f"Found {len(all_files)} audio files\n")

    results = transcribe_all(model, all_files, ground_truth)
    save_results(results, RESULTS_PATH)
    print_summary(results)


if __name__ == "__main__":
    main()
