"""Batch ASR transcription and evaluation using OpenAI Whisper.

Loads clean and noisy audio, transcribes with Whisper (default: medium),
evaluates against ground truth using WER and keyword accuracy, and saves
results to CSV. Exposes run_transcription() for programmatic use with
configurable model size, file limit, and include_noisy option.
"""

import argparse
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
    """Load ground truth JSON and return a dict keyed by script id.

    Args:
        path: Path to the ground truth JSON file.

    Returns:
        Dict mapping script id to ground truth item.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def compute_wer(reference, hypothesis):
    """Compute Word Error Rate using the jiwer library.

    Args:
        reference: Ground truth transcript.
        hypothesis: ASR output transcript.

    Returns:
        WER score between 0 and 1 (rounded to 4 decimal places).
    """
    return round(wer(reference.lower(), hypothesis.lower()), 4)


def compute_keyword_accuracy(keywords, hypothesis):
    """Compute keyword accuracy: fraction of expected keywords in hypothesis.

    Args:
        keywords: List of expected keyword strings.
        hypothesis: ASR output transcript.

    Returns:
        Fraction of keywords found (0-1), or None if keywords is empty.
    """
    if not keywords:
        return None
    hyp_lower = hypothesis.lower()
    matched = sum(1 for kw in keywords if kw.lower() in hyp_lower)
    return round(matched / len(keywords), 4)


def get_script_id_from_filename(filename):
    """Extract script_id from audio filename.

    Args:
        filename: Audio filename (e.g. script_01_clean.mp3).

    Returns:
        Script id (e.g. script_01).
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    return f"{parts[0]}_{parts[1]}"


def get_audio_type(filename):
    """Determine audio type from filename.

    Args:
        filename: Audio filename containing slight_noise, heavy_noise, or neither.

    Returns:
        One of "clean", "slight_noise", "heavy_noise".
    """
    if "slight_noise" in filename:
        return "slight_noise"
    elif "heavy_noise" in filename:
        return "heavy_noise"
    return "clean"


def transcribe_all(model, audio_files, ground_truth):
    """Transcribe all audio files and compute WER and keyword accuracy for each.

    Args:
        model: Loaded Whisper model.
        audio_files: List of paths to audio files.
        ground_truth: Dict of script_id -> ground truth item.

    Returns:
        List of result dicts with script_id, audio_type, hypothesis, wer, etc.
    """
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
    """Save transcription results to CSV.

    Args:
        results: List of result dicts.
        path: Output CSV path.
    """
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
    """Print average WER and keyword accuracy per audio type.

    Args:
        results: List of result dicts with audio_type, wer, keyword_accuracy.
    """
    wer_by_type = defaultdict(list)
    keyword_by_type = defaultdict(list)

    for r in results:
        wer_by_type[r["audio_type"]].append(r["wer"])
        if r["keyword_accuracy"] is not None:
            keyword_by_type[r["audio_type"]].append(r["keyword_accuracy"])

    if not results:
        print("\nNo results to summarize.")
        return

    print("\n========== Evaluation Summary ==========")
    print(f"{'Audio Type':<20} {'Avg WER':>10} {'Keyword Acc':>15}")
    print("-" * 47)
    for audio_type in ["clean", "slight_noise", "heavy_noise"]:
        wers = wer_by_type.get(audio_type)
        if not wers:
            continue
        avg_wer = sum(wers) / len(wers)
        keyword_scores = keyword_by_type.get(audio_type, [])
        avg_keyword = (
            sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0
        )
        print(f"{audio_type:<20} {avg_wer:>9.2%} {avg_keyword:>14.2%}")
    print("=========================================")


def run_transcription(
    model_size: str = MODEL_SIZE,
    max_files: int | None = None,
    include_noisy: bool = True,
) -> None:
    """Run the transcription and evaluation pipeline with configurable options.

    Args:
        model_size: Whisper model size (e.g. tiny, base, small, medium, large).
        max_files: If set, limit processing to the first N audio files.
        include_noisy: If False, only clean audio files are transcribed.
    """
    print(f"Using device : {DEVICE}")
    print(f"Whisper model: {model_size}")
    model = whisper.load_model(model_size, device=DEVICE)

    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    clean_files = list(AUDIO_CLEAN_DIR.glob("*.mp3"))
    audio_files = list(clean_files)

    if include_noisy:
        noisy_files = list(AUDIO_NOISY_DIR.glob("*.mp3"))
        audio_files.extend(noisy_files)

    audio_files = sorted(audio_files)

    if max_files is not None:
        audio_files = audio_files[:max_files]

    print(f"Found {len(audio_files)} audio files\n")

    results = transcribe_all(model, audio_files, ground_truth)
    save_results(results, RESULTS_PATH)
    print_summary(results)


def main():
    """CLI entrypoint for the transcription and evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Batch ASR transcription and evaluation using Whisper."
    )
    parser.add_argument(
        "--model-size",
        default=MODEL_SIZE,
        help="Whisper model size (e.g. tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on the number of audio files to process.",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only transcribe clean audio (ignore noisy variants).",
    )

    args = parser.parse_args()
    include_noisy = not args.clean_only

    run_transcription(
        model_size=args.model_size,
        max_files=args.max_files,
        include_noisy=include_noisy,
    )


if __name__ == "__main__":
    main()
