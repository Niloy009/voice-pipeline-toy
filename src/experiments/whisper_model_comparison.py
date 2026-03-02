"""Compare Whisper model sizes and robustness on a subset of audio.

Runs selected Whisper model sizes (e.g. tiny, small, medium) on the same
clean and noisy audio subset, computes WER and keyword accuracy via
src.asr.transcribe utilities, and prints a table per (model_size, audio_type).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable
import whisper
import torch
from src.asr.transcribe import (
    AUDIO_CLEAN_DIR,
    AUDIO_NOISY_DIR,
    GROUND_TRUTH_PATH,
    compute_keyword_accuracy,
    compute_wer,
    get_audio_type,
    get_script_id_from_filename,
    load_ground_truth,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def select_audio_files(max_files: int | None, per_type: int = 2) -> list[Path]:
    """Select a balanced subset of audio files for the experiment.

    Takes up to per_type clean, per_type slight_noise, and per_type heavy_noise.

    Args:
        max_files: If set, cap total number of files.
        per_type: Max files per audio type (default 2).

    Returns:
        List of paths to selected audio files.
    """
    # Clean files live in AUDIO_CLEAN_DIR
    clean_files = sorted(AUDIO_CLEAN_DIR.glob("*.mp3"))[:per_type]

    # Noisy files live in AUDIO_NOISY_DIR; split by name pattern
    slight_files = sorted(
        p for p in AUDIO_NOISY_DIR.glob("*.mp3") if "slight_noise" in p.name
    )[:per_type]

    heavy_files = sorted(
        p for p in AUDIO_NOISY_DIR.glob("*.mp3") if "heavy_noise" in p.name
    )[:per_type]

    all_files = clean_files + slight_files + heavy_files

    if max_files is not None:
        all_files = all_files[:max_files]

    return all_files


def run_for_model(
    model_size: str,
    audio_files: Iterable[Path],
    ground_truth: dict,
) -> list[dict]:
    """Run Whisper for a given model size on the provided audio files.

    Args:
        model_size: Whisper model size (e.g. tiny, small, medium).
        audio_files: Paths to audio files to transcribe.
        ground_truth: Dict of script_id -> ground truth item.

    Returns:
        List of result dicts with model_size, audio_type, wer, keyword_accuracy.
    """
    print(f"\n=== Running Whisper model: {model_size} on {DEVICE} ===")
    model = whisper.load_model(model_size, device=DEVICE)

    results: list[dict] = []
    for audio_path in audio_files:
        script_id = get_script_id_from_filename(audio_path.name)
        gt = ground_truth.get(script_id)
        if not gt:
            print(f"No ground truth for {script_id} — skipping")
            continue

        print(f"Transcribing ({model_size}): {audio_path.name}")
        out = model.transcribe(str(audio_path), fp16=False)
        hypothesis = out["text"].strip()

        wer_score = compute_wer(gt["full_transcript"], hypothesis)
        keyword_acc = compute_keyword_accuracy(gt.get("keywords", []), hypothesis)

        results.append(
            {
                "model_size": model_size,
                "audio_type": get_audio_type(audio_path.name),
                "script_id": script_id,
                "audio_file": audio_path.name,
                "wer": wer_score,
                "keyword_accuracy": keyword_acc,
            }
        )

    return results


def summarize(results: list[dict]) -> None:
    """Print average WER and keyword accuracy per (model_size, audio_type).

    Args:
        results: List of result dicts with model_size, audio_type, wer,
            keyword_accuracy.
    """
    if not results:
        print("\nNo results to summarize.")
        return

    wer_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    kw_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)

    for r in results:
        key = (r["model_size"], r["audio_type"])
        wer_by_key[key].append(r["wer"])
        if r["keyword_accuracy"] is not None:
            kw_by_key[key].append(r["keyword_accuracy"])

    print("\n========== Whisper Model Comparison ==========")
    print(f"{'Model':<10} {'Audio Type':<15} {'Avg WER':>10} {'Keyword Acc':>15}")
    print("-" * 54)

    for model_size, audio_type in sorted(wer_by_key.keys()):
        wers = wer_by_key[(model_size, audio_type)]
        avg_wer = sum(wers) / len(wers)

        kw_scores = kw_by_key.get((model_size, audio_type), [])
        avg_kw = sum(kw_scores) / len(kw_scores) if kw_scores else 0.0

        print(f"{model_size:<10} {audio_type:<15} {avg_wer:>9.2%} {avg_kw:>14.2%}")

    print("==============================================")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the comparison.

    Returns:
        Namespace with models (list) and max_files (int).
    """
    parser = argparse.ArgumentParser(
        description="Compare Whisper model sizes on a subset of audio files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tiny", "small", "medium"],
        help="Whisper model sizes to compare (e.g. tiny small medium).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=6,
        help="Maximum number of audio files to use for the comparison.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Whisper model comparison and print summary table."""
    args = parse_args()

    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    audio_files = select_audio_files(args.max_files)

    if not audio_files:
        print("No audio files found under data/audio_clean and data/audio_noisy.")
        return

    all_results: list[dict] = []
    for model_size in args.models:
        model_results = run_for_model(model_size, audio_files, ground_truth)
        all_results.extend(model_results)

    summarize(all_results)


if __name__ == "__main__":
    main()
