"""Unified demo entrypoint for the voice pipeline.

Provides a CLI that orchestrates TTS generation, noise augmentation, ASR
transcription, LLM field extraction, readback formatting, and readback TTS.
Use --mode to run full pipeline or specific stages (asr_only, extraction_only,
readback_only). Use --fast-demo for a smaller Whisper model and fewer files.
Performs environment and path checks with friendly error messages.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

from src.tts.generate_audio import main as tts_main
from src.noise.augment import main as noise_main
from src.asr.transcribe import run_transcription
from src.extraction.extract_fields import main as extraction_main
from src.tts_readback.format_fields import main as format_main
from src.tts_readback.synthesize_readback import main as readback_tts_main


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


def check_paths_exist(label: str, paths: Iterable[Path]) -> bool:
    """Return True if all paths exist; print friendly errors otherwise.

    Args:
        label: Description of the paths (e.g. "script file").
        paths: Iterable of paths to check.

    Returns:
        True if all paths exist, False otherwise.
    """
    missing = [p for p in paths if not p.exists()]
    if not missing:
        return True

    print(f"\n[ERROR] Missing required {label}:")
    for p in missing:
        print(f"  - {p}")
    print(
        "\nPlease make sure you have run the preceding pipeline stages or created "
        "the expected data before running this demo mode."
    )
    return False


def check_elevenlabs_env() -> bool:
    """Check for required ElevenLabs environment variables.

    Returns:
        True if all required vars are set, False otherwise (and prints errors).
    """
    required_vars = [
        "eleven_labs_api_key",
        "sales_paul",
        "client_rachel",
    ]
    missing = [v for v in required_vars if not os.getenv(v)]
    if not missing:
        return True

    print("\n[ERROR] Missing ElevenLabs environment variables:")
    for v in missing:
        print(f"  - {v}")
    print(
        "\nSet these variables in your environment or .env file before running "
        "modes that use TTS (tts or readback)."
    )
    return False


def check_ollama(model_name: str = "llama3.1:8b") -> bool:
    """Check that Ollama is importable and the model can be queried.

    Args:
        model_name: Ollama model to check (e.g. llama3.1:8b).

    Returns:
        True if Ollama is available and model responds, False otherwise.
    """
    try:
        import ollama  # type: ignore
    except Exception as exc:  # pragma: no cover - import robustness
        print("\n[ERROR] Could not import `ollama` Python package.")
        print("Install it and make sure Ollama is running locally.")
        print(f"Details: {exc}")
        return False

    try:
        print(f"\nChecking Ollama model '{model_name}'...")
        _ = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
        )
    except Exception as exc:  # pragma: no cover - runtime service check
        print(f"\n[ERROR] Ollama appears to be unavailable or model '{model_name}' is not ready.")
        print(
            "Make sure the Ollama daemon is running and that you have pulled "
            f"the model, e.g.: `ollama pull {model_name}`."
        )
        print(f"Details: {exc}")
        return False

    return True


def run_mode_full(fast_demo: bool) -> None:
    """Run the full pipeline: TTS, noise, ASR, extraction, readback."""
    print("=== Mode: full ===")

    if not check_elevenlabs_env():
        sys.exit(1)

    # 1) TTS generation (scripts -> clean audio)
    scripts_path = DATA_DIR / "scripts" / "scripts_final.txt"
    if not check_paths_exist("script file", [scripts_path]):
        sys.exit(1)
    tts_main()

    # 2) Noise augmentation (clean audio -> noisy audio)
    musan_dir = DATA_DIR / "musan" / "noise"
    if not check_paths_exist("MUSAN noise directory", [musan_dir]):
        sys.exit(1)
    noise_main()

    # 3) ASR + evaluation
    # For fast demo, use a smaller model and limit the number of files.
    model_size = "small" if fast_demo else "medium"
    max_files = 6 if fast_demo else None
    run_transcription(model_size=model_size, max_files=max_files, include_noisy=True)

    # 4) LLM extraction
    if not check_ollama():
        sys.exit(1)
    extraction_main()

    # 5) Readback formatting + TTS
    format_main()
    readback_tts_main()


def run_mode_asr_only(fast_demo: bool) -> None:
    """Run only the ASR and evaluation stage.

    Assumes clean and optionally noisy audio already exist under data/audio_*.
    """
    print("=== Mode: asr_only ===")
    clean_dir = DATA_DIR / "audio_clean"
    noisy_dir = DATA_DIR / "audio_noisy"
    if not check_paths_exist("audio directories", [clean_dir]):
        sys.exit(1)

    # Noisy dir is optional here; we handle its absence gracefully.
    if not noisy_dir.exists():
        print(f"[INFO] No noisy audio directory found at {noisy_dir} — running on clean audio only.")
        include_noisy = False
    else:
        include_noisy = True

    model_size = "small" if fast_demo else "medium"
    max_files = 6 if fast_demo else None
    run_transcription(model_size=model_size, max_files=max_files, include_noisy=include_noisy)


def run_mode_extraction_only() -> None:
    """Run only the LLM extraction stage.

    Assumes results/transcription_results.csv already exists.
    """
    print("=== Mode: extraction_only ===")
    transcriptions_csv = RESULTS_DIR / "transcription_results.csv"
    if not check_paths_exist("transcription results CSV", [transcriptions_csv]):
        sys.exit(1)

    if not check_ollama():
        sys.exit(1)

    extraction_main()


def run_mode_readback_only() -> None:
    """Run only the readback formatting and TTS stages.

    Assumes extraction results CSV exists; formatted CSV is regenerated if missing.
    """
    print("=== Mode: readback_only ===")
    extraction_csv = RESULTS_DIR / "extraction_results_few_shot_fuzz_2.csv"
    if not check_paths_exist("extraction results CSV", [extraction_csv]):
        sys.exit(1)

    if not check_elevenlabs_env():
        sys.exit(1)

    # format_fields.main() will read from extraction CSV and write formatted summaries.
    format_main()
    readback_tts_main()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the demo.

    Args:
        argv: Argument list (defaults to sys.argv).

    Returns:
        Parsed namespace with mode and fast_demo.
    """
    parser = argparse.ArgumentParser(
        description="Unified demo entrypoint for the Nogui voice pipeline.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "asr_only", "extraction_only", "readback_only"],
        default="full",
        help="Which part of the pipeline to run.",
    )
    parser.add_argument(
        "--fast-demo",
        action="store_true",
        help=("Use a smaller Whisper model and limit the number of files for faster demo."),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the demo with the selected mode and options."""
    args = parse_args(argv)

    if args.mode == "full":
        run_mode_full(fast_demo=args.fast_demo)
    elif args.mode == "asr_only":
        run_mode_asr_only(fast_demo=args.fast_demo)
    elif args.mode == "extraction_only":
        run_mode_extraction_only()
    elif args.mode == "readback_only":
        run_mode_readback_only()
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

