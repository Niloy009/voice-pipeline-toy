"""TTS synthesis for the readback pipeline.

Loads formatted summaries from readback_formatted.csv, sends each to the
ElevenLabs API (salesperson voice), exports one .mp3 per row to
data/audio_readback/, skips existing files, and saves a synthesis log CSV.
"""

import os
import io
import csv
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from pydub import AudioSegment
from .format_fields import load_extraction_results, build_formatted_rows, save_formatted_results

# Config
load_dotenv()

API_KEY = os.getenv("eleven_labs_api_key")
VOICE_READBACK = os.getenv("sales_paul")  # Paul voice
MODEL_ID = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"

FORMATTED_PATH = Path("results/readback_formatted.csv")
EXTRACTION_PATH = Path("results/extraction_results_few_shot_fuzz_2.csv")
OUTPUT_DIR = Path("data/audio_readback")
RESULTS_DIR = Path("results")
SYNTHESIS_LOG = RESULTS_DIR / "readback_synthesis_log.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ElevenLabs client
client = ElevenLabs(api_key=API_KEY)


def load_formatted_summaries(path: Path) -> list[dict]:
    """Load formatted summaries CSV; regenerate from extraction if missing.

    Args:
        path: Path to readback_formatted.csv.

    Returns:
        List of dicts with script_id, audio_type, audio_file, formatted_summary.
    """
    if not path.exists():
        print(f"Warning: {path} not found — regenerating from extraction results...")
        raw_rows = load_extraction_results(EXTRACTION_PATH)
        formatted = build_formatted_rows(raw_rows)
        save_formatted_results(formatted, path)
        return formatted

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows)} formatted summaries from {path}")
    return rows



def build_readback_filename(audio_file: str) -> str:
    """Derive the readback output filename from the source audio filename.

    Args:
        audio_file: Original audio filename (e.g. script_01_clean.mp3).

    Returns:
        Readback filename (e.g. script_01_clean_readback.mp3).
    """
    stem = Path(audio_file).stem   # e.g. script_01_clean
    return f"{stem}_readback.mp3"


def text_to_audio_bytes(text: str) -> bytes:
    """Send text to ElevenLabs API and return raw mp3 bytes.

    Uses the salesperson (Paul) voice for consistency with generate_audio.

    Args:
        text: The formatted spoken summary string.

    Returns:
        Raw mp3 audio bytes.
    """
    audio_bytes = b""
    for chunk in client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_READBACK,
        model_id=MODEL_ID,
        output_format=OUTPUT_FORMAT,
    ):
        audio_bytes += chunk
    return audio_bytes


def synthesize_readback(row: dict, output_dir: Path) -> dict:
    """Synthesize and save one readback audio file for a single row.

    Skips the API call if the output file already exists.

    Args:
        row: Dict with script_id, audio_type, audio_file, formatted_summary.
        output_dir: Directory to save the output .mp3 file.

    Returns:
        Dict with script_id, audio_type, audio_file, readback_file, status.
    """
    script_id = row["script_id"]
    audio_type = row["audio_type"]
    audio_file = row["audio_file"]
    summary = row["formatted_summary"]
    readback_file = build_readback_filename(audio_file)
    output_path = output_dir / readback_file

    # Skip if already generated
    if output_path.exists():
        print(f"Skipping (already exists): {readback_file}")
        return {
            "script_id": script_id,
            "audio_type": audio_type,
            "audio_file": audio_file,
            "readback_file": readback_file,
            "status": "skipped",
        }

    print(f"Synthesizing: {readback_file}")
    print(f"Summary: {summary[:80]}{'...' if len(summary) > 80 else ''}")

    try:
        audio_bytes = text_to_audio_bytes(summary)

        # Load via pydub to validate the audio before saving
        segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        segment.export(output_path, format="mp3")

        duration_s = round(len(segment) / 1000, 1)
        print(f"Saved: {output_path} ({duration_s}s)")

        return {
            "script_id": script_id,
            "audio_type": audio_type,
            "audio_file": audio_file,
            "readback_file": readback_file,
            "status": "success",
        }

    except Exception as e:
        print(f"Error synthesizing {readback_file}: {e}")
        return {
            "script_id": script_id,
            "audio_type": audio_type,
            "audio_file": audio_file,
            "readback_file": readback_file,
            "status": f"error: {e}",
        }


def save_synthesis_log(log: list[dict], path: Path) -> None:
    """Save the synthesis log to CSV.

    Args:
        log: List of result dicts with script_id, audio_type, readback_file, status.
        path: Output CSV path.
    """
    fieldnames = ["script_id", "audio_type", "audio_file", "readback_file", "status"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log)
    print(f"\nSynthesis log saved to {path}")


def print_summary(log: list[dict]) -> None:
    """Print a summary of synthesis results to console.

    Args:
        log: List of result dicts with status (success, skipped, error).
    """
    total = len(log)
    success = sum(1 for r in log if r["status"] == "success")
    skipped = sum(1 for r in log if r["status"] == "skipped")
    errors = sum(1 for r in log if r["status"].startswith("error"))

    print("\n========== Synthesis Summary ==========")
    print(f"Total rows : {total}")
    print(f"Synthesized : {success}")
    print(f"Skipped : {skipped}")
    print(f"Errors : {errors}")
    print("========================================")

    if errors:
        print("\nFailed files:")
        for r in log:
            if r["status"].startswith("error"):
                print(f"{r['readback_file']} — {r['status']}")


def main():
    """Orchestrate the full TTS synthesis pipeline.

    Loads formatted summaries (regenerating if missing), synthesizes readback
    .mp3 files via ElevenLabs, skips existing files, saves to
    data/audio_readback/, and writes the synthesis log CSV.
    """
    summaries = load_formatted_summaries(FORMATTED_PATH)

    log = []
    for row in summaries:
        result = synthesize_readback(row, OUTPUT_DIR)
        log.append(result)

    save_synthesis_log(log, SYNTHESIS_LOG)
    print_summary(log)


if __name__ == "__main__":
    main()
