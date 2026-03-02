"""Text-to-speech audio generation using the ElevenLabs API.

Parses conversation scripts (salesperson-client dialogues), assigns voices
per speaker, and exports merged MP3 files with pauses between lines.
"""

import os
import io
import re
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from pydub import AudioSegment


# Load environment variables
load_dotenv()

# Config
API_KEY = os.getenv("eleven_labs_api_key")
VOICE_SALESPERSON = os.getenv("sales_paul")    # "Paul" voice ID from ElevenLabs
VOICE_CLIENT = os.getenv("client_rachel") # "Rachel" voice ID from ElevenLabs
SCRIPTS_PATH = Path("data/scripts/scripts_final.txt")
OUTPUT_DIR = Path("data/audio_clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = ElevenLabs(api_key=API_KEY)


def parse_scripts(path):
    """Parse script file into a list of (script_id, lines) tuples.

    Args:
        path: Path to the script file (e.g. scripts_final.txt).

    Returns:
        List of (script_id, lines) where lines are "Salesperson:" or "Client:" strings.
    """
    scripts = []
    current_id = None
    current_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if re.match(r"^Script \d+$", line):
                if current_id and current_lines:
                    scripts.append((current_id, current_lines))
                current_id = line.replace(" ", "_").lower()
                current_lines = []
            elif line.startswith("Salesperson:") or line.startswith("Client:"):
                current_lines.append(line)

    if current_id and current_lines:
        scripts.append((current_id, current_lines))

    return scripts


def text_to_audio_segment(text, voice_name):
    """Convert text to pydub AudioSegment via ElevenLabs API.

    Args:
        text: Text to synthesize.
        voice_name: ElevenLabs voice ID.

    Returns:
        pydub AudioSegment (MP3).
    """
    audio_bytes = b""
    for chunk in client.text_to_speech.convert(
        text=text,
        voice_id=voice_name,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    ):
        audio_bytes += chunk
    return AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")


def generate_script_audio(script_id, lines):
    """Generate merged audio for a single script.

    Args:
        script_id: Identifier for the script (e.g. script_1).
        lines: List of "Salesperson:" or "Client:" dialogue lines.
    """
    print(f"Generating audio for {script_id}...")
    merged = AudioSegment.empty()
    silence = AudioSegment.silent(duration=500)  # 500ms pause between lines

    for line in lines:
        if line.startswith("Salesperson:"):
            text = line.replace("Salesperson:", "").strip()
            voice = VOICE_SALESPERSON
        elif line.startswith("Client:"):
            text = line.replace("Client:", "").strip()
            voice = VOICE_CLIENT
        else:
            continue

        segment = text_to_audio_segment(text, voice)
        merged += segment + silence

    out_path = OUTPUT_DIR / f"{script_id}_clean.mp3"
    merged.export(out_path, format="mp3")
    print(f"Saved: {out_path}")


def main():
    """Parse scripts and generate audio files for all scripts."""
    scripts = parse_scripts(SCRIPTS_PATH)
    print(f"Found {len(scripts)} scripts")
    for script_id, lines in scripts:
        generate_script_audio(script_id, lines)
    print("Done! All audio files generated.")


if __name__ == "__main__":
    main()
