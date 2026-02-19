"""Audio augmentation module for adding noise to clean audio files.

This module creates noisy versions of clean audio files for data augmentation
in voice processing tasks. It mixes clean audio with noise samples from the
MUSAN dataset at configurable noise levels to generate both slightly noisy
and heavily noisy audio variants.

Key features:
- Loads clean audio from a source directory
- Applies MUSAN noise files at specified dB levels
- Generates two augmented versions per input file (slight and heavy noise)
- Exports results as MP3 files
"""

import os
import random
from pathlib import Path
from pydub import AudioSegment

# Config
CLEAN_DIR = Path("data/audio_clean")
NOISY_DIR = Path("data/audio_noisy")
MUSAN_DIR = Path("data/musan/noise")
NOISY_DIR.mkdir(parents=True, exist_ok=True)

# Noise levels in dB — negative means quieter than original
SLIGHT_NOISE_DB = -30   # slightly noisy
HEAVY_NOISE_DB  = -10   # heavily noisy


def get_all_noise_files(musan_dir):
    """Recursively collect all .wav noise files from MUSAN noise folder."""
    noise_files = []
    for root, _, files in os.walk(musan_dir):
        for f in files:
            if f.endswith(".wav"):
                noise_files.append(Path(root) / f)
    print(f"Found {len(noise_files)} noise files")
    return noise_files


def mix_with_noise(clean_audio, noise_file, noise_db):
    """Mix a clean AudioSegment with a noise file at a given dB level."""
    noise = AudioSegment.from_file(noise_file)

    # Loop noise if shorter than clean audio
    while len(noise) < len(clean_audio):
        noise += noise

    # Trim noise to match clean audio length
    noise = noise[:len(clean_audio)]

    # Adjust noise volume
    noise = noise + noise_db

    # Mix clean audio with noise
    return clean_audio.overlay(noise)


def augment_script(clean_path, noise_files):
    """Create 2 noisy versions of a single clean audio file."""
    clean_audio = AudioSegment.from_file(clean_path)
    script_id = clean_path.stem.replace("_clean", "")

    # Pick a random noise file for each version
    noise_file_1 = random.choice(noise_files)
    noise_file_2 = random.choice(noise_files)

    # Slightly noisy version
    slight = mix_with_noise(clean_audio, noise_file_1, SLIGHT_NOISE_DB)
    slight_path = NOISY_DIR / f"{script_id}_slight_noise.mp3"
    slight.export(slight_path, format="mp3")
    print(f"Saved: {slight_path}")

    # Heavily noisy version
    heavy = mix_with_noise(clean_audio, noise_file_2, HEAVY_NOISE_DB)
    heavy_path = NOISY_DIR / f"{script_id}_heavy_noise.mp3"
    heavy.export(heavy_path, format="mp3")
    print(f"Saved: {heavy_path}")


def main():
    """Run the full audio augmentation pipeline.

    Loads all noise files from the MUSAN directory and all clean MP3s from the
    clean audio directory, then generates two noisy variants (slight and heavy)
    for each clean file, saving the results to the noisy audio directory.
    """
    noise_files = get_all_noise_files(MUSAN_DIR)
    clean_files = sorted(CLEAN_DIR.glob("*.mp3"))
    print(f"Found {len(clean_files)} clean audio files")

    for clean_path in clean_files:
        augment_script(clean_path, noise_files)

    print(f"Done! Generated {len(clean_files) * 2} noisy audio files in {NOISY_DIR}")


if __name__ == "__main__":
    main()
