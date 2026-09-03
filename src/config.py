import os
from pathlib import Path

# Repository Root Directory (parent of src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Models Paths
MODELS_DIR = BASE_DIR / "models"
VOSK_MODEL_PATH = MODELS_DIR / "vosk-model-small-hi-0.22"
MMS_TTS_MODEL_PATH = MODELS_DIR / "mms_tts_hin"

# Database Paths
DATABASE_DIR = BASE_DIR / "database"
DB_PATH = DATABASE_DIR / "translations.db"

# Output Paths
OUTPUT_DIR = BASE_DIR / "output"
HINDI_TRANSCRIPTS_DIR = OUTPUT_DIR / "hindi_transcripts"
SANTALI_TRANSCRIPTS_DIR = OUTPUT_DIR / "santali_transcripts"
AUDIO_TRANSLATED_DIR = OUTPUT_DIR / "audio_translated"

HINDI_TRANSCRIPT_PATH = HINDI_TRANSCRIPTS_DIR / "transcript.txt"
SANTALI_TRANSCRIPT_PATH = SANTALI_TRANSCRIPTS_DIR / "transcript_santhali.txt"
OUTPUT_AUDIO_PATH = AUDIO_TRANSLATED_DIR / "santali_output.wav"

# Ensure output subdirectories exist
HINDI_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
SANTALI_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_TRANSLATED_DIR.mkdir(parents=True, exist_ok=True)

# SIH Bhasha Setu Santali Speech Engine Configuration
ENGINE_NAME = "AURA - Ai"
DEFAULT_DESCRIPTION_PROMPT = (
    "A male speaker delivers a clear, natural, and expressive educational speech in Santali."
)

# Audio Parameters
SAMPLE_RATE = 24000

# Language Codes
SRC_LANG = "hin_Deva"  # Hindi in Devanagari script
TGT_LANG = "sat_Olck"  # Santali in Ol Chiki script
