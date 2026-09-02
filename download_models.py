#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Downloader for Hackheritage4.0
====================================
Downloads Meta MMS-TTS Hindi model weights from Hugging Face Transformers
and saves them to `./models/mms_tts_hin/`.

Usage:
    python download_models.py
"""

import sys
from pathlib import Path

# Ensure UTF-8 output streams in Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Base Directory
BASE_DIR = Path(__file__).resolve().parent
MMS_TTS_DIR = BASE_DIR / "models" / "mms_tts_hin"



def download_mms_tts():
    """
    Downloads facebook/mms-tts-hin model and tokenizer using transformers
    and saves pretrained files into ./models/mms_tts_hin/ if missing.
    """
    safetensor_path = MMS_TTS_DIR / "model.safetensors"
    if MMS_TTS_DIR.exists() and safetensor_path.exists():
        print(f"✅ MMS-TTS Hindi model already present at: {MMS_TTS_DIR}")
        return

    print("📥 Downloading facebook/mms-tts-hin model & tokenizer...")
    MMS_TTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoTokenizer, VitsModel
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
        model = VitsModel.from_pretrained("facebook/mms-tts-hin")

        tokenizer.save_pretrained(MMS_TTS_DIR)
        model.save_pretrained(MMS_TTS_DIR)
        print(f"✨ Successfully downloaded and saved MMS-TTS model to: {MMS_TTS_DIR}")
    except Exception as e:
        print(f"❌ Failed to download MMS-TTS model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_mms_tts()
