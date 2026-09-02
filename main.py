import os
import sys
import io
import json
import threading
import sounddevice as sd
from pathlib import Path

# Ensure UTF-8 output streams in Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import dependencies
from vosk import Model, KaldiRecognizer  # type: ignore
import soundfile as sf
import numpy as np

from src import config
from src.translate_hindi_santhali import HindiSanthaliTranslator
from src.santali_tts import SantaliTTS

# File Paths
MODEL_PATH = config.VOSK_MODEL_PATH
TRANSCRIPT_HINDI_PATH = config.HINDI_TRANSCRIPT_PATH
TRANSCRIPT_SANTALI_PATH = config.SANTALI_TRANSCRIPT_PATH
OUTPUT_AUDIO_PATH = config.OUTPUT_AUDIO_PATH


def play_audio(audio: np.ndarray, sample_rate: int, wav_path: Path):
    """
    Plays audio live via sounddevice, with fallback to Windows winsound.
    """
    try:
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"[Fallback Playback] sounddevice play error ({e}), trying winsound...")
        try:
            import winsound
            winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
        except Exception as win_err:
            print(f"[Warning] Audio playback failed: {win_err}")


def save_wav_file(audio: np.ndarray, sample_rate: int, wav_path: Path):
    """
    Saves float32 audio waveform to 24kHz WAV file.
    """
    try:
        sf.write(str(wav_path), audio, sample_rate)
        print(f"💾 Saved Audio WAV to: {wav_path}")
    except Exception as e:
        print(f"[Error] Failed to save WAV file: {e}")


def main():
    if not MODEL_PATH.exists():
        print(f"\n❌ [Error] Vosk Hindi speech model not found at: {MODEL_PATH}")
        sys.exit(1)

    print("=" * 65)
    print(" 🎙️  REAL-TIME HINDI SPEECH -> SANTALI TRANSLATION -> AUDIO PIPELINE")
    print("=" * 65)
    print("1. Loading Vosk Hindi Speech-to-Text Model...")
    vosk_model = Model(str(MODEL_PATH))
    SAMPLE_RATE = 16000

    print("2. Initializing 3-Tier Hindi -> Santali Translator Engine...")
    translator = HindiSanthaliTranslator()

    print("3. Initializing Santali Text-to-Speech Engine...")
    tts = SantaliTTS()

    print("\n✅ ALL ENGINES LOADED SUCCESSFULLY!")
    print(f"📄 Hindi Transcripts Path  : {TRANSCRIPT_HINDI_PATH}")
    print(f"📄 Santali Transcripts Path: {TRANSCRIPT_SANTALI_PATH}")
    print(f"🔊 Output Audio WAV Path  : {OUTPUT_AUDIO_PATH}")
    print("=" * 65)

    try:
        while True:
            print("\n" + "-" * 65)
            user_cmd = input("👉 Press [ENTER] to START recording (or type 'q' and press ENTER to exit): ")
            if user_cmd.strip().lower() in ['q', 'quit', 'exit']:
                print("\nExiting pipeline...")
                break

            stop_recording = False

            def wait_for_stop():
                nonlocal stop_recording
                input()
                stop_recording = True

            stop_thread = threading.Thread(target=wait_for_stop)
            stop_thread.daemon = True
            stop_thread.start()

            print("🎙️ RECORDING... Speak Hindi into microphone! Press [ENTER] to STOP...")

            recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)

            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=4000, dtype='int16', channels=1) as stream:
                while not stop_recording:
                    data, _ = stream.read(4000)
                    recognizer.AcceptWaveform(bytes(data))

            # Decode spoken Hindi text
            result = json.loads(recognizer.FinalResult())
            hindi_text = result.get("text", "").strip()

            if not hindi_text:
                print("\n⚠️ No clear speech detected. Please try again.")
                continue

            print(f"\n✅ 1. Hindi Speech Transcribed: \"{hindi_text}\"")

            # 1. Save Hindi text to transcript.txt
            with open(TRANSCRIPT_HINDI_PATH, "a", encoding="utf-8") as f:
                f.write(hindi_text + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"   📝 Saved to {TRANSCRIPT_HINDI_PATH.name}")

            # 2. Translate Hindi to Santali (Ol Chiki) using 3-tier engine
            res = translator.translate(hindi_text)
            santali_text = res["santali_ol_chiki"]
            santali_roman = res["santali_roman"]

            print(f"\n✨ 2. Santali Translated (Ol Chiki) : {santali_text}")
            print(f"   🔤 Santali Pronunciation (Roman) : {santali_roman}")

            # Save Santali text to transcript_santhali.txt
            with open(TRANSCRIPT_SANTALI_PATH, "a", encoding="utf-8") as f:
                f.write(santali_text + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"   📝 Saved to {TRANSCRIPT_SANTALI_PATH.name}")

            # 3. Generate Santali Audio waveform
            print("\n🔊 3. Generating Santali Audio Speech...")
            audio_arr, sr = tts.synthesize(santali_text)

            # Save audio file
            save_wav_file(audio_arr, sr, OUTPUT_AUDIO_PATH)

            # 4. Play audio immediately
            print("▶️  Playing Santali Audio Output...")
            play_audio(audio_arr, sr, OUTPUT_AUDIO_PATH)
            print("✅ Audio Playback Complete!")

    except KeyboardInterrupt:
        print("\n\nPipeline stopped by user (KeyboardInterrupt).")
    finally:
        translator.close()
        print("\nPipeline shutdown cleanly. Goodbye!")


if __name__ == "__main__":
    main()
