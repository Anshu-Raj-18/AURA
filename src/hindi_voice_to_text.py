import os
import sys
import io
import json
import threading
import sounddevice as sd
from vosk import Model, KaldiRecognizer  # type: ignore

# Ensure UTF-8 output streams
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    from . import config
except ImportError:
    import config

MODEL_PATH = str(config.VOSK_MODEL_PATH)
TRANSCRIPT_PATH = str(config.HINDI_TRANSCRIPT_PATH)

if not os.path.exists(MODEL_PATH):
    print(f"\n[Error] Vosk Hindi model not found at: {MODEL_PATH}")
    sys.exit(1)


print("Loading Hindi speech model...")
model = Model(MODEL_PATH)
SAMPLE_RATE = 16000

print("\n" + "=" * 55)
print("  PALASH Push-to-Talk Hindi Speech-to-Text")
print(f"  Transcripts will save to: {TRANSCRIPT_PATH}")
print("=" * 55)

while True:
    try:
        input("\n👉 Press [ENTER] to START recording (or Ctrl+C to exit)... ")
        
        stop_recording = False
        def wait_for_stop():
            global stop_recording
            input()
            stop_recording = True

        stop_thread = threading.Thread(target=wait_for_stop)
        stop_thread.daemon = True
        stop_thread.start()

        print("🎙️ RECORDING... Speak now! Press [ENTER] again to STOP...")

        recognizer = KaldiRecognizer(model, SAMPLE_RATE)
        
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=4000, dtype='int16', channels=1) as stream:
            while not stop_recording:
                data, _ = stream.read(4000)
                recognizer.AcceptWaveform(bytes(data))

        # Decode spoken sentence
        result = json.loads(recognizer.FinalResult())
        hindi_text = result.get("text", "").strip()

        if hindi_text:
            print(f"\n✅ Transcribed: {hindi_text}")
            
            # Write, flush buffer, and force immediate OS disk sync
            with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
                f.write(hindi_text + "\n")
                f.flush()
                os.fsync(f.fileno())
                
            print(f"📝 Saved to {os.path.basename(TRANSCRIPT_PATH)}")
        else:
            print("\n⚠️ No clear speech detected.")

    except KeyboardInterrupt:
        print("\n\nStopping Speech-to-Text Engine...")
        break