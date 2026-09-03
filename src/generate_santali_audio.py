import argparse
import sys
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except ImportError:
    HAVE_SOUNDFILE = False
    from scipy.io import wavfile

try:
    from . import config
    from .santali_tts import SantaliTTS
except ImportError:
    import config
    from santali_tts import SantaliTTS


def save_audio_file(audio: np.ndarray, sample_rate: int, output_path: Path):
    """
    Saves audio array to WAV file format at specified sample rate (24000 Hz).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if HAVE_SOUNDFILE:
        sf.write(str(output_path), audio, sample_rate)
    else:
        # Scale float32 (-1.0 to 1.0) to int16 PCM for scipy.io.wavfile
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(str(output_path), sample_rate, audio_int16)

    print(f"[SUCCESS] Audio saved to: {output_path} (Sample Rate: {sample_rate} Hz)")


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize Santali text (Ol Chiki/ISO) from transcript_santhali.txt or CLI text to 24kHz WAV audio."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(config.SANTALI_TRANSCRIPT_PATH),
        help=f"Input text file containing Santali transcript (default: {config.SANTALI_TRANSCRIPT_PATH})."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Direct Santali text input in Ol Chiki script or Latin ISO transliteration."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.OUTPUT_AUDIO_PATH),
        help=f"Target output path for .wav file (default: {config.OUTPUT_AUDIO_PATH})."
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="female_educator",
        help="Voice prompt style description (default: female_educator)."
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech playback speed factor (default: 1.0)."
    )

    args = parser.parse_args()

    lines_to_process = []
    source_description = ""

    if args.text and args.text.strip():
        lines_to_process = [args.text.strip()]
        source_description = "Direct CLI text input"
    else:
        input_path = Path(args.input_file)
        if not input_path.is_absolute():
            input_path = config.BASE_DIR / input_path

        if input_path.exists():
            with open(input_path, "r", encoding="utf-8") as f:
                raw_lines = [line.strip() for line in f if line.strip()]
            if raw_lines:
                lines_to_process = raw_lines
                source_description = f"File '{input_path.name}' ({len(raw_lines)} lines)"

        if not lines_to_process:
            lines_to_process = ["ᱚᱞ ᱪᱤᱠᱤ"]
            source_description = "Default fallback text ('ᱚᱞ ᱪᱤᱠᱤ')"

    print("=" * 60)
    print("AURA TTS Voice Generation")
    print("=" * 60)
    print(f"Engine Name        : {config.ENGINE_NAME}")
    print(f"Sample Rate        : {config.SAMPLE_RATE} Hz")
    print(f"Source             : {source_description}")

    tts = SantaliTTS()

    audio_segments = []
    sr = config.SAMPLE_RATE

    for idx, line in enumerate(lines_to_process, 1):
        print(f"\n[Line {idx}/{len(lines_to_process)}] Synthesizing: {line}")
        segment_audio, sr = tts.synthesize(text=line, voice=args.voice, speed=args.speed)
        audio_segments.append(segment_audio)
        
        # Add 0.3s pause between lines if processing multiple sentences
        if idx < len(lines_to_process):
            silence_samples = int(0.3 * sr)
            audio_segments.append(np.zeros(silence_samples, dtype=np.float32))

    if audio_segments:
        final_audio = np.concatenate(audio_segments)
    else:
        final_audio = np.zeros(sr, dtype=np.float32)

    save_audio_file(final_audio, sr, Path(args.output))


if __name__ == "__main__":
    main()
