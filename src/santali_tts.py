import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from transformers import AutoTokenizer, VitsModel

try:
    from . import config
except ImportError:
    import config

# Force UTF-8 encoding for standard output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Attempt aksharamukha import for Ol Chiki -> Devanagari transliteration
try:
    from aksharamukha import transliterate
    HAVE_AKSHARAMUKHA = True
except ImportError:
    HAVE_AKSHARAMUKHA = False

# Local Model Path
LOCAL_MODEL_PATH = config.MMS_TTS_MODEL_PATH



# Fallback Ol Chiki to Devanagari mapping
OL_CHIKI_TO_DEVANAGARI = {
    'ᱚ': 'अ',  'ᱛ': 'त',  'ᱜ': 'ग',  'ᱝ': 'ं',  'ᱞ': 'ल',
    'ᱟ': 'आ',  'ᱠ': 'क',  'ᱡ': 'ज',  'ᱢ': 'म',  'ᱣ': 'व',
    'ᱤ': 'इ',  'ᱥ': 'स',  'ᱦ': 'ह',  'ᱧ': 'ञ',  'ᱨ': 'र',
    'ᱩ': 'उ',  'ᱪ': 'च',  'ᱫ': 'द',  'ᱬ': 'ण',  'ᱭ': 'य',
    'ᱮ': 'ए',  'ᱯ': 'प',  'ᱰ': 'ड',  'ᱱ': 'न',  'ᱲ': 'ड़',
    'ᱳ': 'ओ',  'ᱴ': 'ट',  'ᱵ': 'ब',  'ᱶ': 'ं',  'ᱷ': 'ह',
    'ᱸ': 'ं',  'ᱹ': '',   'ᱺ': 'ं',  'ᱻ': '',   'ᱼ': '',
    'ᱽ': '',   '᱾': '।',  '᱿': '॥'
}

def fallback_transliterate_ol_chiki(text: str) -> str:
    """
    Fallback transliteration from Ol Chiki script into Devanagari phonetics.
    """
    res = []
    for ch in text:
        res.append(OL_CHIKI_TO_DEVANAGARI.get(ch, ch))
    return "".join(res)

def transliterate_to_devanagari(text: str) -> str:
    """
    Transliterates Santali Ol Chiki text into Devanagari script.
    Uses Aksharamukha if available, with a rule-based fallback.
    """
    if HAVE_AKSHARAMUKHA:
        try:
            res = transliterate.process('Santali', 'Devanagari', text)
            if res:
                return res
        except Exception as e:
            print(f"[NOTE] Aksharamukha transliteration note: {e}")
    return fallback_transliterate_ol_chiki(text)


class SantaliTTS:
    """
    Offline Text-to-Speech Engine for Santali Ol Chiki using Meta MMS-TTS Hindi VITS model.
    Synthesizes Santali text (transliterated to Devanagari) into single-channel float32 audio waveform.
    """

    def __init__(
        self,
        default_prompt: str = config.DEFAULT_DESCRIPTION_PROMPT,
        sample_rate: int = config.SAMPLE_RATE
    ):
        self.default_prompt = default_prompt
        self.sample_rate = sample_rate

        if not LOCAL_MODEL_PATH.exists():
            raise FileNotFoundError(f"Local MMS-TTS Hindi model checkpoint not found at: {LOCAL_MODEL_PATH}")

        print(f"Loading local MMS-TTS Hindi model from {LOCAL_MODEL_PATH}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_PATH))
        self.model = VitsModel.from_pretrained(str(LOCAL_MODEL_PATH))
        self.model.eval()

        # Load vocab keys for safe input sanitization
        vocab_file = LOCAL_MODEL_PATH / "vocab.json"
        if vocab_file.exists():
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.vocab = set(json.load(f).keys())
        else:
            self.vocab = set(self.tokenizer.get_vocab().keys())

        # Determine native model sample rate (MMS-TTS Hin native SR is typically 16000 Hz)
        self.model_sr = getattr(self.model.config, "sampling_rate", 16000)

    def sanitize_for_vocab(self, text: str) -> str:
        """
        Strips any characters not present in the model's Devanagari vocabulary
        to prevent [UNK] static and out-of-vocabulary tensor index errors.
        """
        sanitized = []
        for char in text:
            if char in self.vocab:
                sanitized.append(char)
            elif char in OL_CHIKI_TO_DEVANAGARI:
                dev_char = OL_CHIKI_TO_DEVANAGARI[char]
                for dc in dev_char:
                    if dc in self.vocab:
                        sanitized.append(dc)
            elif char == '.' and '।' in self.vocab:
                sanitized.append('।')
            elif char.isspace():
                if ' ' in self.vocab:
                    sanitized.append(' ')
        return "".join(sanitized).strip()

    def synthesize(
        self,
        text: str,
        description_prompt: Optional[str] = None,
        voice: Optional[str] = None,
        speed: float = 0.85
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesizes Santali Ol Chiki text into a 1D float32 audio numpy array at target sample rate.
        """
        if not text or not text.strip():
            # Return 0.5s silence for empty input
            silent_samples = int(0.5 * self.sample_rate)
            return np.zeros(silent_samples, dtype=np.float32), self.sample_rate

        # 1. Transliterate Ol Chiki to Devanagari script
        devanagari_text = transliterate_to_devanagari(text)

        # 2. Sanitize against vocab.json
        clean_text = self.sanitize_for_vocab(devanagari_text)

        if not clean_text:
            silent_samples = int(0.5 * self.sample_rate)
            return np.zeros(silent_samples, dtype=np.float32), self.sample_rate

        try:
            # 3. Tokenize input Devanagari text
            inputs = self.tokenizer(clean_text, return_tensors="pt")

            # 4. Generate raw audio waveform
            with torch.no_grad():
                output = self.model(**inputs).waveform

            # 5. Extract 1D numpy float32 array
            audio = output.squeeze().cpu().numpy().astype(np.float32)

            # Adjust speed if requested (simple speed scaling)
            if speed != 1.0 and speed > 0.1 and len(audio) > 0:
                num_speed_samples = int(round(len(audio) / speed))
                audio = np.interp(
                    np.linspace(0, len(audio), num_speed_samples, endpoint=False),
                    np.arange(len(audio)),
                    audio
                ).astype(np.float32)

            # 6. Resample from model native rate to target sample rate (e.g. 16kHz -> 24kHz)
            if self.model_sr != self.sample_rate and len(audio) > 0:
                num_target_samples = int(round(len(audio) * float(self.sample_rate) / float(self.model_sr)))
                audio = np.interp(
                    np.linspace(0, len(audio), num_target_samples, endpoint=False),
                    np.arange(len(audio)),
                    audio
                ).astype(np.float32)

            return audio, self.sample_rate

        except Exception as e:
            print(f"[Error] TTS Synthesis failure: {e}")
            silent_samples = int(0.5 * self.sample_rate)
            return np.zeros(silent_samples, dtype=np.float32), self.sample_rate


if __name__ == "__main__":
    print("=" * 60)
    print("SANTALI TTS LOCAL MMS-TTS HINDI ENGINE DEMONSTRATION")
    print("=" * 60)
    tts = SantaliTTS()
    sample_text = "ᱤᱧ ᱡᱚᱢᱟᱜ ᱡᱚᱢ ᱟᱹᱰᱤ ᱧ ᱠᱩᱥᱤᱭᱟᱜ-ᱟ ᱾"
    print(f"Synthesizing demo text : {sample_text}")
    dev_text = transliterate_to_devanagari(sample_text)
    print(f"Devanagari Transliterated: {dev_text}")
    print(f"Sanitized Vocab Text    : {tts.sanitize_for_vocab(dev_text)}")
    audio, sr = tts.synthesize(sample_text)
    out_path = Path("output/santali_output.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sf.write(str(out_path), audio, sr)
    print(f"[SUCCESS] Audio saved to: {out_path} (Sample Rate: {sr} Hz, Shape: {audio.shape})")
