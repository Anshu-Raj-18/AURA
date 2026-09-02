import unittest
import sys
import numpy as np
from pathlib import Path

# Add repository root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src import config
    from src.santali_tts import SantaliTTS
except ImportError:
    import config
    from santali_tts import SantaliTTS




class TestSantaliTTS(unittest.TestCase):

    def setUp(self):
        self.tts = SantaliTTS()

    def test_config_model_settings(self):
        """
        Validate config file specifies Quipus / AI4Bharat model settings and 24kHz sample rate.
        """
        self.assertEqual(config.SAMPLE_RATE, 24000)
        self.assertEqual(config.SRC_LANG, "hin_Deva")
        self.assertEqual(config.TGT_LANG, "sat_Olck")
        self.assertEqual(config.ENGINE_NAME, "SIH_Bhasha_Setu_Phonetic_TTS")

    def test_santali_speech_synthesis(self):
        """
        Validate Santali TTS speech synthesis produces valid 24kHz audio output array for Ol Chiki text.
        """
        ol_chiki_text = "ᱚᱞ ᱪᱤᱠᱤ"
        audio, sr = self.tts.synthesize(ol_chiki_text)

        self.assertEqual(sr, 24000)
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.dtype, np.float32)
        self.assertGreater(audio.size, 0)


if __name__ == "__main__":
    unittest.main()

