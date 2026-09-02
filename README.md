# 🎙️ Bhasha Setu (Hackheritage 4.0) — Hindi-to-Santali Speech & Translation Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-Meta--MMS--TTS-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/facebook/mms-tts-hin)
[![Vosk](https://img.shields.io/badge/Vosk-Hindi--ASR-00599C)](https://alphacephei.com/vosk/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Bhasha Setu** is an AI-powered offline-first speech recognition, multi-tiered translation, and speech synthesis platform. It is engineered to bridge linguistic barriers for migrant teachers, frontline healthcare workers (ASHA/Gram Sevaks), and indigenous tribal communities by translating real-time **Hindi spoken voice** into **Santali text (Ol Chiki & Roman script)** and generating **24kHz Santali speech audio**.

---

## 🌟 Key Architectural Features

### 1. 🎙️ Real-Time Offline Hindi Speech Recognition (ASR)
- Powered by **Vosk** (`vosk-model-small-hi-0.22`), providing zero-latency push-to-talk speech-to-text directly on local hardware without internet dependency.

### 2. 🧠 3-Tier Multi-Directional Translation Engine
- **Tier 1 (Phrase Bank & Exact DB Match):** Matches full sentences against verified educational phrases and a SQLite database (`translations.db`) containing 6,780+ curated entries.
- **Tier 2 (Grammar & Token Dictionary Mapping):** Word-level mapping for pronouns, verb stems, postpositions, and classroom vocabulary.
- **Tier 3 (Phonetic Transliteration Fallback):** Employs **Aksharamukha** syllabic mapping to transliterate out-of-vocabulary words and proper nouns, ensuring zero raw Devanagari characters remain in output.

### 3. 🔊 Santali Neural Text-to-Speech (TTS)
- Adapts Meta's **MMS-TTS Hindi** VITS architecture (`facebook/mms-tts-hin`) with an Ol Chiki-to-Devanagari phonetic mapping layer to produce clear, expressive 24kHz Santali voice waveforms.

### 4. 📂 Clean & Modular Directory Hygiene
- Automated setup script (`download_models.py`) downloads heavy model safetensors while lightweight models remain tracked under Git version control.

---

## 📂 Repository Structure

```text
Hackheritage4.0/
│── .gitignore                      # Git exclusion rules (ignores heavy model weights & cache)
│── README.md                       # Comprehensive architectural & user guide
│── requirements.txt                # Python environment dependencies
│── download_models.py              # Automated setup script for Hugging Face TTS weights
│── main.py                         # Master real-time CLI pipeline entry point
│
├── database/
│   ├── __init__.py
│   ├── translations.db             # SQLite lexicon database (6,780+ phrase & word entries)
│   └── datamerge.ipynb             # Dataset curation notebook
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized configuration & path management
│   ├── santali_tts.py              # Santali VITS speech synthesis module
│   ├── translate_hindi_santhali.py # 3-tier Hindi -> Santali translation engine
│   ├── generate_santali_audio.py   # CLI tool for generating audio files from text/transcripts
│   ├── hindi_voice_to_text.py      # Standalone push-to-talk Hindi ASR engine
│   ├── hindi_santhali_text.py      # Transformer sequence model inference utilities
│   ├── inference.py                # Standalone inference helper functions
│   └── test_tts.py                 # Unit tests for config & speech synthesis
│
├── models/
│   ├── vosk-model-small-hi-0.22/   # Tracked directly in Git (~45-75 MB ASR model)
│   └── mms_tts_hin/                # Ignored heavy weights (downloaded via download_models.py)
│
└── output/
    ├── hindi_transcripts/
    │   └── transcript.txt          # Appended Hindi ASR spoken text transcript
    ├── santali_transcripts/
    │   └── transcript_santhali.txt # Appended Santali Ol Chiki translated text transcript
    └── audio_translated/
        └── santali_output.wav      # Generated 24kHz audio WAV speech file
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python**: 3.10 or higher
- **Microphone & Audio Hardware**: Required for real-time push-to-talk voice recording.

### 2. Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Alfanshaikh786/SIH_Bhasha_Setu.git
   cd Hackheritage4.0
   ```

2. **Create & Activate Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Hugging Face Model Weights**:
   Execute the automated model setup script to fetch Meta MMS-TTS Hindi weights:
   ```bash
   python download_models.py
   ```

---

## 🎙️ Running the Pipeline

### Main Real-Time Voice Pipeline
Launch the interactive push-to-talk speech translation application:
```bash
python main.py
```

**Workflow:**
1. Press `[ENTER]` to start recording your voice.
2. Speak Hindi into your microphone.
3. Press `[ENTER]` again to stop recording.
4. The system transcribes your spoken Hindi, translates it to Santali Ol Chiki & Roman script, synthesizes speech, and plays the audio waveform back immediately.

---

## 🛠️ CLI Utilities & Testing

### 1. Standalone Translation CLI
Translate Hindi text directly to Santali Ol Chiki and Roman pronunciation:
```bash
python src/translate_hindi_santhali.py "मेरा नाम अंशु है"
```

### 2. Standalone Santali Audio Generator
Synthesize audio from a Santali transcript file or direct string:
```bash
python src/generate_santali_audio.py --text "ᱚᱞ ᱪᱤᱠᱤ" --output output/audio_translated/santali_output.wav
```

### 3. Run Unit Tests
Execute system test suite to verify configuration and synthesis integrity:
```bash
python -m unittest discover -s src
```

---

## 👥 Contributors & Acknowledgments
Developed for **Smart India Hackathon (SIH)**.

- **Institution**: Sahyadri College of Engineering and Management, Mangaluru
- **Core Engineering**: Smart India Hackathon Bhasha Setu Team
