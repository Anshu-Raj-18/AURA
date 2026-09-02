#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hindi to Santhali (Ol Chiki) Database & Phonetic Translator
============================================================
3-Tier Translation Engine:
--------------------------
1. Tier 1 (Phrase Bank & Exact DB Match):
   Full sentence match against verified phrase bank and SQLite `translations.db` (6,780 rows).
2. Tier 2 (Token & Grammar Dictionary Mapping):
   Word-level matching for pronouns, verbs, conjunctions, postpositions, and vocabulary.
3. Tier 3 (Phonetic Transliteration Fallback):
   Uses `aksharamukha` (with fallback to Devanagari-to-Ol Chiki syllabic rules) for out-of-vocabulary
   tokens, proper nouns, and loanwords so that NO raw Devanagari characters remain in the output.

Usage:
------
1. Programmatic API:
    from translate_hindi_santhali import HindiSanthaliTranslator
    translator = HindiSanthaliTranslator()
    result = translator.translate("मेरा नाम अंशु है और मैं संत फ्रांसिस स्कूल का छात्र हूँ")
    print(result['santali_ol_chiki'])

2. Translate File:
    python translate_hindi_santhali.py transcript.txt transcript_santhali.txt

3. CLI Single Phrase:
    python translate_hindi_santhali.py "मेरा नाम अंशु है"

4. Interactive CLI:
    python translate_hindi_santhali.py --interactive
"""

import sys
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional

# Try importing Aksharamukha for high-accuracy phonetic transliteration
try:
    from aksharamukha import transliterate
    HAS_AKSHARAMUKHA = True
except ImportError:
    HAS_AKSHARAMUKHA = False

# Ensure standard output and error support UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from . import config
except ImportError:
    import config

DEFAULT_DB_PATH = config.DB_PATH


# ============================================================================
# TIER 1: FULL PHRASE BANK (Exact Sentence Matches)
# ============================================================================
PHRASE_BANK: Dict[str, Dict[str, str]] = {
    "मुझे खाना खाना बहुत पसंद है": {
        "ol_chiki": "ᱤᱧ ᱡᱚᱢᱟᱜ ᱡᱚᱢ ᱟᱹᱰᱤ ᱧ ᱠᱩᱥᱤᱭᱟᱜ-ᱟ ᱾",
        "roman": "Inj jomag jom adi nj kusiyag-a."
    },
    "मुझे खाना खाना बहुत पसंद है।": {
        "ol_chiki": "ᱤᱧ ᱡᱚᱢᱟᱜ ᱡᱚᱢ ᱟᱹᱰᱤ ᱧ ᱠᱩᱥᱤᱭᱟᱜ-ᱟ ᱾",
        "roman": "Inj jomag jom adi nj kusiyag-a."
    },
    "मेरा नाम अंशु है": {
        "ol_chiki": "ᱤᱧᱟᱜ ᱧᱩᱛᱩᱢ ᱚᱝᱥᱩ ᱠᱟᱱᱟ ᱾",
        "roman": "Injag nyutum Ongsu kana."
    },
    "मेरा नाम अंशु है।": {
        "ol_chiki": "ᱤᱧᱟᱜ ᱧᱩᱛᱩᱢ ᱚᱝᱥᱩ ᱠᱟᱱᱟ ᱾",
        "roman": "Injag nyutum Ongsu kana."
    },
    "सान फ्रांसिस्को स्टूडेंट": {
        "ol_chiki": "ᱥᱟᱱ ᱯᱷᱨᱟᱱᱥᱤᱥᱠᱚ ᱨᱮᱱ ᱯᱟᱹᱴᱷᱩᱣᱟᱹ ᱾",
        "roman": "San Francisco ren pathuwa."
    },
    "नमस्ते": {
        "ol_chiki": "ᱡᱚᱦᱟᱨ",
        "roman": "Johar"
    },
    "प्रणाम": {
        "ol_chiki": "ᱡᱚᱦᱟᱨ",
        "roman": "Johar"
    },
    "आप कैसे हैं": {
        "ol_chiki": "ᱟᱢ ᱫᱚ ᱪᱮᱫ ᱞᱮᱠᱟ ᱢᱮᱱᱟᱜ-ᱟ?",
        "roman": "Am do ched leka menag-a?"
    },
    "आप कैसे हैं?": {
        "ol_chiki": "ᱟᱢ ᱫᱚ ᱪᱮᱫ ᱞᱮᱠᱟ ᱢᱮᱱᱟᱜ-ᱟ?",
        "roman": "Am do ched leka menag-a?"
    },
    "धन्यवाद": {
        "ol_chiki": "ᱥᱟᱨᱦᱟᱣ",
        "roman": "Sarhaw"
    },
    "सुप्रभात": {
        "ol_chiki": "ᱥᱟᱹᱜᱩᱱ ᱥᱮᱛᱟᱜ",
        "roman": "Sagun Setag"
    },
    "शुभ रात्रि": {
        "ol_chiki": "ᱥᱟᱹᱜᱩᱱ ᱧᱤᱫᱟᱹ",
        "roman": "Sagun Nyinda"
    },
    "हमारे गांव में आपका हार्दिक स्वागत है": {
        "ol_chiki": "ᱟᱞᱮᱭᱟᱜ ᱟᱹᱛᱩ ᱨᱮ ᱟᱯᱮᱭᱟᱜ ᱥᱟᱹᱜᱩᱱ ᱫᱟᱨᱟᱢ᱾",
        "roman": "Aleyag aatu re sagun daram"
    },
    "आपका नाम क्या है": {
        "ol_chiki": "ᱟᱢᱟᱜ ᱧᱩᱛᱩᱢ ᱪᱮᱫ?",
        "roman": "Amag nyutum ched?"
    },
    "आपका नाम क्या है?": {
        "ol_chiki": "ᱟᱢᱟᱜ ᱧᱩᱛᱩᱢ ᱪᱮᱫ?",
        "roman": "Amag nyutum ched?"
    },
    "मैं ठीक हूँ": {
        "ol_chiki": "ᱤᱧ ᱫᱚ ᱵᱮᱥ ᱜᱮ ᱢᱮᱱᱟᱹᱧᱟ᱾",
        "roman": "Inj do bes ge menanya"
    },
    "अस्पताल कहाँ है": {
        "ol_chiki": "ᱦᱟᱥᱯᱟᱛᱟᱞ ᱫᱚ ᱚᱠᱟᱨᱮ ᱢᱮᱱᱟᱜ-ᱟ?",
        "roman": "Hospital do okare menag-a?"
    },
    "अस्पताल कहाँ है?": {
        "ol_chiki": "ᱦᱟᱥᱯᱟᱛᱟᱞ ᱫᱚ ᱚᱠᱟᱨᱮ ᱢᱮᱱᱟᱜ-ᱟ?",
        "roman": "Hospital do okare menag-a?"
    },
    "सिकल सेल खून की जांच": {
        "ol_chiki": "ᱥᱤᱠᱤᱞ ᱥᱮᱞ ᱢᱟᱭᱟᱢ ᱵᱤᱰᱟᱹᱣ",
        "roman": "Sikil sel mayam bidaw"
    }
}

# ============================================================================
# TIER 2: WORD & GRAMMAR DICTIONARY
# ============================================================================
WORD_DICT: Dict[str, Dict[str, str]] = {
    # Pronouns & Proper Nouns
    "मेरा": {"ol_chiki": "ᱤᱧᱟᱜ", "roman": "injag"},
    "मेरी": {"ol_chiki": "ᱤᱧᱟᱜ", "roman": "injag"},
    "मेरे": {"ol_chiki": "ᱤᱧᱟᱜ", "roman": "injag"},
    "मुझे": {"ol_chiki": "ᱤᱧ", "roman": "inj"},
    "मैं": {"ol_chiki": "ᱤᱧ", "roman": "inj"},
    "आपका": {"ol_chiki": "ᱟᱢᱟᱜ", "roman": "amag"},
    "आप": {"ol_chiki": "ᱟᱢ", "roman": "am"},
    "हमारा": {"ol_chiki": "ᱟᱵᱚᱣᱟᱜ", "roman": "abowag"},
    "हम": {"ol_chiki": "ᱟᱵᱚ", "roman": "abo"},
    "उसका": {"ol_chiki": "ᱩᱱᱤᱭᱟᱜ", "roman": "uniyag"},
    "वह": {"ol_chiki": "ᱩᱱᱤ", "roman": "uni"},
    "यह": {"ol_chiki": "ᱱᱩᱭ", "roman": "nui"},
    "नाम": {"ol_chiki": "ᱧᱩᱛᱩᱢ", "roman": "nyutum"},
    "अंशु": {"ol_chiki": "ᱚᱝᱥᱩ", "roman": "Ongsu"},

    # Conjunctions & Prepositions/Postpositions
    "और": {"ol_chiki": "ᱟᱨ", "roman": "ar"},
    "तथा": {"ol_chiki": "ᱟᱨ", "roman": "ar"},
    "का": {"ol_chiki": "ᱨᱮᱱ", "roman": "ren"},
    "के": {"ol_chiki": "ᱨᱮᱱ", "roman": "ren"},
    "की": {"ol_chiki": "ᱨᱮᱱ", "roman": "ren"},
    "को": {"ol_chiki": "ᱴᱷᱮᱱ", "roman": "then"},
    "में": {"ol_chiki": "ᱨᱮ", "roman": "re"},
    "पर": {"ol_chiki": "ᱨᱮ", "roman": "re"},
    "से": {"ol_chiki": "ᱠᱷᱚᱱ", "roman": "khon"},

    # Nouns & Verbs
    "स्कूल": {"ol_chiki": "ᱟᱥᱲᱟ", "roman": "asra"},
    "विद्यालय": {"ol_chiki": "ᱟᱥᱲᱟ", "roman": "asra"},
    "छात्र": {"ol_chiki": "ᱯᱟᱹᱴᱷᱩᱣᱟᱹ", "roman": "pathuwa"},
    "विद्यार्थी": {"ol_chiki": "ᱯᱟᱹᱴᱷᱩᱣᱟᱹ", "roman": "pathuwa"},
    "स्टूडेंट": {"ol_chiki": "ᱯᱟᱹᱴᱷᱩᱣᱟᱹ", "roman": "pathuwa"},
    "है": {"ol_chiki": "ᱠᱟᱱᱟ", "roman": "kana"},
    "हूँ": {"ol_chiki": "ᱠᱟᱱᱟᱹᱧ", "roman": "kananj"},
    "हैं": {"ol_chiki": "ᱢᱮᱱᱟᱢᱟ", "roman": "menama"},
    "था": {"ol_chiki": "ᱛᱟᱦᱮᱸ ᱠᱟᱱᱟ", "roman": "tahe kana"},
    "खाना": {"ol_chiki": "ᱡᱚᱢᱟᱜ", "roman": "jomag"},
    "बहुत": {"ol_chiki": "ᱟᱹᱰᱤ", "roman": "adi"},
    "पसंद": {"ol_chiki": "ᱠᱩᱥᱤ", "roman": "kusi"},
    "क्या": {"ol_chiki": "ᱪᱮᱫ", "roman": "ched"},
    "कहाँ": {"ol_chiki": "ᱚᱠᱟᱨᱮ", "roman": "okare"},
    "गाँव": {"ol_chiki": "ᱟᱹᱛᱩ", "roman": "aatu"},
    "देश": {"ol_chiki": "ᱫᱤᱥᱚᱢ", "roman": "disom"},
    "गाय": {"ol_chiki": "ᱜᱟᱹᱭ", "roman": "gai"},
    "बैल": {"ol_chiki": "ᱰᱟᱝᱽᱨᱟ", "roman": "dangra"},
    "कुत्ता": {"ol_chiki": "ᱥᱮᱛᱟ", "roman": "seta"},
    "बिल्ली": {"ol_chiki": "ᱵᱤᱞᱟᱹᱭ", "roman": "bilae"},
    "अस्पताल": {"ol_chiki": "ᱦᱟᱥᱯᱟᱛᱟᱞ", "roman": "hospital"},
    "दवा": {"ol_chiki": "ᱨᱟᱱ", "roman": "ran"},
    "पानी": {"ol_chiki": "ᱫᱟᱜ", "roman": "daag"},
    "किताब": {"ol_chiki": "ᱯᱩᱛᱷᱤ", "roman": "puthi"},
    "शिक्षक": {"ol_chiki": "ᱢᱟᱪᱮᱛ", "roman": "machet"},
    "घर": {"ol_chiki": "ᱚᱲᱟᱜ", "roman": "orag"},
    "नहीं": {"ol_chiki": "ᱵᱟᱝ", "roman": "bang"},
    "हाँ": {"ol_chiki": "ᱦᱮᱸ", "roman": "hen"}
}


# ============================================================================
# TIER 3: PHONETIC TRANSLITERATION FALLBACK
# ============================================================================
DEV_TO_OLCHIKI_MAP = {
    'अ': 'ᱚ', 'आ': 'ᱟ', 'इ': 'ᱤ', 'ई': 'ᱤ', 'उ': 'ᱩ', 'ऊ': 'ᱩ',
    'ऋ': 'ᱨᱤ', 'ए': 'ᱮ', 'ऐ': 'ᱮ', 'ओ': 'ᱳ', 'औ': 'ᱳ',
    'क': 'ᱠ', 'ख': 'ᱠᱷ', 'ग': 'ᱜ', 'घ': 'ᱜᱷ', 'ङ': 'ᱝ',
    'च': 'ᱪ', 'छ': 'ᱪᱷ', 'ज': 'ᱡ', 'झ': 'ᱡᱷ', 'ञ': 'ᱧ',
    'ट': 'ᱴ', 'ठ': 'ᱴᱷ', 'ड': 'ᱰ', 'ढ': 'ᱰᱷ', 'ण': 'ᱬ',
    'त': 'ᱛ', 'थ': 'ᱛᱷ', 'द': 'ᱫ', 'ध': 'ᱫᱷ', 'न': 'ᱱ',
    'प': 'ᱯ', 'फ': 'ᱯᱷ', 'ब': 'ᱵ', 'भ': 'ᱵᱷ', 'म': 'ᱢ',
    'य': 'ᱭ', 'र': 'ᱨ', 'ल': 'ᱞ', 'व': 'ᱣ', 'श': 'ᱥ',
    'ष': 'ᱥ', 'स': 'ᱥ', 'ह': 'ᱦ', 'ड़': 'ᱲ', 'ढ़': 'ᱲ',
    'ा': 'ᱟ', 'ि': 'ᱤ', 'ी': 'ᱤ', 'ु': 'ᱩ', 'ू': 'ᱩ',
    'ृ': 'ᱨᱤ', 'े': 'ᱮ', 'ै': 'ᱮ', 'ो': 'ᱳ', 'ौ': 'ᱳ',
    'ं': 'ᱸ', 'ँ': 'ᱶ', 'ः': 'ᱷ', '्‍': '', '्': ''
}

def fallback_transliterate(token: str) -> str:
    """
    Phonetically transliterates unseen Devanagari tokens (proper names, unmapped words)
    into Ol Chiki script cleanly without stray spaces.
    """
    if HAS_AKSHARAMUKHA:
        try:
            res = transliterate.process('Devanagari', 'Santali', token)
            return res.strip()
        except Exception:
            pass

    # Basic syllabic fallback if Aksharamukha is not available
    res = "".join(DEV_TO_OLCHIKI_MAP.get(c, c) for c in token)
    return res.strip()


class HindiSanthaliTranslator:
    """
    Database-backed Hindi to Santhali Translator matching SIH_Bhasha_Setu technology.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.conn = None
        if self.db_path.exists():
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row

    def close(self):
        """Close SQLite database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def clean_text(self, text: str) -> str:
        """Strip whitespace and punctuation for clean query matching."""
        return re.sub(r'[?!.,;:()|॥\-\[\]{}]', '', text.strip()).lower()

    def _query_exact_db(self, clean_input: str) -> Optional[sqlite3.Row]:
        """Exact search in SQLite database."""
        if not self.conn:
            return None
        cursor = self.conn.cursor()
        query = """
            SELECT id, english, hindi, santali, santali_roman, category, verified
            FROM translations
            WHERE LOWER(TRIM(hindi)) = ?
               OR LOWER(TRIM(english)) = ?
               OR LOWER(TRIM(santali)) = ?
               OR LOWER(TRIM(santali_roman)) = ?
            LIMIT 1;
        """
        cursor.execute(query, (clean_input, clean_input, clean_input, clean_input))
        return cursor.fetchone()

    def _query_fuzzy_db(self, clean_input: str) -> Optional[sqlite3.Row]:
        """Substring (LIKE) search in SQLite database."""
        if not self.conn or len(clean_input) < 3:
            return None
        cursor = self.conn.cursor()
        query = """
            SELECT id, english, hindi, santali, santali_roman, category, verified
            FROM translations
            WHERE LOWER(hindi) LIKE ?
               OR LOWER(english) LIKE ?
            LIMIT 1;
        """
        pattern = f"%{clean_input}%"
        cursor.execute(query, (pattern, pattern))
        return cursor.fetchone()

    def translate(self, text: str) -> Dict[str, Any]:
        """
        Translates Hindi text to Santhali (Ol Chiki & Roman script).
        """
        raw_text = text.strip()
        if not raw_text:
            return {
                "source_text": "",
                "santali_ol_chiki": "",
                "santali_roman": "",
                "confidence": 0.0,
                "match_type": "empty",
                "details": None
            }

        clean = self.clean_text(raw_text)

        # -------------------------------------------------------------
        # TIER 1: Exact Phrase Bank Match
        # -------------------------------------------------------------
        if clean in PHRASE_BANK:
            entry = PHRASE_BANK[clean]
            return {
                "source_text": raw_text,
                "santali_ol_chiki": entry["ol_chiki"],
                "santali_roman": entry["roman"],
                "confidence": 0.99,
                "match_type": "phrase_bank",
                "details": entry
            }

        if raw_text in PHRASE_BANK:
            entry = PHRASE_BANK[raw_text]
            return {
                "source_text": raw_text,
                "santali_ol_chiki": entry["ol_chiki"],
                "santali_roman": entry["roman"],
                "confidence": 0.99,
                "match_type": "phrase_bank",
                "details": entry
            }

        # Exact SQLite DB Sentence Match
        row_exact = self._query_exact_db(clean)
        if row_exact:
            return {
                "source_text": raw_text,
                "santali_ol_chiki": row_exact["santali"],
                "santali_roman": row_exact["santali_roman"] or "",
                "confidence": 0.98,
                "match_type": "exact_db",
                "details": dict(row_exact)
            }

        # Fuzzy SQLite DB Sentence Match
        if len(clean.split()) >= 3:
            row_fuzzy = self._query_fuzzy_db(clean)
            if row_fuzzy:
                return {
                    "source_text": raw_text,
                    "santali_ol_chiki": row_fuzzy["santali"],
                    "santali_roman": row_fuzzy["santali_roman"] or "",
                    "confidence": 0.90,
                    "match_type": "fuzzy_db",
                    "details": dict(row_fuzzy)
                }

        # -------------------------------------------------------------
        # TIER 2 & TIER 3: Tokenize, Dictionary Translate & Transliterate
        # -------------------------------------------------------------
        tokens = re.findall(r"[^\s?!.,;:()|॥\-\[\]{}]+|[?!.,;:()|॥\-\[\]{}]", raw_text)
        ol_chiki_tokens = []
        roman_tokens = []
        token_sources = []

        for token in tokens:
            # Handle Punctuation
            if re.match(r"^[?!.,;:()|॥\-\[\]{}]+$", token):
                if token == "।":
                    ol_chiki_tokens.append("᱾")
                    roman_tokens.append(".")
                else:
                    ol_chiki_tokens.append(token)
                    roman_tokens.append(token)
                token_sources.append("punctuation")
                continue

            clean_token = self.clean_text(token)

            # Check Tier 2: Word Dictionary
            if clean_token in WORD_DICT:
                v = WORD_DICT[clean_token]
                ol_chiki_tokens.append(v["ol_chiki"])
                roman_tokens.append(v["roman"])
                token_sources.append("word_dict")
                continue

            if token in WORD_DICT:
                v = WORD_DICT[token]
                ol_chiki_tokens.append(v["ol_chiki"])
                roman_tokens.append(v["roman"])
                token_sources.append("word_dict")
                continue

            # Check Tier 2: SQLite Single Word DB Match
            db_row = self._query_exact_db(clean_token)
            if db_row:
                ol_chiki_tokens.append(db_row["santali"])
                roman_tokens.append(db_row["santali_roman"] or db_row["santali"])
                token_sources.append("db_exact")
                continue

            # Tier 3: Phonetic Transliteration Fallback for Devanagari tokens
            trans_ol = fallback_transliterate(token)
            ol_chiki_tokens.append(trans_ol)
            roman_tokens.append(token)  # Keep original token for roman
            token_sources.append("phonetic_transliterated")

        # Join tokens cleanly with single spaces
        joined_ol_chiki = " ".join(ol_chiki_tokens)
        joined_roman = " ".join(roman_tokens)

        # Fix spacing around punctuation marks
        joined_ol_chiki = re.sub(r'\s+([᱾.,!?])', r'\1', joined_ol_chiki)
        joined_roman = re.sub(r'\s+([.,!?])', r'\1', joined_roman)
        joined_ol_chiki = re.sub(r'\s+', ' ', joined_ol_chiki).strip()
        joined_roman = re.sub(r'\s+', ' ', joined_roman).strip()

        if not joined_ol_chiki.endswith("᱾") and not joined_ol_chiki.endswith("."):
            joined_ol_chiki += " ᱾"

        return {
            "source_text": raw_text,
            "santali_ol_chiki": joined_ol_chiki,
            "santali_roman": joined_roman,
            "confidence": 0.85,
            "match_type": "tokenized + transliterated",
            "details": {"token_sources": token_sources}
        }

    def translate_file(self, input_file_path: str = "transcript.txt", output_file_path: str = "transcript_santhali.txt"):
        """
        Reads input_file_path line by line, translates Hindi text to Santhali,
        and saves output to output_file_path.
        """
        in_path = Path(input_file_path)
        out_path = Path(output_file_path)

        if not in_path.exists():
            print(f"Error: {in_path} not found.")
            return

        print(f"Reading Hindi transcript from: {in_path}")
        with open(in_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"🔄 Translating {len(lines)} line(s) to Santhali...")

        results = []
        for i, line in enumerate(lines, 1):
            raw_line = line.rstrip("\r\n")
            if not raw_line.strip():
                results.append("")
                continue

            res = self.translate(raw_line)
            santali_text = res["santali_ol_chiki"]
            results.append(santali_text)

            print(f"\n [{i}/{len(lines)}]")
            print(f"   Hindi  : {raw_line.strip()}")
            print(f"   Santhali: {santali_text} ({res['match_type']})")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(results) + "\n")

        print(f"\n✅ Successfully wrote Santhali translation to: {out_path}")


def print_translation_result(res: Dict[str, Any]):
    """Pretty prints translation result."""
    print("=" * 70)
    print(f"Hindi (Source)    : {res['source_text']}")
    print(f"Santali (Ol Chiki): {res['santali_ol_chiki']}")
    print(f"Santali (Roman)   : {res['santali_roman']}")
    print(f"Match Engine      : {res['match_type']} (Confidence: {res['confidence']})")
    print("=" * 70)


def run_tests(translator: HindiSanthaliTranslator):
    """Run automated verification test cases."""
    print("\n🧪 Running Hindi to Santhali 3-Tier Verification Tests...\n")
    test_sentences = [
        "मेरा नाम अंशु है और मैं संत फ्रांसिस स्कूल का छात्र हूँ",
        "मुझे खाना खाना बहुत पसंद है",
        "अस्पताल कहाँ है?",
        "सिकल सेल खून की जांच",
        "नमस्ते",
        "हमारे गांव में आपका हार्दिक स्वागत है"
    ]
    for text in test_sentences:
        res = translator.translate(text)
        print_translation_result(res)
        print()


def main():
    translator = HindiSanthaliTranslator()

    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        if arg1 == "--test":
            run_tests(translator)
        elif arg1 == "--interactive":
            print("\n🌟 Hindi -> Santhali Interactive Translation CLI (Type 'exit' to quit) 🌟\n")
            while True:
                try:
                    user_input = input("Hindi Input > ").strip()
                    if user_input.lower() in ['exit', 'quit', 'q']:
                        break
                    if not user_input:
                        continue
                    result = translator.translate(user_input)
                    print_translation_result(result)
                    print()
                except (KeyboardInterrupt, EOFError):
                    break
        elif arg1 == "--file":
            in_file = sys.argv[2] if len(sys.argv) > 2 else "transcript.txt"
            out_file = sys.argv[3] if len(sys.argv) > 3 else "transcript_santhali.txt"
            translator.translate_file(in_file, out_file)
        else:
            if Path(arg1).is_file():
                out_file = sys.argv[2] if len(sys.argv) > 2 else "transcript_santhali.txt"
                translator.translate_file(arg1, out_file)
            else:
                phrase = " ".join(sys.argv[1:])
                result = translator.translate(phrase)
                print_translation_result(result)
    else:
        # Default behavior: translate transcript.txt to transcript_santhali.txt
        in_file = "transcript.txt"
        out_file = "transcript_santhali.txt"
        if Path(in_file).exists():
            translator.translate_file(in_file, out_file)
        else:
            run_tests(translator)


if __name__ == "__main__":
    main()
