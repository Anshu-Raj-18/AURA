import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import sentencepiece as spm

# ============================================================
# PATHS
# ============================================================

try:
    from . import config
except ImportError:
    import config

BASE = config.BASE_DIR

CHECKPOINT = BASE / "checkpoints" / "best_model (3).pt"

TOKENIZER_DIR = BASE / "tokenizer"
HINDI_MODEL = TOKENIZER_DIR / "hindi.model"
SANTALI_MODEL = TOKENIZER_DIR / "santali.model"

INPUT_FILE = config.HINDI_TRANSCRIPT_PATH
OUTPUT_FILE = config.SANTALI_TRANSCRIPT_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL CONFIGURATION
# ============================================================

SRC_VOCAB_SIZE = 16000
TGT_VOCAB_SIZE = 16000

D_MODEL = 256
NHEAD = 4

NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4

DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3

MAX_LEN = 128


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================
# EXACT TRAINING MODEL
# ============================================================

class HindiSantaliTransformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=128
    ):
        super().__init__()

        self.d_model = d_model

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=PAD_ID
        )

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model,
            padding_idx=PAD_ID
        )

        self.src_positional = PositionalEncoding(
            d_model,
            max_len,
            dropout
        )

        self.tgt_positional = PositionalEncoding(
            d_model,
            max_len,
            dropout
        )

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.output_projection = nn.Linear(
            d_model,
            tgt_vocab_size
        )

    def generate_square_subsequent_mask(self, size, device):

        mask = torch.triu(
            torch.ones(size, size, device=device),
            diagonal=1
        )

        mask = mask.masked_fill(
            mask == 1,
            float("-inf")
        )

        return mask

    def forward(
        self,
        src,
        tgt,
        src_padding_mask=None,
        tgt_padding_mask=None
    ):

        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)

        src_emb = src_emb * math.sqrt(self.d_model)
        tgt_emb = tgt_emb * math.sqrt(self.d_model)

        src_emb = self.src_positional(src_emb)
        tgt_emb = self.tgt_positional(tgt_emb)

        tgt_mask = self.generate_square_subsequent_mask(
            tgt.size(1),
            tgt.device
        )

        output = self.transformer(
            src_emb,
            tgt_emb,

            tgt_mask=tgt_mask,

            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,

            memory_key_padding_mask=src_padding_mask
        )

        output = self.output_projection(output)

        return output


# ============================================================
# LOAD TOKENIZERS
# ============================================================

print("=" * 70)
print("HINDI -> SANTALI TRANSCRIPT TRANSLATION")
print("=" * 70)

print("Device:", DEVICE)
print()

for p in [
    CHECKPOINT,
    HINDI_MODEL,
    SANTALI_MODEL,
    INPUT_FILE
]:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

    print("✓ Found:", p)

print()

hindi_tokenizer = spm.SentencePieceProcessor()
santali_tokenizer = spm.SentencePieceProcessor()

hindi_tokenizer.load(str(HINDI_MODEL))
santali_tokenizer.load(str(SANTALI_MODEL))

print("Hindi vocab size  :", hindi_tokenizer.get_piece_size())
print("Santali vocab size:", santali_tokenizer.get_piece_size())


# ============================================================
# CREATE EXACT MODEL
# ============================================================

model = HindiSantaliTransformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,

    d_model=D_MODEL,
    nhead=NHEAD,

    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,

    dim_feedforward=DIM_FEEDFORWARD,

    dropout=DROPOUT,
    max_len=MAX_LEN
).to(DEVICE)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("=" * 70)
print("LOADING CHECKPOINT")
print("=" * 70)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("✓ Checkpoint loaded")
print("Epoch:", checkpoint.get("epoch"))
print("Train loss:", checkpoint.get("train_loss"))
print("Dev loss:", checkpoint.get("val_loss"))

print()
print("✓ Model ready")


# ============================================================
# GREEDY TRANSLATION
# ============================================================

@torch.no_grad()
def translate_sentence(hindi_text):

    # --------------------------------------------------------
    # Hindi text -> Hindi SentencePiece IDs
    # --------------------------------------------------------

    src_ids = hindi_tokenizer.encode(
        hindi_text,
        out_type=int
    )

    # Add BOS and EOS
    src_ids = [BOS_ID] + src_ids + [EOS_ID]

    # Limit source length
    src_ids = src_ids[:MAX_LEN]

    # Ensure EOS if truncation happened
    if src_ids[-1] != EOS_ID:
        src_ids[-1] = EOS_ID

    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=DEVICE
    )

    # --------------------------------------------------------
    # GREEDY DECODING
    # --------------------------------------------------------

    generated = [BOS_ID]

    for _ in range(MAX_LEN - 1):

        tgt = torch.tensor(
            [generated],
            dtype=torch.long,
            device=DEVICE
        )

        src_padding_mask = (src == PAD_ID)
        tgt_padding_mask = (tgt == PAD_ID)

        output = model(
            src,
            tgt,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask
        )

        # Last decoder position
        logits = output[:, -1, :]

        # Greedy: choose highest probability token
        next_token = torch.argmax(
            logits,
            dim=-1
        ).item()

        generated.append(next_token)

        if next_token == EOS_ID:
            break

    # --------------------------------------------------------
    # Santali IDs -> Santali text
    # --------------------------------------------------------

    # Remove BOS/EOS
    output_ids = generated[1:]

    if EOS_ID in output_ids:
        output_ids = output_ids[:output_ids.index(EOS_ID)]

    if len(output_ids) == 0:
        return ""

    santali_text = santali_tokenizer.decode(
        output_ids
    )

    return santali_text


# ============================================================
# READ TRANSCRIPT
# ============================================================

print()
print("=" * 70)
print("READING TRANSCRIPT")
print("=" * 70)

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    lines = f.readlines()

print("Total lines:", len(lines))


# ============================================================
# TRANSLATE TRANSCRIPT
# ============================================================

translated_lines = []

print()
print("=" * 70)
print("TRANSLATING")
print("=" * 70)

for i, line in enumerate(lines, start=1):

    hindi_text = line.strip()

    # Preserve empty lines
    if not hindi_text:
        translated_lines.append("")
        continue

    try:

        santali_text = translate_sentence(
            hindi_text
        )

        translated_lines.append(
            santali_text
        )

        print()
        print(f"[{i}/{len(lines)}]")
        print("Hindi  :", hindi_text)
        print("Santali:", santali_text)

    except Exception as e:

        print()
        print(f"[{i}/{len(lines)}] ERROR:", e)

        # Keep an empty output for failed line
        translated_lines.append("")


# ============================================================
# SAVE OUTPUT
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for line in translated_lines:
        f.write(line + "\n")


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("TRANSLATION COMPLETE")
print("=" * 70)

print("Input :", INPUT_FILE)
print("Output:", OUTPUT_FILE)

print()
print("✓ Hindi transcript read from:")
print("  transcript.txt")

print()
print("✓ Santali translation saved to:")
print("  transcript_santhali.txt")
print("=" * 70)