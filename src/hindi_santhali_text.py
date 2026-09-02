import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import sentencepiece as spm

# ============================================================
# CONSTANTS & CONFIGURATION
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

try:
    from . import config
except ImportError:
    import config

BASE = config.BASE_DIR

CHECKPOINT_PATH = BASE / "checkpoints" / "best_model (3).pt"
HINDI_MODEL = BASE / "tokenizer" / "hindi.model"
SANTALI_MODEL = BASE / "tokenizer" / "santali.model"

INPUT_FILE = config.HINDI_TRANSCRIPT_PATH
OUTPUT_FILE = config.SANTALI_TRANSCRIPT_PATH


# ============================================================
# MODEL ARCHITECTURE (MATCHING TRAINING NOTEBOOK)
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


class HindiSantaliTransformer(nn.Module):

    def __init__(
        self,
        src_vocab_size=16000,
        tgt_vocab_size=16000,
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
# TRANSLATION INFERENCE FUNCTION
# ============================================================

@torch.no_grad()
def translate_sentence(model, hindi_tokenizer, santali_tokenizer, text, device, max_len=128):
    text = text.strip()
    if not text:
        return ""

    # 1. Tokenize Hindi input
    src_tokens = hindi_tokenizer.encode(text, out_type=int)

    # 2. Add BOS and EOS tokens
    src_ids = [BOS_ID] + src_tokens + [EOS_ID]
    src_ids = src_ids[:max_len]
    if src_ids[-1] != EOS_ID:
        src_ids[-1] = EOS_ID

    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    # 3. Autoregressive Greedy Decoding
    generated = [BOS_ID]

    for _ in range(max_len - 1):
        tgt = torch.tensor([generated], dtype=torch.long, device=device)

        # Forward pass matching exact training setup
        output = model(src, tgt)

        logits = output[0, -1, :]
        next_token = torch.argmax(logits, dim=-1).item()

        if next_token == EOS_ID:
            break

        generated.append(next_token)

    # 4. Decode to Santali text
    output_ids = generated[1:]  # remove BOS
    santali_text = santali_tokenizer.decode(output_ids)
    return santali_text.strip()


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("HINDI -> SANTALI TRANSCRIPT TRANSLATION")
    print("=" * 70)
    print(f"\nDevice: {device.type}\n")

    # Check required files
    missing_files = []
    if not CHECKPOINT_PATH.exists():
        missing_files.append(f"Checkpoint: {CHECKPOINT_PATH}")
    if not HINDI_MODEL.exists():
        missing_files.append(f"Hindi Tokenizer: {HINDI_MODEL}")
    if not SANTALI_MODEL.exists():
        missing_files.append(f"Santali Tokenizer: {SANTALI_MODEL}")
    if not INPUT_FILE.exists():
        missing_files.append(f"Input Transcript: {INPUT_FILE}")

    if missing_files:
        print("ERROR: Missing required files:")
        for mf in missing_files:
            print(f"  - {mf}")
        sys.exit(1)

    print(f"✓ Checkpoint found: {CHECKPOINT_PATH.name}")
    print(f"✓ Hindi tokenizer found: {HINDI_MODEL.name}")
    print(f"✓ Santali tokenizer found: {SANTALI_MODEL.name}")
    print(f"✓ Transcript found: {INPUT_FILE.name}")
    print()

    # Load Tokenizers
    hindi_tokenizer = spm.SentencePieceProcessor()
    santali_tokenizer = spm.SentencePieceProcessor()

    hindi_tokenizer.load(str(HINDI_MODEL))
    santali_tokenizer.load(str(SANTALI_MODEL))

    print(f"Hindi vocab size  : {hindi_tokenizer.get_piece_size()}")
    print(f"Santali vocab size: {santali_tokenizer.get_piece_size()}")
    print(f"PAD ID: {hindi_tokenizer.pad_id()} | BOS ID: {hindi_tokenizer.bos_id()} | EOS ID: {hindi_tokenizer.eos_id()}")
    print()

    # Load Model & Checkpoint
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
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())

    print("=" * 70)
    print("MODEL CHECKPOINT LOADED")
    print("=" * 70)
    print(f"Architecture    : HindiSantaliTransformer")
    print(f"Parameters      : {num_params:,}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Train loss      : {checkpoint.get('train_loss'):.4f}")
    print(f"Dev loss        : {checkpoint.get('val_loss'):.4f}")
    print("✓ Strict loading passed (strict=True)")
    print("✓ MODEL READY")
    print()

    # Read Input Transcript
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Perform Translation
    print("=" * 70)
    print("TRANSLATING TRANSCRIPT")
    print("=" * 70)
    print()

    translated_lines = []
    total_lines = len(lines)

    for i, line in enumerate(lines, 1):
        raw_line = line.rstrip("\r\n")
        hindi_text = raw_line.strip()

        if not hindi_text:
            translated_lines.append("")
            print(f"[{i}/{total_lines}] (empty line preserved)")
            continue

        try:
            santali_text = translate_sentence(
                model,
                hindi_tokenizer,
                santali_tokenizer,
                hindi_text,
                device,
                max_len=MAX_LEN
            )
        except Exception as e:
            print(f"[{i}/{total_lines}] ERROR translating line '{hindi_text}': {e}")
            raise e

        translated_lines.append(santali_text)

        print(f"[{i}/{total_lines}]")
        print(f"Hindi:\n{hindi_text}")
        print(f"Santali:\n{santali_text}\n")

    # Save Output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(translated_lines) + "\n")

    print("=" * 70)
    print("TRANSLATION COMPLETE")
    print("=" * 70)
    print(f"Input : {INPUT_FILE.name}")
    print(f"Output: {OUTPUT_FILE.name}")
    print("✓ Result saved successfully.")
    print()

if __name__ == "__main__":
    main()