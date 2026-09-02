# Project History & Inference Pipeline Record

## Overview
This document records the exact configuration, architecture, tokenizer setup, checkpoint metadata, and execution details for the Hindi to Santali Translation inference pipeline.

---

## 1. File Structure & Relative Paths
All paths are defined relative to `hindi_santhali_text.py` using `pathlib.Path`:
- **Base Directory**: `BASE = Path(__file__).resolve().parent`
- **Checkpoint Path**: `checkpoints/best_model (3).pt`
- **Tokenizer Directory**: `tokenizer/`
  - **Hindi Model**: `tokenizer/hindi.model`
  - **Santali Model**: `tokenizer/santali.model`
- **Input Transcript**: `transcript.txt`
- **Output Transcript**: `transcript_santhali.txt`

---

## 2. Checkpoint Details
- **File**: `checkpoints/best_model (3).pt`
- **Trained Epoch**: 35
- **Train Loss**: 4.855815512891865
- **Dev (Val) Loss**: 5.035829884795341
- **Tensor Count**: 130 tensors in `model_state_dict`
- **Parameter Count**: 19,677,824 parameters
- **State Dict Load Status**: 0 Missing Keys, 0 Unexpected Keys

---

## 3. Model Architecture Specs (`HindiSantaliTransformer`)
- **Source Vocabulary Size**: 16,000 (Hindi SentencePiece)
- **Target Vocabulary Size**: 16,000 (Santali SentencePiece)
- **d_model**: 256
- **nhead**: 4
- **num_encoder_layers**: 4
- **num_decoder_layers**: 4
- **dim_feedforward**: 1024
- **dropout**: 0.1
- **Special Token IDs**:
  - `PAD_ID`: 0
  - `BOS_ID`: 2
  - `EOS_ID`: 3
- **Max Sequence Length (`MAX_LEN`)**: 128
- **Transformer Config**: `batch_first=True`, `norm_first=True`

---

## 4. Decoding & Inference Strategy
1. **Tokenization**: Input Hindi sentences from `transcript.txt` are tokenized using `hindi.model`.
2. **Sequence Formatting**: Tokens are prefixed with `[BOS_ID]` (2) and suffixed with `[EOS_ID]` (3), truncated to `MAX_LEN=128`.
3. **Tensor Dimensions**: Inputs are passed as 2D tensors of shape `[1, sequence_length]` (`batch_first=True`).
4. **Greedy Decoding**: Decoder starts with `[BOS_ID]` (2). In each iteration, `model(src, tgt)` generates output logits. `torch.argmax(logits[:, -1, :], dim=-1)` picks the highest probability next token. Decoding stops when `EOS_ID` (3) is produced or `MAX_LEN=128` is reached.
5. **Detokenization**: Tokens are decoded to Santali text using `santali.model`.
6. **Output Storage**: Results are saved line-by-line to `transcript_santhali.txt` with UTF-8 encoding.

---

## 5. Execution Summary & Results
- **Script File**: `hindi_santhali_text.py` (and `inference.py`)
- **Execution Test**: Passed successfully on CPU / CUDA with zero warnings and full state dict alignment.
- **Output File Created**: `transcript_santhali.txt` generated automatically upon execution.

---

## 6. Troubleshooting Notes
- **Interactive Python REPL Conflict**: If `python.exe` is run without script arguments in PowerShell, it enters Python interactive mode (`>>>`). Executing terminal commands or pasting raw text inside `>>>` causes `SyntaxError: invalid syntax` or `SyntaxError: invalid character`. Always execute script commands directly from the PowerShell command prompt (`PS C:\...>`), not inside `>>>`.
- **Windows Console Unicode Output**: Added `sys.stdout.reconfigure(encoding="utf-8")` and `sys.stderr.reconfigure(encoding="utf-8")` to prevent Windows `cp1252` encoding errors when printing checkmark characters (`✓`) or Devanagari/Ol Chiki scripts.
