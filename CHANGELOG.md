# Changelog

What has actually been delivered. Intent of changes: `specs/`. Proof: tests.

## [Unreleased]

### Added
- CDD project structure: `README.md`, `MANIFESTO.md`, `CHANGELOG.md`, specifications 0001–0007 and the source document in `specs/`.
- `scripts/download_library.sh` — resumable download of library shelves into `data/library/` (outside git) — [0001].
- Project name: Cerbeto; specification refinements (fp32/fp16 precision without bf16, default teacher `nemotron-3.5-bolt:e4b`, nursery time budget on M1 Pro).
- Python brain project `brain/` (uv, Python 3.12, package `cerbeto`) with CLI `cerbeto library build|tokenize|report` and `cerbeto tokenizer train` — [0001].
- School library shelves built offline into `data/shelves/` (JSONL text + uint16 `.bin` tokens + `manifest.json` with licences): `s0_speech` 83.6M, `s0_stories` 544.5M, `s1_books` 37.3M, `s2_world` 0.8M, `s3_edu` 4.4M training tokens — [0001].
- Librarian: Flesch-Kincaid grade filter, clean-prose filter, paragraph/document deduplication, deterministic 0.5% validation split — [0001].
- Own byte-level BPE tokenizer, 8192 tokens incl. 9 special tokens (`brain/cerbeto/tokenizer/artifacts/tokenizer.json`); 1.241 tokens/word on TinyStories valid — [0001].
- Gate tests G1–G4, N1, N2 (27 tests, `uv run pytest` in `brain/`) — [0001].
- Student model `cerbeto.model.student`: transformer with prelude / recurrent core × K / coda, RoPE, SwiGLU, RMSNorm, tied embeddings, sampling; starts at 6.9M parameters (`config/student.yaml`) — [0002].
- Function-preserving growth operators (`grow_depth`, `grow_width`, `grow_model_dim`), MLP pruning, neuron renewal (continual backprop), optimizer-state transplant, growth log, 30M hard limit — [0002].
- EWC synaptic consolidation, safetensors checkpoints with architecture and optimizer state, activation checkpointing — [0002].
- `cerbeto train --smoke`: 30-min fp32 run on TinyStories cut validation loss 9.08 → 1.75 (−81%) at ~14k tokens/s; fp16 was 4.5% slower → fp32 stays — [0002].
- Gate tests for 0002 (G1–G3, G5, N1, N2): 53 tests; 30M student peaks at 3.05 GB with checkpointing — [0002].
- Specs 0003/0004: social skills training (conversation roles in nursery speech, polite routines, agree/disagree, opinions, honest "I don't know") — spec change only.
- Speech shelf rebuilt with conversation roles (CHILDES child → `<|student|>`, adults → `<|human|>`; Switchboard B/A) — [0003].
- Nursery training loop `cerbeto nursery run`: phased data mix (speech → speech + stories), gradient accumulation, growth on validation plateau along a fixed schedule, versioned checkpoints every 30 min, exact resume — [0003].
- Nursery exams `cerbeto exam nursery`: BLiMP (BabyLM filtered, 67 phenomena) and a 50-situation conversation probe graded by the local teacher; minimal Ollama teacher client — [0003].
- License: PolyForm Noncommercial 1.0.0 (`LICENSE.md`) — noncommercial use, modification and forks allowed; commercial use requires the author's permission.
- Nursery growth deadlines (60/120/180/240M tokens), learning-rate re-warm segment after each growth, 32 validation batches per shelf; nursery restarted (attempt 2) — [0003].
