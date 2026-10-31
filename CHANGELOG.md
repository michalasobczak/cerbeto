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
