# 0001 — School library and tokenizer

Status: delivered (G5 awaiting human review) · Depends on: — · Unblocks: 0002, 0003

## 1. TL;DR
We download an offline "school library" (BabyLM, TinyStories, Simple English Wikipedia, FineWeb-Edu), split it into shelves by difficulty level and train our own BPE tokenizer (8192 tokens). The result is confirmed by the library report (shelves, token counts, licences) and by the tokenizer tests.

## 2. Abstract
The project starts from an empty directory. The student is to learn "like a human at school, step by step", and stage 0 (nursery) requires 200–300 million tokens of simple English. A local teacher (30–60 tok/s) cannot generate such an amount in reasonable time, so the nursery relies on ready-made corpora with a child-oriented profile. Our own web scraping was considered — rejected ("adult language" quality, legal risk when distributing the model, cost of building the pipeline); instead we use the ready-made, cleaned FineWeb-Edu filtered down to the grade 1–3 level. The tokenizer must be our own (manifesto §1) and small, because at 8–30M parameters the embeddings with a 32k vocabulary would eat half of the budget.

## 3. Objective
**Goals**
- Library shelves available offline in `data/library/` (outside git), downloaded by the script `scripts/download_library.sh`.
- Every shelf has a manifest: source, licence, size, word/token count, difficulty level.
- "Librarian": text difficulty assessment (Flesch-Kincaid Grade Level) and assignment of fragments to the shelves of stages 0–3.
- Our own BPE tokenizer (byte-level, 8192 tokens + special tokens for roles and reasoning).
- Token sets ready for training (`uint16`, memmap) with a train/valid split.

**Out of scope**
- School task generators (0004), the LLM teacher (0005), our own web crawler.
- Gutenberg/Wikijunior as separate downloads — Gutenberg and Simple Wiki are already in BabyLM; Wikijunior [TO BE DETERMINED] if needed.

**Success criteria**
- ≥ 250 million tokens on the stage 0 (nursery) shelves after tokenization.
- Average length in tokens per English word ≤ 1.4 on the TinyStories validation set.
- 100% of shelves have a manifest with a licence.

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `scripts/download_library.sh` | system | exists |
| `data/library/babylm/` — BabyLM-2026-Strict (CHILDES, BNC spoken, Switchboard, OpenSubtitles, children's Gutenberg, Simple Wiki), ~100M words, MIT | data | exists (downloading) |
| `data/library/tinystories/` — TinyStoriesV2-GPT4 train/valid, CDLA-Sharing-1.0 | data | exists (downloading) |
| `data/library/simplewiki/` — Wikipedia simple 20231101 (parquet), CC BY-SA | data | exists (downloading) |
| `data/library/fineweb_edu/` — 1 shard sample-10BT (~2 GB), ODC-By | data | exists (downloading) |
| `brain/cerbeto/library/` — shelf manifests, librarian (readability), shelf building | system | to create |
| `brain/cerbeto/tokenizer/` — BPE training and loading | system | to create |
| `config/library.yaml` — definition of shelves and difficulty thresholds | data | to create |
| `data/shelves/*.bin` + `manifest.json` | data | to create |
| Python project `brain/` (uv, Python 3.12, PyTorch) | system | to create |

## 5. Functional Domain Changes
**Current state:** no data and no tokenizer.

**Target state — shelves:**
| Shelf | Contents | Stage |
|---|---|---|
| `s0_speech` | CHILDES, Switchboard, BNC spoken, OpenSubtitles | 0 — nursery |
| `s0_stories` | TinyStories V2 | 0 — nursery / 1 — preschool |
| `s1_books` | children's Gutenberg (BabyLM) | 1–2 |
| `s2_world` | Simple Wiki (BabyLM + full dump), fragments FK ≤ 4 | 2–3 |
| `s3_edu` | FineWeb-Edu, fragments FK ≤ 4 and edu score ≥ 3 | 3 |

**Rules:**
- A fragment (paragraph) goes onto a shelf only if its FK level is within the shelf threshold; the rest is discarded and counted in the report.
- Exact deduplication (hash of the normalized paragraph) across shelves.
- The tokenizer is trained on a sample balanced across shelves (so as not to overfit to movie subtitles).
- Special tokens: `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|teacher|>`, `<|student|>`, `<|human|>`, `<|think|>`, `<|/think|>`, `<|answer|>`.

**Resilience:** downloading is resumable (`curl -C -`); shelf building is idempotent (manifest with input hashes — a re-run skips what is already done).

**Data:** raw ~5 GB; shelves after tokenization ~0.6–1 GB.

## 6. Critical Path
1. Download the shelves (script) → 2. `brain/` project with uv → 3. librarian + building text shelves → 4. tokenizer training → 5. tokenization to `.bin` → 6. library report.

**Completion condition:** `uv run cerbeto library report` prints all shelves with licence and token count, and the stage 0 total is ≥ 250 million.

## 7. Control Gates
| # | Type | Pass condition | Verification | Blocks |
|---|---|---|---|---|
| G1 | Unit | Tokenizer: `decode(encode(x)) == x` for 10,000 random paragraphs | `pytest tests/test_tokenizer.py` | 0002 |
| G2 | Unit | Librarian: FK of known reference texts within a tolerance of ±0.5 | `pytest tests/test_librarian.py` | shelf building |
| G3 | Integration | Every shelf has a `manifest.json` with the fields `source, license, tokens, fk_max` | `pytest tests/test_shelves.py` | 0003 |
| G4 | Integration | Stage 0 tokens ≥ 250 million; tokens/word ≤ 1.4 | `cerbeto library report` | 0003 |
| G5 | Manual | 20 random fragments from each shelf readable for a ~8-year-old | human review | 0003 |
| N1 | Negative | No file from `data/` is tracked by git | `git ls-files data/` returns empty | commit |
| N2 | Negative | The tokenizer contains no tokens outside the 8192 vocabulary + special tokens | test G1 | 0002 |

## 8. Expected Outcome
- **Direct:** library and tokenizer ready offline.
- **Measurable:** library report (tokens per shelf, discarded fragments, licences).
- **Side effects:** ~6 GB of disk usage in `data/`.
- **Rollback:** delete `data/shelves/` and `brain/cerbeto/tokenizer/artifacts/`; raw data may stay.

## 9. Additional Information
- **Rejected alternatives:** our own web scraping; generating the nursery with the local teacher (months); 32k tokenizer (parameter cost).
- **Technical debt:**
  - FK is an approximate measure for dialogues (CHILDES) — the FK threshold is not applied to the speech shelves.
  - The librarian's prose filter is heuristic (sentence ending, ≥ 90% letters, no markup/links/citations).
  - `s2_world` is small (~0.5M words) because Simple Wikipedia rarely reaches FK ≤ 4 and is dominated by stub articles ("X is a commune in…"); `s3_edu` (FineWeb-Edu) is not detoxified like BabyLM. Both are stage 2–3 shelves — revisit thresholds and add a content filter before 0005.
- **External dependencies:** Hugging Face (one-time download); the `tokenizers` library is used only as the BPE *training algorithm* — the vocabulary and merges are learned from scratch on our shelves (manifesto §1).
- **Sources:** BabyLM Challenge (Findings, arXiv 2504.08165); TinyStories (Eldan & Li, 2023); FineWeb-Edu (Penedo et al., 2024).
- **Open questions:** whether to add Wikijunior as a shelf 1–2.
