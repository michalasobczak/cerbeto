# 0014 — A second billion tokens: new edu text, the 16k vocabulary

Status: in progress (2026-11-12) — shelf `s3_edu_8` and arm `bil2` written; queued on the M6 after `vext` · Depends on: 0010, 0011, 0012 · Unblocks: v7, 0013

## 1. TL;DR
v6 knows little of the world because it has read little: ~1B tokens for 107M parameters (~10 per parameter; SmolLM-135M
read 600B). We train **a second billion tokens** from `v16k/final` (v6 on the 16k vocabulary, whose new words 60M tokens
could not teach), mostly on text the first billion never reached, then tune conversations again as talk2.

## 2. Abstract
After the 1B run the shelves hold ~1.62B tokens (8k) and the run read only part of them: edu ~40%, stories ~20%,
dialogue ~45%, textbooks ~70%, speech ~60%, books ~90%, Wikipedia ~1.6×. FineWeb-Edu's 14 shards, already on the M6,
give ~1.06B tokens at grade ≤ 8 against ~600M at ≤ 7 (0010 measurement), so a new shelf of the grade 7–8 paragraphs adds
~470M tokens of new text without a download (the user chose this over SmolLM-Corpus, Cosmopedia v2 or full Wikipedia,
2026-11-12). Spec 0012 showed the 16k vocabulary loses in 60M tokens only on its new, rarer words; a billion tokens
teach them, so the vocabulary change costs nothing extra here.

## 3. Objective
- **Shelf `s3_edu_8`** (`config/library.yaml`): FineWeb-Edu, edu score ≥ 3, clean prose, paragraphs of grade
  7 < FK ≤ 8 (`fk_min`), so it shares no paragraph with `s3_edu_plus`; a document whose grade ≤ 7 paragraphs are in
  `s3_edu_plus`'s validation is left out (`exclude_valid_of`), so v6's edu validation stays unseen. Exam-leak filter on.
- **Arm `bil2`** (`final_test.py`): from `checkpoints/final/v16k/final`, 1B tokens, 16k bins.
  - three of four steps shelves: edu 8 30%, edu 22%, textbooks 12%, dialogue 12%, stories 10%, speech 6%, books 5%,
    wiki 3% (no shelf above 4 passes, as 0010);
  - the fourth step talk2's conversations (teacher questions, memory, dialogues, …), so the form of answers stays;
  - learning rate 1e-4 after 1,000 warm-up steps, cosine to 1e-5; K = 1; validation on the 0010 shelves and edu 8.
- **Then:** conversation tuning as talk2 (3e-5, every other step the shelves, ~50M tokens), bits per byte and the spot
  check against v6.

**Out of scope:** new downloads, a larger model, 0013 (it follows on the result).

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `shelves.py`: `fk_min`, `exclude_valid_of` | system | created |
| `config/library.yaml`: `s3_edu_8` | config | created |
| `final_test.py`: arm `bil2` | system | created |
| `scripts/final/bil2.sh` (wait for vext, build, tokenize, train) | system | created |
| panel tab *1Bt run #2* | system | created |

## 5. Critical Path
1. `vext` ends (~23:40) → 2. build `s3_edu_8` and cut it with the 16k tokenizer → 3. `final_test train bil2 --tokens 1e9`
(~4.5 days at ~2,600 tokens/s) → 4. tuning as talk2 → 5. bits per byte and spot check against v6 → v7.

## 6. Control Gates
| Gate | What | Pass |
|---|---|---|
| G1 | Shelf | ≥ 300M tokens; 0 exam leaks; no paragraph shared with `s3_edu_plus` |
| G2 | Run | after 50M tokens shelf validation (bits per byte) not above the start by > 0.05 |
| G3 | Result | after tuning: bits per byte below v6 on edu and textbooks; spot check chat, completing, memory not worse than v6 by > 5 / 5 / 3, completing better |

## 7. Expected Outcome
Edu and textbook validation clearly below v6 (more knowledge, the 16k words learnt), completing in the spot check up,
chat at v6's level after tuning.

## 8. Additional Information
- **The learning rate (2026-11-12):** 0011 showed 1e-4 knocks a student out of its minimum within a few million tokens;
  over a billion tokens of the same kind of text that is the usual price of a warm restart, paid back long before the end,
  and the shelves the student already read keep ~40% of the mix.
