# 0016 — Ten billion tokens: a new data set on other commodity hardware

Status: shelves built (2026-11-15), run starting (2026-11-16) · Depends on: 0014, 0015 · Unblocks: v7, 0013

## 1. TL;DR
The student has read ~2.4B tokens in all, ~20 per parameter; small models of its size read hundreds of billions. We
train **ten billion tokens** in one run of several days, non-stop, on other commodity hardware (DiLoCo, 0015), from a
new data set built where the training runs: the harder half of FineWeb-Edu the first two billions left out, and
Cosmopedia's synthetic textbooks and stories for older readers.

## 2. Abstract
0014 fed the M6 from shelves built on the M6 and copied to every machine that trained. Ten billion tokens need ~14B
tokens of text so that little is read twice: ~28 GB of token files, too much to build on the M6 (41 GB free) and copy
over a VPN. The data is now downloaded and built on the training machine itself, and only the token files go to the
other machines over their LAN. The run starts from `bil2-avg/final` (0015: the second billion, level with v6).

## 3. Objective
- **Download** (`scripts/download_library_10b.sh`, on the training machine): FineWeb-Edu's 14 shards of the 10BT
  sample; Cosmopedia's stories (all 43 files), stanford, openstax, khanacademy, wikihow, auto_math_text and 24 of the
  118 files of web_samples_v2 — ~60 GB of parquet.
- **Shelves** (`config/library.yaml`, 16k tokenizer):

  | shelf | what | planned | built |
  |---|---|---|---|
  | `s4_edu_12` | FineWeb-Edu, edu score ≥ 3, paragraphs of grade 8–12 (none shared with `s3_edu_plus`/`s3_edu_8`) | ~4B | 2.41B |
  | `s4_textbooks` | Cosmopedia textbooks and lessons, paragraphs up to grade 12 | ~5B | 0.88B |
  | `s4_stories` | Cosmopedia stories for every audience but young children, up to grade 12 | ~3B | 1.12B |
  | earlier shelves | edu, edu 8, textbooks, dialogue, stories, speech, books, wiki (0010, 0014) | ~2B | 1.87B |
  | **all** | | **~14B** | **6.28B** |

- **Arm `bil10`** (`final_test.py`): from `checkpoints/final/bil2-avg/final`, **10B tokens**, three of four steps
  shelves — new 74% (edu 12 40%, stories 19%, textbooks 15%), earlier 26% — the fourth step talk2's conversations so
  the student keeps answering. Learning rate 1e-4 after 2,000 steps, cosine to 1e-5; validation on the 0014 shelves and
  the three new ones. Shares follow the built shelves, so each is read ~1.25 times.
- **Time:** ~3.5 days non-stop; DiLoCo with plain averaging, H = 500, checkpoints every 30 minutes.
- **Then:** conversation tuning as talk2, bits per byte and the spot check against v6 → v7.

**Out of scope:** a larger model (10B tokens are ~90 per parameter: still the data, not the size, is the limit),
new tokenizer.

## 4. Control Gates
| Gate | What | Pass |
|---|---|---|
| G1 | Shelves | ≥ 4B tokens of new text; 0 exam leaks; no paragraph shared with the earlier edu shelves |
| G2 | Run, 500M | bits per byte on edu and textbooks not above the start by > 0.05 |
| G3 | Result | after tuning: bits per byte below v6 on every shelf; spot check not worse than v6 |

## 5. Additional Information
- **Why grade 8–12 now:** the school story keeps the early stages simple, but 0010 and 0014 showed knowledge is
  what the student lacks; the paragraphs above grade 8 are most of FineWeb-Edu and most of its facts.
- **Smaller than planned:** the grade-12 limit on paragraphs keeps less of Cosmopedia than estimated (textbooks
  0.88B of ~5B: most of their paragraphs read above grade 12). Ten billion tokens therefore read every shelf ~1.25
  times, still far below the ~4 passes where repetition starts to cost (Muennighoff et al., 2023).
