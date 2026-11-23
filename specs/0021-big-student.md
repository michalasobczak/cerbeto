# 0021 — A larger student (~290M) grown from v20, with basic facts in its data

Status: in progress (2026-11-23): data generating, growth checked · Depends on: 0016, 0020 · Unblocks: a release

## 1. TL;DR
v20 (`facts2`) is at its ceiling: more fact pairs and continued tuning leave knowledge at ~120 of 280. pass@8 shows
little hidden knowledge (156), and most of what it never answers is not in our pairs at all. We add **basic facts**
(question pairs from the opening of the most mentioned Simple Wikipedia articles) and **grow v20 to 287.7M**
function-preservingly, then train it on the shelves with the fact pairs in the stream from the start.

## 2. Abstract
Two findings of 2026-11-22/23 decide this step:
- **facts3 rounds** (facts2's recipe with more pairs, then continued tuning from round 4):

  | round | pairs | bench1000 | knowledge /280 |
  |---|---|---|---|
  | v20 (`facts2`) | 145k | 547 | 120 |
  | r1 | 160k | 541 | 119 |
  | r2 | 188k | 539 | 121 |
  | r3 | 203k | 548 | 122 |
  | r4 (continued from r3) | +15k new, replay | 530 | 106 |
  | r5 (continued from r4) | +new, replay | 537 (GPU) | 115 |

  r1–r4 measured on the CPU, r5 on a GPU, where v20 also scores 537 (10 below the CPU). More pairs of the same kind do
  not move knowledge, and continued tuning on a few new ones forgets.
- **pass@k of v20** (`scripts/hf/passk.py`, K = 8, T = 0.7, bench1000):

  | part | pass@1 | sample avg | pass@8 | of |
  |---|---|---|---|---|
  | yes/no | 122 | 120.4 | 132 | 165 |
  | preschool | 117 | 113.1 | 136 | 160 |
  | knowledge | 120 | 104.5 | 156 | 280 |
  | arithmetic | 6 | 10.2 | 34 | 100 |
  | complete | 80 | 64.6 | 95 | 160 |
  | memory | 102 | 99.0 | 108 | 135 |
  | **all** | **547** | 511.9 | **661** | 1000 |

  RL could reach at most part of the gap (+36 on knowledge, some of it luck). 124 knowledge questions are never answered
  right in 8 samples.
- **Coverage** of those questions by the pairs (`qa_facts.jsonl`, 203k; a pair counts when it holds the question's
  content words and its answer matches the benchmark's pattern): of the 120 never answered, **84 have no pair with
  their answer** (capitals of Poland, Norway, Egypt; a baby cow); of the 120 answered greedily, 64 have none either
  (learned from the shelves). Random shelf windows rarely hold the plainest facts, and Simple Wikipedia was 0.2% of
  `bil10`'s shelf steps.

## 3. Objective
- **Basic-fact passages** (`scripts/qa/lead_passages.py`): the 40,000 Simple Wikipedia articles mentioned most by
  other articles (title of 1–3 words found in their text), each its opening paragraphs (60–350 words), none from the
  earlier passage files → `passages3.jsonl`. The same teacher, checks and merge as 0020 (`generate.py`, `merge.py`):
  pairs close to a bench1000 question are dropped, so the benchmark asks facts taught in other words.
- **Growth** (`final_test.py grow big`): `facts2/final` (112.1M with the 16k vocabulary) → d 896, heads 20 → 28,
  MLP 1792 → 2432, then 8 core blocks: 2 + 24 + 2 = 28 blocks, **287,688,576 parameters**. New readers at std 0.003;
  each new block a copy of core block 8–15 with its outputs zeroed (see §5).
- **Arm `big`**: from `big/start`, `bil10`'s shelves with Simple Wikipedia raised to 5% three of four steps; the fourth
  conversations, a quarter of them fact pairs (all pairs, passages3's included), the rest talk2's mix; 1e-4 after 2,000
  steps, cosine to 1e-5; validation on `bil10`'s shelves. ~3B tokens, DiLoCo on three cards (~4,600 tokens/s per card).
- **Arm `big_facts`**: facts2's tuning (50M tokens at 1e-5, every other step pairs) of `big/final`.

**Out of scope:** RL (after this, on the larger model), new shelves, a new tokenizer.

## 4. Control Gates
| Gate | What | Pass |
|---|---|---|
| G1 | Growth | every step changes the loss on shelf windows by < 0.01 (passed: ≤ 2.4e-7) |
| G2 | Run, ~300M tokens | shelf validation within +0.25 of `big/start` and falling; else restart at 3e-5 |
| G3 | Result | `big_facts` bench1000 ≥ v20 + 25 on the same device, knowledge ≥ v20 + 25 |

## 4a. Artefacts
| Artefact | Kind | Status |
|---|---|---|
| `scripts/qa/lead_passages.py` | system | created |
| `growth.grow_depth(like=…)` | system | created |
| `final_test.py`: `grow big`, arms `big`, `big_facts` | system | created |
| `scripts/hf/passk.py` | system | created |
| `passages3.jsonl` (40,000; data host) | data | built |
| `checkpoints/final/big/start` (.15) | data | built |

## 5. Additional Information
- **The first growth jumped** (2026-11-23): fresh blocks (norm gains 1, readers at 0.02) and new readers at 0.01 took the
  shelf loss from 3.08 to 4.5 within 10 steps at a learning rate of 5e-7, and a single run stayed at ~4.5 after 400
  steps. A probe showed why: v20's residual stream is quiet (RMS 0.015, a few dimensions up to 3.4), its blocks read it
  through norm gains ~0.35 and their MLP hidden units have RMS ~0.001; the fresh blocks' hidden units had RMS 0.19, so a
  1e-6 sign step on one of their writers moved the loss by 0.10 (an old block's: 0.001–0.002). Freezing only the new
  writers kept the loss flat. With copied blocks and quieter readers, 200 warm-up steps (to 1e-5) give +0.058 on edu 12
  (v20 alone, the same steps: +0.005).
- **Not RL first:** RL pulls out what the model already reaches sometimes; pass@8 bounds it at ~+36 on knowledge, and the
  84 missing facts are not in the data.

## 6. Results
- **Data (2026-11-23):** passages3 built (40,000); generation on all four cards from 07:52 UTC, ~15k pairs in the first
  ~2.5 h on three cards.
