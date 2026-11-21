# 0020 — Fact questions from the shelves, then tuning

Status: generating (2026-11-21) · Depends on: 0016, 0017, 0018 · Unblocks: v8, 0013

## 1. TL;DR
The 1000-question benchmark (2026-11-21) showed that ten billion tokens of school text barely moved the answers (v6 488 →
`bil10` 503 of 1000) and knowledge asked as a question not at all (70 → 69 of 280), while conversation tuning moved them by
172. The student has read the facts but cannot reach them when asked. We have a local teacher write **hundreds of
thousands of question → answer pairs from the very passages the student read**, check each pair twice, and tune
`bil10/final` on them.

## 2. Abstract
The conversation data so far (`qa_teacher.jsonl`, 17.4k pairs) are questions a five-year-old asks: what floats, what is
hot. They taught the answering form (v5 → v6) but not the facts of the 8–12 grade shelves. A question written from a
passage the student has read asks for exactly what is in its weights; the answer is the same fact in the form of a reply.
The teacher runs on the four training cards (bil20 cancelled, 0018), which makes ~10x more pairs per hour than the
Mac mini.

## 3. Objective
- **Passages** (`cerbeto.train.qa_passages`): 120k runs of whole paragraphs, 120–350 words, from the train split of the
  shelves (Simple Wikipedia 25%, Cosmopedia textbooks 25%, FineWeb-Edu grade 8–12 20%, grade ≤ 7 15%, grade 7–8 10%,
  stories that teach 5%), mostly letters; 2% held out by hash for validation.
- **Teacher** (`scripts/qa/generate.py`): Qwen3-8B (AWQ, no thinking) in vLLM, one per card. Per passage 3–8 pairs:
  closed book (the question stands alone, never "the passage"), general facts only, answer in one short complete sentence
  that repeats the subject, plain English, a mix of what / who / where / when / why / how / how many / which of two /
  yes-no.
- **Checks:** rules (question mark, length, no reference to the text, yes/no answers start with Yes or No, ASCII), then
  the teacher again, at temperature 0, judges every pair: stands alone, correct and supported, general. Only pairs passing
  everything are kept.
- **Merge** (`scripts/qa/merge.py`): duplicates out; any question equal or close (≥ 60% of content words) to one of
  bench1000's out, so the benchmark stays a test; held-out passages → `data/talk/qa_facts_valid.jsonl`.
- **Tuning** (after the data): from `bil10/final`, at 1e-5 (its last learning rate, 0017), half the steps fact
  questions, a quarter talk2's conversations, a quarter the shelves; measured on bench1000 (all and knowledge) and on the
  shelves.

**Out of scope:** a larger student, new shelves, a looped core.

## 4. Control Gates
| Gate | What | Pass |
|---|---|---|
| G1 | Data | ≥ 300k kept pairs; a hand check of 100 random pairs finds ≥ 90 good (standalone, correct, general) |
| G2 | Answers | bench1000 above `bil10/final` (503) by ≥ 25, knowledge above 69 of 280 by ≥ 15 |
| G3 | Shelves | validation on every `bil10` shelf at most 0.05 above `bil10/final` |
| G4 | Release | G2 and G3 → v8 on the Space (default) and in the model repo |

## 5. Additional Information
- **Why a local teacher:** no costs, no limits, and no shelf text leaves our machines.
- **Why Qwen3-8B in AWQ:** 5.7 GB leaves ~5 GB of the 3060's 12 GB for the batch, so vLLM serves dozens of passages at
  once; the 14B (9.3 GB) would leave room for a few.
- **bench1000 stays clean:** the merge drops pairs close to its questions; the 64 already in `qa_teacher.jsonl` were left
  out of bench1000 when it was built.

## 6. Results
- (generating)
