# 0020 — Fact questions from the shelves, then tuning

Status: tuned (2026-11-22): `facts2` passes G2, misses G3 by 0.02; more pairs generating · Depends on: 0016, 0017, 0018 · Unblocks: a release, 0013

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
| G4 | Release | G2 and G3 → a release (tag v<row>-<tokens>M) on the Space (default) and in the model repo |

## 5. Additional Information
- **Why a local teacher:** no costs, no limits, and no shelf text leaves our machines.
- **Why Qwen3-8B in AWQ:** 5.7 GB leaves ~5 GB of the 3060's 12 GB for the batch, so vLLM serves dozens of passages at
  once; the 14B (9.3 GB) would leave room for a few.
- **bench1000 stays clean:** the merge drops pairs close to its questions; the 64 already in `qa_teacher.jsonl` were left
  out of bench1000 when it was built.

## 6. Results
- **Data (2026-11-21, G1 partly):** generation stopped early on the user's call (three cards; the fourth's fan failed after
  1,024 passages), 157,461 pairs from ~132k of 220,721 passages; the merge dropped 8,952 duplicates and 338 close to
  bench1000: **145,234 train, 2,937 valid** (`what` 80k, `why` 20k, `where` 10k, …; textbooks 45k, edu 12 44k, wiki 35k).
  Below the 300k of G1, accepted to test the idea sooner (the rest of the passages can be generated later, resumable).
  Hand check of 100 random pairs: ~90 good; bad ones are non-answers ("Communicating with a partner is so important."),
  questions that lean on their passage (Miss Alice, the lemonade stand), tautologies and a few wrong facts (the penalty
  spot "within the six-yard box").
- **Tuning:** 50M tokens from `bil10/final`, each arm on one card (not DiLoCo: a few hours); fact pairs are 2/3 of
  the pair steps (1–4 pairs chained per sequence), the rest `talk2`'s mix.

  | # | Arm | Pair steps | bench1000 | knowledge /280 | completing /160 | edu 12 | textbooks | s4 stories | dialogue | speech |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 15 | `bil10/final` | – | 503 | 69 | – | 3.353 | 2.403 | 2.404 | 1.509 | 3.173 |
  | 19 | `facts` | 3 of 4 | 525 | 117 | 63 | 3.549 (+0.20) | 2.618 (+0.22) | 2.615 (+0.21) | 1.618 (+0.11) | 3.307 (+0.13) |
  | 20 | `facts2` | 1 of 2 | **547** | **120** | 80 | 3.417 (+0.06) | 2.471 (+0.07) | 2.470 (+0.07) | 1.544 (+0.04) | 3.215 (+0.04) |

  (validation at the arm's last check: `facts` 49.8M tokens, `facts2` 47.2M.) G2 passes for both on knowledge
  (+48, +51); overall `facts` is +22 (below 25), `facts2` +44. G3 fails: `facts` forgets about as much as `talk10`,
  `facts2` is 0.01–0.02 over the limit on three shelves. The pairs had been seen about 2–3 times (`facts`) and 2
  (`facts2`); another pass over the same pairs would teach the answers, not more facts.
- **More pairs (2026-11-22):** generation resumed on three cards (two non-stop; the dead-fan card in 128-passage chunks
  with 5 min rest and a wait below 60 °C, `--pause`/`--cool`); the fourth card is free for tuning.
