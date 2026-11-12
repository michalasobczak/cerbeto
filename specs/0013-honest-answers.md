# 0013 — Honest answers: say what you know, "I don't know" about the rest

Status: in progress (2026-11-12) — facts test being written by the teacher; probing and the `honest` arm wait for 0012's `v16k` · Depends on: 0011, 0012 · Unblocks: v7

## 1. TL;DR
A small model knows few facts, so it should answer those and say **"I don't know"** about the rest instead of guessing
(the user, 2026-11-12: "if it has to not know something, better it says it does not know than guesses"). We do it the
way the large labs do for factuality, at our scale: (1) a **held-out facts test** that counts correct, wrong and
"I don't know" separately and scores **correct − wrong**; (2) **probe what the student itself knows** on every training
question; (3) **tune** with the teacher's answer where the student knows the fact and "I don't know" where it does not.

## 2. Abstract
Fine-tuning a model on facts it does not know teaches it to answer confidently without knowing — to hallucinate
(Gekhman et al. 2024, *Does fine-tuning LLMs on new knowledge encourage hallucinations?*). Spec 0011 did exactly that:
17,400 teacher answers, whether the student knew them or not. Llama 3's factuality recipe (2024) and R-Tuning (Zhang
et al. 2023) instead probe the model: questions it answers right keep their answers, questions it gets wrong get a
refusal. OpenAI's SimpleQA (2024) and *Why language models hallucinate* (2025) argue that tests must not reward
guessing: a wrong answer has to cost more than "I don't know". We follow both, with a local teacher (gemma4) and a
cheap probe (the student's own likelihood of the right answer), so the whole spec costs a few teacher hours and
~4 h of training.

## 3. Objective
- **Facts test** (`brain/cerbeto/exams/facts_eval.jsonl`, committed, never trained on): 120 *child* questions (what a
  5-year-old knows), 120 *school* questions (what a 10-year-old learns: capitals, planets, organs — mostly beyond the
  student), 60 *unanswerable* (places and things that do not exist, the human's untold facts), written by code.
  Teacher questions are dropped when they equal a training question or have its content words, or hit the spot check.
- **Grading:** greedy reply (K = 1, temperature 0), teacher labels it correct / wrong / idk against the reference.
  Reported per part: correct, wrong, idk, **score = correct − wrong**, and **wrong when answering** = wrong / (correct + wrong).
- **Probe** (on `v16k`): for every training question the student's greedy reply and its mean log-probability of the
  teacher's answer; a 600-question sample graded by the teacher sets the threshold that separates "knows" from "does not".
- **Data:** known → the teacher's answer; not known → "I don't know." (a few phrasings, never more than ~30% of the
  questions); in between → left out. The 0011 memory conversations already say "I don't know. You did not tell me."
- **Arm `honest`** from `v16k/final`: talk2's stream with the probed questions, peak 3e-5, every other step the
  shelves, 25M tokens (~3 h).

**Out of scope:** preference tuning (DPO) — the next step if the tuned student still guesses; retrieval; a bigger teacher.

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `brain/cerbeto/train/honest.py`: make-eval, answer, grade (probe, build next) | system | created |
| `brain/cerbeto/exams/facts_eval.jsonl` | data | being written |
| `runs/honest/` answers, grades, summaries (git-ignored) | data | — |
| `final_test.py`: arm `honest` | system | planned |

## 5. Critical Path
1. Facts test (MacBook) → 2. baseline: v5, v6 (and `v16k` when it ends) → 3. probe `v16k` on the training questions →
4. calibrate the threshold on 600 teacher-graded replies → 5. build the honest data → 6. `final_test train honest` →
7. facts test + spot check → v7 if it passes G3.

## 6. Control Gates
| Gate | What | Pass |
|---|---|---|
| G1 | Facts test | 300 questions, 0 equal to or with the content words of a training or spot-check question |
| G2 | Probe | the threshold predicts the teacher's grade on held-out probe replies with ≥ 75% accuracy |
| G3 | Result | wrong-when-answering on child + school down by ≥ 30% (relative) against the start; correct on child not down by > 5 of 120; unanswerable: idk ≥ 70%; spot check chat not down by > 5 |

## 7. Expected Outcome
On school questions most replies become "I don't know" instead of a confident wrong answer; child questions keep
their answers; questions about things that do not exist get "I don't know" almost always. Score (correct − wrong)
clearly above v6 even if correct falls a little.

## 8. Additional Information
- **Why the student's likelihood and not the teacher for every probe:** grading ~17k replies with gemma4 takes over a
  day on the MacBook; the likelihood is one forward pass per question, and 600 teacher grades are enough to set and check
  the threshold.
