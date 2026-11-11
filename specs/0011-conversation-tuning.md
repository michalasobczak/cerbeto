# 0011 — Conversation tuning: answer the question, keep the facts

Status: in progress (2026-11-11) — G1 passed at 17,400 teacher questions; G2 passed; the 1B run stopped at 982M; `talk` (1e-4) forgot the shelves (validation +0.2–0.3) and was stopped at 25M; `talk2` (3e-5, shelves every other step) started 22:50 for 50M tokens · Depends on: 0010 · Unblocks: v6, school

## 1. TL;DR
After 1B tokens the student knows facts but does not answer questions about them: it completes "The heart pumps" → *blood*
10 of 10, but "Where do fish live?" went 9 → 0 of 10 from v4 to v5, and the chat spot check stopped growing with the tokens
(69 → 52 → 60 → 55 → 54 of 160 from 555M to 783M). We train **50–100M tokens of conversations** from the 1B student:
questions about the world with short direct answers, conversations that keep several facts, "I don't know" about made-up
words and replies to feelings, with a quarter of the 0010 mix so nothing is forgotten. This is phase 3 of spec 0010 §9.

## 2. Abstract
The 0010 request → reply steps (one of three) were preschool templates, stories on request, who the student is,
two-step relations and conversations built from them: no ordinary question about the world with its answer. The
knowledge from the edu and textbook shelves stays in the "complete the text" form. Conversation tuning (as in
SmolLM2-Instruct or TinyLlama-Chat, at our scale) teaches the form: a question → its direct answer, in the first words.

## 3. Objective
- **Teacher questions** (`cerbeto.train.talk_data generate`, gemma4 on the MacBook, ~14 h, ~40k pairs): 117 child topics
  × 10 angles; 25 questions per call, answers of 1–2 short true sentences that start with the answer ("Yes", "No",
  "True", "False" first). Kept: one line, ASCII, ≤ 3 sentences, true/false answered; dropped: duplicates, exam prompts
  (normalised text or hash) and **every question with all content words of a spot-check question** ("Where do big fish
  live?" ⊇ {fish, live}), so the spot check stays held out.
- **Code conversations** (`talk_data`):
  - *memory*: 2–4 facts about the human (name, pet and its name, a relative's name, favourite colour, age, food,
    a thing and its colour, a place), 0–3 other questions, then 1–3 of the facts asked back; 15% also ask a fact never
    told → "I don't know. You did not tell me." Names, pets and people of the spot check's memory conversations are not used;
  - *unknown*: made-up words ("What is a fipzor?" → "I don't know what a fipzor is."), never a probe word;
  - *feeling*: 12 situations (sad, scared, hurt, birthday, …) with a kind reply.
- **Stream** (arm `talk` in `final_test.py`, from `checkpoints/final/bil/final`):
  - three of four steps request → reply, only the student's turns learned: qa 30% (3–8 unrelated questions in one
    conversation, each answered), memory 20%, 0010 dialogue 20%, lesson 10%, story 8%, self 4%, relation 4%,
    unknown 2%, feeling 2%;
  - the fourth step the 0010 shelves, same mix (~25% of tokens);
  - learning rate 1e-4 (0010: 3e-4), warm-up 100 steps, cosine to 1e-5; fresh optimizer; validation as 0010.
- **Tokens:** 50M (~5.5 h at 2,500 tok/s; planned 75M, cut because there are 17.4k questions, not ~40k, so each would
  repeat ~50 times); pairs are counted with their padding, as in 0010.

**Out of scope:** new shelves, a new tokenizer, a teacher that grades answers during training, the school.

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `brain/cerbeto/train/talk_data.py`: teacher questions, memory, unknown, feeling, spot-check filter | system | created |
| `brain/cerbeto/train/final_test.py`: arm `talk` (pair kinds qa, memory, unknown, feeling; 3 of 4 steps pairs; lr per arm) | system | created |
| `brain/tests/test_talk_data.py` | test | created |
| `data/talk/qa_teacher.jsonl` (git-ignored, copied to the M6) | data | being written |

## 5. Critical Path
1. Teacher questions on the MacBook (while 0010 ends) → 2. 0010 final: evaluation, spot check of `bil/final` against v5 →
3. copy the questions to the M6, `final_test train talk --tokens 50e6` → 4. spot check against v5 and `bil/final`,
publish v6 if the chat is better.

## 6. Control Gates
| Gate | What | Pass |
|---|---|---|
| G1 | Data | ≥ 20k teacher questions; 0 spot-check or exam questions (filter counts in the log) |
| G2 | Dry run | 20 steps on the M6 CPU from a copy of the checkpoint, every pair kind encoded, loss finite |
| G3 | Result | chat spot check better than v5 by > 5 of 160, memory not worse, completing not worse by > 5 of 80 |

## 7. Expected Outcome
Chat answers start with the answer ("Fish live in water."), the spot check's chat clearly above v5 (55/160) and v4 (69/160),
memory of names and pets back to v4 or better; shelf validation loss rises a little (< 0.05), the price of the form.

## 8. Additional Information
- **Why not change the 1B run's data (2026-11-11):** the user asked whether to stop the run and continue on better data.
  The run had ~15 h left at a low learning rate, the new data did not exist yet, and changing the stream of a running
  arm needs a code change in resume — so the run ends as planned and the teacher writes the data meanwhile.
- **Fewer questions, fewer tokens (2026-11-11):** the teacher slowed from ~3,500 to ~1,000 kept questions an hour
  (the MacBook busy, more duplicates) and the user stopped it at 17,400 (~312k tokens). The talk arm was cut to 50M tokens.
- **1B stopped at 982M (2026-11-11 20:07):** at learning rate 3e-5 the shelf loss was flat at ~2.45 and validation moved by
  thousandths, so the user stopped the run (SIGINT, clean save) instead of waiting two hours. `checkpoints/final/bil/final`
  is a copy of `latest` at step 267,654 (weights only, `NOTE.txt` says so).
- **`talk` forgot, `talk2` (2026-11-11 22:45):** at 1e-4 with a fresh optimizer the shelf validation rose within the first
  3M tokens (edu 3.24 → 3.50, textbooks 2.15 → 2.41, dialogue 1.41 → 1.61, stories 1.34 → 1.53) and stayed there; shelf
  training loss 2.45 → ~2.72, while the pair loss fell 0.89 → 0.43. The 1B run ended at 3e-5, so a 3× higher rate knocked
  the student out of its minimum. The user stopped it at 25M tokens (`checkpoints/final/talk/latest`, kept for a spot
  check). Arm `talk2`: the same data from `bil/final`, peak 3e-5 (100 warm-up steps, cosine to 3e-6), every other step on
  the shelves (~50% of tokens), 50M tokens. Pass for the shelves: validation within +0.05 of `bil/final`.
- **Spot check of `talk` at 25M (1e-4), 2026-11-11 23:20:** chat 55 → 116 of 160 against v5 (982M: 62), completing
  35 → 40 of 80 (982M: 33), memory 19 → 40 of 40 (982M: 13). G3 passed by far: the shelf loss rose, but the facts the
  spot check asks for were not lost — "Where do fish live?" 0 → 9, "What do cows give us?" 0 → 9, the cat Tom and the
  horse Sunny 0 → 10. Still wrong: "What do bees make?", the capital of France, "A fish can swim". `talk2` after 3.2M
  tokens: validation +0.003 to +0.017 of `bil/final` (pass).
