# 0011 — Conversation tuning: answer the question, keep the facts

Status: in progress (2026-11-11) — teacher questions being written on the MacBook while the 1B run ends on the M6; G2 passed (30 steps on the M6 CPU from v5: every pair kind encoded, weights finite, replies sane) · Depends on: 0010 · Unblocks: v6, school

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
- **Tokens:** 75M (~8 h at 2,570 tok/s); pairs are counted with their padding, as in 0010.

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
3. copy the questions to the M6, `final_test train talk --tokens 75e6` → 4. spot check against v5 and `bil/final`,
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
