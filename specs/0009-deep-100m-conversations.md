# 0009 — A deep, thin 100M student that learns conversations

Status: closed (2026-11-07) — success: the `mid` arm (106.8M) beats `v0017` on every measure and passes preschool by the rule of 0004; `/stats` analysis pending · Depends on: 0002, 0004, 0005, 0008 · Unblocks: decision on student size, growth method and conversation data for the school

## 1. TL;DR
We grow the school's best version `v0017` (34.6M) to a deep, thin **106.8M** student — width 640, 20 heads, 20 blocks without a loop — and train it on 80M tokens: shelf windows plus request → reply data in which a third are **conversations of 2–5 turns**. Unlike 0008, the unrolled copies start apart (noise) and the dead attention is renewed. Proof: a held-out conversation probe, `/stats` attention analysis and the 0008 measures, against `v0017`.

## 2. Abstract
0008 grew `v0017` to 283M and trained it on 18M tokens with no gain: validation flat, the unrolled blocks still copies of `v0017`, four of them a dead attention, and the student continued a story when asked "who are you?" after it, because it had only ever seen one turn. 0009 changes three things at once: a smaller, deep-and-thin shape (like MobileLLM-125M) that trains ~2.7× faster, so 80M tokens is a larger share per parameter; a growth that breaks the symmetry of the copies and renews the dead attention; and conversations in which the subject changes, so the student must answer the last turn and use earlier ones only when asked about them. It runs outside the school; the school's 40M limit stays.

## 3. Objective
**Goals**
- **Growth** of `v0017` (all through `growth.apply` with the Δ loss guard):
  - width 512 → 640 (`grow_model_dim`), heads 16 → 20 (`grow_width`), MLP 1,792 unchanged;
  - the core run 4 times unrolled into 16 blocks (`unroll_core`, `noise = 0.01`: Gaussian noise of 1% of each weight's spread on every copy after the first);
  - every block with a dead attention (q·k norms < ¼ of the median) gets fresh q, k, v and a zero output o (`renew_attention`);
  - **2 + 16 + 2 = 20 blocks, 106.8M parameters, K = 1**; activation checkpointing; bf16.
- **Stream, 80M tokens:**
  - two of three steps: shelf windows as in 0008 (stories 40%, speech 30%, books 18%, edu 9%, Wikipedia 3%; every token learned);
  - every third step: 8 conversations — a story on request 30%, preschool lessons 25% (never an exam prompt), who the student is 15% (18 questions), **multi-turn conversations 30%**; every student turn learned, no human turn.
- **Multi-turn conversations** (`Pairs.dialogue`), up to 504 tokens: 60% start with a story about "a ‹kind› named ‹Name›", followed in 70% by a question about it (5 templates: "What was the dog's name?" → "The dog's name was Max."); then lessons and questions about the student up to 2–5 turns.
- **One turn format** in training (`encode_dialogues`) and in the chat (`cerbeto chat --memory`): `<|eos|>` once, then per turn `<|human|> …\n<|student|><|answer|> …<|eos|>`.
- **Evaluation** of `v0017` and `mid`: the 0008 measures (validation loss per shelf, stories on request graded by the teacher, nursery conversation probe, preschool exam) plus a **conversation probe**: 10 held-out conversations, the last reply must contain an expected word (a name from the story, "Cerbeto"/"student", "blue", "no").

**Budget:** 80M tokens at ~3,000–3,500 tokens/s expected on the M6 (to be measured) ≈ 7 h.

**Out of scope:** training to the end (~2B tokens); mixture of experts; changing the school, its limit or its turn format.

**Success criteria**
- Growth accepted at every step (Δ loss ≤ 0.01 for width, ≤ 0.02 for the unroll with noise, ≤ 0.05 for the renewal).
- The conversation probe clearly above `v0017` (target ≥ 7 of 10).
- In `/stats`: no block with spread 1.00; heads that point at content tokens ("sky", the name in a story).
- Preschool exam and nursery language not worse than `v0017` by more than 2 pp / 0.05 loss.

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `growth.unroll_core(noise=…)`, `growth.renew_attention(where=…)` | system | created |
| `school.smoke.encode_dialogues` | system | created |
| `final_test.py`: arm `mid`, `Pairs.dialogue`, `RECALL`, `SELF` (18), `DIALOGUE_PROBE`, `dialogue_probe` | system | created |
| `cerbeto chat`: every line separate by default, `--memory` / `/memory on` in the training format, `/stats` | system | created |
| Panel tab *100M* (evaluation column *dialogue*) | system | created |
| `checkpoints/final/mid/`, `runs/final/mid/` | data | created |

## 5. Functional Domain Changes
**Rules:**
- Growth is function-preserving except the renewal of a dead attention, which removes only that block's even averaging; every step is guarded and logged (`runs/final/mid/growth_log.jsonl`).
- A conversation's human turns are never learned; the student's turns are learned in order, each closed by `<|eos|>`.
- Conversation turns come from code-generated material only (stories of the shelf, lessons of 0004, fixed answers about the student); no exam or validation prompt (exam hashes excluded).
- The conversation probe uses phrasings and names never seen in training.

**Resilience:** a checkpoint every 30 minutes; resumable and extendable as in 0008.

## 6. Critical Path
1. Grow `v0017` → 106.8M (four guarded steps) → 2. train 80M tokens on the M6 → 3. evaluate `v0017` and `mid` → 4. `/stats` on the probe questions → 5. record the decision in §9.

**Completion condition:** results table, conversation probe and attention analysis in §9.

## 7. Control Gates
| # | Type | Condition | Verification | Blocks |
|---|---|---|---|---|
| G1 | Unit | Each growth step within its Δ loss limit | `growth_log.jsonl`, `tests/test_growth.py` | 2 |
| G2 | Unit | A conversation learns every student turn and no human turn; one turn equals a plain pair | `tests/school/test_day.py` | 2 |
| G3 | Unit | The chat keeps its history in the training format | `tests/test_talk.py` | 3 |
| G4 | Integration | 80M tokens reached | `runs/final/mid/state.json` | 3 |
| G5 | Integration | `v0017` and `mid` evaluated with the same settings, K as trained | `runs/final/eval_*.json` | 5 |

## 8. Expected Outcome
- **Direct:** the student keeps turns apart — after a story it answers "who are you?" about itself and "what was the dog's name?" with the name from the story.
- **Measurable:** conversation probe, story grades, conversation probe of the nursery, preschool exam, validation loss; attention per block in `/stats`.
- **Reading the result:** a gain in the conversation probe with a flat validation loss points to the data format, not size; content-pointing heads and no spread 1.00 show the growth method works. No gain at all would point to the amount of data (80M is ~4% of ~20 tokens per parameter).
- **Side effects:** the M6 is busy ~7 h; the school does not run meanwhile.
- **Rollback:** none needed — the school's checkpoints are not touched.

## 9. Additional Information
- **Growth (2026-11-06, `runs/final/mid/growth_log.jsonl`):** `grow_model_dim` 34.6M → 43.3M, Δ loss 0.0; `grow_width` → 45.9M, Δ loss 0.0; `unroll_core` with noise → 106.8M, Δ loss 0.00005; `renew_attention` of core 3, 7, 11, 15, Δ loss 0.00002 — the dead attention's averaging was worth almost nothing.
- **Memory:** without activation checkpointing the 106.8M student ran out of GPU memory on the 16 GB M6 at the first steps; with it, as in 0008.
- **Next, if the MLPs turn out to be the bottleneck:** a mixture of experts grown from the existing MLP (copies as experts plus a router).
- **Sources:** MobileLLM (Liu et al., 2024); attention sinks (Xiao et al., 2023); most heads prunable, few specialised (Voita et al., 2019); induction heads (Olsson et al., 2022); function-preserving growth (0002).
- **Training (2026-11-06 21:40 → 11-07 05:00, M6, bf16):** 19,531 steps, **69.4M tokens** (the step count was planned for 80M; request → reply steps carry fewer tokens than shelf windows), ~2,630 tokens/s. Validation after growth rose to 1.50 on stories and came back over the whole run; still falling at the end with the learning rate at its minimum. Pair loss ~1.1 → ~0.9.
- **Results (2026-11-07, `runs/final/eval_v0017.json`, `eval_mid.json`, same settings, M6, teacher `gemma4:e4b-mlx`):**

| Measure | `v0017` (34.6M) | **`mid` (106.8M)** |
|---|---|---|
| validation loss: speech / stories | 3.17 / 1.38 | **3.09** / 1.38 |
| conversation probe (10 held-out) | 2 / 10 | **8 / 10** |
| stories on request: teacher's mean grade (1–5) / share graded ≥ 4 | 1.5 / 0% | **4.3 / 90%** |
| nursery conversation probe (fitting) | 84% | **88%** |
| preschool exam | 68.5% | **76.3%** |
| preschool passed (rule of 0004) | no | **yes** |

  Passing rule of 0004 (code-graded skills): `id` ≥ 85% in each — `mid` 89.7–100%; mean `ood_hard` ≥ 70% — **78.5%**; every `ood_hard` ≥ 50% — lowest **50.0%** (facts, exactly at the floor). `v0017` fails on agree/disagree `id` 57%. `ood_combo` is not part of the rule (opposites 0%).

  Preschool exam per skill, `id` / `ood_hard` / `ood_combo`:

| Skill | `v0017` | `mid` |
|---|---|---|
| polite routines | 97 / 77 / 74 | 100 / 69 / 83 |
| agree / disagree | 57 / 56 / 57 | **90 / 79** / 42 |
| opinion with a reason | 90 / 73 / 46 | **100 / 100 / 90** |
| asking for clarification | 100 / 52 / 45 | 100 / **77 / 67** |
| facts | 95 / 57 / 69 | 90 / 50 / **51** |
| opposites | 94 / 85 / 13 | 100 / 85 / **0** |
| sizes | 97 / 69 / 50 | 100 / **90** / 52 |
| categories | 66 / 74 / 53 | **100 / 100 / 77** |

  Conversation probe, last reply of `mid`:
  - passed: "Tell me a story. / Who are you?" → *"I'm Cerbeto. I'm a little student, and I'm learning to talk."* (`v0017`: went on with a story); "…a bear named Ben. / Is a cat an animal? / What was the bear's name?" → *"The bear's name was Ben."* (three turns, the subject changed in between); "…a boy named Sam. / Who was that story about?" → *"It was about a boy named Sam."*; "…a bird named Pip. / Are you a bird?" → *"No, I'm not a bird. I'm a little student."*
  - failed: "…a cat named Luna. / Do you remember the cat's name?" → *"No, I do not remember the cat's name."* (a phrasing never trained); "…a fox named Red. / Was the story about a dog?" → *"Yes, it was about a dog named Red."* (keeps the name, loses the kind; it agrees with the question).
- **Reading the result:**
  - **Success on every criterion measured:** growth accepted at every step; conversation probe 8/10 (target ≥ 7); preschool exam +7.8 pp and nursery language better, not worse.
  - **The data format did most of it:** turns, who the student is, and stories on request come from the conversations and pairs; 283M (0008) on single turns did not get them.
  - **The preschool exam rose without a single school night:** +7.8 pp, more than days 9–18 of school gave `v0017`. The gain is in `id` and in the social skills, which the request → reply data practised; categories and agree/disagree reached 100% and 90% `id`.
  - **Weak spots:** new combinations of facts and opposites (`ood_combo` 51% and 0%) went down — the knowledge is memorised, not composed; it agrees with leading questions ("Was the story about a dog?" → "Yes").
  - **Data, not capacity, is the limit now:** validation was still falling at 69M tokens (~0.6 tokens per parameter). Next: [0010](0010-one-billion-tokens.md), 1B tokens with new data.
- **Not done yet:** the `/stats` attention analysis (content-pointing heads, no spread 1.00) on the probe questions.
