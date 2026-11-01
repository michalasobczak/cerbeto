# 0003 — Nursery: language learning (stage 0) and baseline measurement

Status: delivered for the growing student (2026-11-01) — still open: control student run, G4 baseline (waits for 0004 exam pools), G5 human review · Depends on: 0001, 0002 · Unblocks: 0005

## 1. TL;DR
We teach the student to speak English on the stage 0 shelves (child-directed speech → simple stories), in parallel with a control student of fixed size 30M. Speech is kept as a **conversation with roles** (adult `<|human|>` ↔ child `<|student|>`), so turn-taking is learned from day one. Proof: the nursery exam (BLiMP + a conversation probe) and a recorded baseline measurement for all school stages 1–3 (preschool, grade 1, grade 2).

## 2. Abstract
The student starts with random weights, so before it begins lessons with a teacher it must master language. This is the most compute-heavy stage (≥ 250M tokens) and it happens once, centrally, on a 32 GB computer; people receive a student that already speaks (0007). The data order mimics development: first child-directed speech, then stories. A child learns language **in conversation**, so the speech shelf keeps who is speaking: in CHILDES the child's turns become `<|student|>` and the adults' turns `<|human|>` — the same chat format the student later uses with the teacher and in the admin panel. (In 0001 speaker tags were stripped; this specification rebuilds `s0_speech` with roles.) To prove the project's thesis, we train in parallel a control student without growth (30M from the start).

## 3. Objective
**Goals**
- Training loop: AdamW, cosine LR with warm-up, mixed precision where the device supports it, gradient accumulation, resuming after interruption.
- `s0_speech` rebuilt with conversation roles: CHILDES `*CHI` → `<|student|>`, every other speaker → `<|human|>`; Switchboard A/B → alternating `<|human|>`/`<|student|>`; subtitles and BNC (no reliable speaker info) stay plain text.
- Nursery curriculum: phase A `s0_speech` (~30% of tokens), phase B a mix of `s0_speech` + `s0_stories` (~70%).
- Growing student (start ~8M, growth according to 0002 on plateau) and control student (fixed 30M) on the same data in the same order.
- Nursery exam and baseline measurement: results of both students on the stage 1–3 exams (0004: preschool, grade 1, grade 2) **before** school.

**Out of scope:** lessons with a teacher, hippocampus, sleep.

**Success criteria**
- BLiMP (BabyLM filtered version) ≥ 65% [TO BE VERIFIED after the first measurement].
- **Conversation probe:** 50 everyday situations (greeting, thanks, saying goodbye, a simple question about the student, an offer it can accept or refuse); ≥ 60% of the student's replies rated by the teacher-examiner as a fitting conversational turn (rubric 1–5, threshold ≥ 3) [TO BE VERIFIED after the first measurement]. Story continuation is no longer an exam criterion — it only checks that training works (0002).
- The whole of stage 0 for both students ≤ 24 h on a MacBook Pro M1 Pro 32 GB.

**Time budget (estimate for M1 Pro, ~300M tokens, real GPU utilization ~1–1.5 TFLOPS):**
| Student | Estimated time |
|---|---|
| Growing (~8M + loop K) | 5–7 h |
| Control (fixed 30M) | 12–15 h |
| Total | ~20 h — fits within the limit, small margin |

**Attempt 1 (20:46–21:50) was stopped at 52.9M tokens** (mean validation loss 9.07 → 3.66, no growth yet): the trend showed the plateau rule alone would leave the student at ~15M by the end, which would make the control comparison meaningless. Attempt 2 adds growth deadlines, learning-rate re-warm and larger validation; expected end ~12:00–14:00 on 2026-11-01. Logs of attempt 1: `runs/nursery/attempt1/`.

**Measured throughput (2026-10-31, M1 Pro, fp32, micro-batch 8 × 512, accumulation 2, no activation checkpointing):** growing student at 6.9M — ~12.6k tokens/s; control student at 29.9M — ~3.3k tokens/s. Revised estimate for 300M tokens: growing ~10–14 h (slows down as it grows), control **~25 h**. Together ~35–40 h → exceeds the 24 h criterion; the control student runs separately (e.g. over a weekend) on the same 300M tokens, so comparability is kept.

The students are trained **sequentially** (first the growing one, then the control one), e.g. over two nights; simultaneous training would share the same GPU with no gain. Precision per 0002: **fp32** (G6 measured fp16 as slower on M1 Pro). Training uses activation checkpointing and micro-batches of 8 × 512 with gradient accumulation (0002 memory measurement); its ~30% speed cost is not yet in the time budget above — re-estimate after the first hour of the real run.

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `brain/cerbeto/train/loop.py` | system | to create |
| `brain/cerbeto/train/nursery.py` (stage 0 curriculum) | system | to create |
| `brain/cerbeto/exams/blimp.py` | system | to create |
| `brain/cerbeto/library/sources.py` — speech reader with conversation roles | system | to change |
| `brain/cerbeto/exams/conversation_probe.yaml` + rubric | data | to create |
| `data/library/blimp/` (BabyLM-community/BabyLM-BLIMP-Filtered) | data | to create |
| `config/nursery.yaml` (phases, mix, growth schedule 6.9M → 14.0M → 15.8M → 26.7M → 29.9M) | data | to create |
| `brain/cerbeto/teachers/__init__.py` + `config/teachers.yaml` (minimal Ollama client for grading; extended in 0005) | system | to create |
| `checkpoints/{growing,control}/v0001…` | data | to create |
| `runs/nursery/metrics.jsonl` | data | to create |

## 5. Functional Domain Changes
**Rules:**
- The control student differs only by the absence of growth/pruning; the data seed is identical.
- Growth in the nursery happens when the validation loss decreases by < 1% over 3 consecutive evaluation windows (at least 20M tokens after the previous growth) **or at the latest at a deadline: 60M, 120M, 180M and 240M tokens**. The deadlines guarantee the growing student reaches 29.9M before the end of the nursery, so the comparison with the control student is about *growing*, not about *final size*.
- After every growth the learning rate starts a new segment: 200-step re-warm to 70% of the peak, then cosine decay to the floor by the end of training — new neurons need a usable learning rate to learn.
- Growth decisions use 32 fixed validation batches per shelf (~130k tokens) — the plateau test hinges on ~1% changes.
- Checkpoint every 30 min and at the end of each phase; every checkpoint is a version (manifesto §4).
- The nursery exam runs on a validation set the student has not seen.

**Resilience:** interruption (closing the laptop) → resume from the last checkpoint at the exact position in the data stream.

**Data:** ~2 × 300 MB of checkpoints, metrics in JSONL.

## 6. Critical Path
1. Training loop + resuming → 2. phase A on the growing student (short test run) → 3. BLiMP exam → 4. full run of both students in the background → 5. baseline measurement for stages 1–3.

**Completion condition:** both students have a "post-nursery" version with exam results and the baseline measurement recorded in `runs/nursery/`.

## 7. Control Gates
| # | Type | Condition | Verification | Blocks |
|---|---|---|---|---|
| G1 | Unit | Resuming gives the same next batch and the same loss (±1e-5) | `pytest tests/test_resume.py` | full run |
| G2 | Integration | BLiMP ≥ 65% (growing student) — **passed: 65.9%** | `cerbeto exam nursery` | 0005 |
| G3 | Integration | Conversation probe ≥ 60% fitting replies — **passed: 80%, but the grading is too lenient (see §9); from now on a baseline measurement, not a blocking gate** | `cerbeto exam nursery --conversation` | — |
| G6 | Unit | `s0_speech`: in CHILDES every child turn starts with `<|student|>`, every adult turn with `<|human|>`; the tokenizer keeps both as single tokens | `pytest tests/test_shelves.py` | full run |
| G4 | Integration | Baseline measurement for stages 1–3 recorded for both students (runs on the saved post-nursery checkpoints once 0004 exam pools exist) | file `runs/nursery/baseline.json` | 0005 |
| G5 | Manual | 20 random student utterances readable by a human | review | 0007 |
| N1 | Negative | No exam example (BLiMP, stages 1–3) that appears in the training data is scored: the leak check records every BLiMP sentence found verbatim (normalized) in the stage 0 shelves, and those pairs are excluded from the exam (first check: 26 unique BLiMP sentences occur in the shelves — 1,495 occurrences, almost all simple TinyStories sentences; 27 of ~60k pairs excluded) | `cerbeto exam` leak check + `pytest tests/test_exams_nursery.py` | G2–G4 |

## 8. Expected Outcome
- **Direct:** a student speaking simple English.
- **Measurable:** BLiMP, perplexity, conversation probe, baseline measurement for stages 1–3.
- **Side effects:** a dozen or so hours of full GPU load.
- **Rollback:** any nursery checkpoint.

## 9. Additional Information
- **Rejected alternatives:** a single student without a control group (impossible to prove the benefit of growth).
- **Technical debt:** the conversation probe is graded by an LLM (soft); the hard gates are BLiMP and the code-based exams in 0004 (including the code-graded social skills).
- **Rejected alternatives (added):** story continuation as the nursery ability test — it does not show whether the student can hold a conversation.
- **Results — growing student, attempt 2 (2026-10-31 21:55 → 2026-11-01 13:37, 15 h 42 min, M1 Pro, fp32):**
  - 300M tokens, 36,622 optimizer steps; final version `checkpoints/growing/v0032`, 29,893,120 parameters.
  - Growth: 60M deadline → `grow_model_dim` 256→384 + `grow_width` (14.0M); 120M deadline → `grow_depth` core.2 (15.8M); plateau at 145M → `grow_model_dim` 384→512 + `grow_width` (26.7M); plateau at 170M → `grow_depth` core.3 (29.9M). Every operator passed the Δloss ≤ 0.01 guard.
  - Throughput: ~12.6k tokens/s at 6.9M, ~3.9k tokens/s at 29.9M; ~6–8 h lost to the 30M size, slightly more to another local Ollama model sharing the GPU for a while.
  - Validation loss (32 batches per shelf): mean 9.07 → **2.248**; `s0_speech` **3.138**, `s0_stories` **1.358**. Stories improved steadily; speech stalled at ~3.30 between ~45M and ~240M tokens and moved again only during the final LR decay.
  - **BLiMP 65.9%** (gate 65% — passed at the edge). Best: anaphor agreement / principle A (96–100%), wh-vs-that without a gap (97–99%). Worst — below chance, the model systematically prefers the wrong sentence: long-distance wh/that with a gap (9%), existential *there* + quantifiers (13%), NPI licensing (14–22%), coordinate-structure islands (22%). These constructions are nearly absent from child speech and TinyStories.
  - **Conversation probe 80%** fitting (by category: greeting 0.8, thanks 0.4, goodbye 0.8, about self 0.8, offers 1.0, feelings 1.0, apology/help 0.8, world questions 0.8, agree/disagree 1.0, "I don't know" 0.6). Replies are short, toddler-like turns: "Hello!" → "hello, hello!", "How old are you?" → "I got two.", "What color is the sky?" → "blue.", "Do cows say moo?" → "moo.", "What did I eat for breakfast yesterday?" → "I don't know."; failures: echoing ("I made you a sandwich." → "I made you a sandwich."), loops, "no." to "What is your name?".
  - **The probe overrates the student.** The teacher grades whether a reply is a fitting *turn*, not whether it is *true*: "Cats can fly, right?" → "yeah." got 4/5, "Is the sun cold?" → "yeah." got 3/5. The student has a strong "yeah/no" reflex; agree/disagree at 1.0 reflects that reflex, not knowledge. A strict human re-reading counts about half of the replies as both fitting and true. Decision: the conversation probe stays as a baseline measurement only; truthfulness of agree/disagree and honest "I don't know" are graded by code in 0004.
  - Full results: `runs/nursery/growing/exam_v0032.json`; metrics: `runs/nursery/growing/metrics.jsonl`.
- **Sources:** BabyLM Challenge; BLiMP (Warstadt et al., 2020).
- **Open questions:** BLiMP threshold after the control student's measurement; if the real time exceeds 24 h — shorten the control student's nursery to the same number of tokens as the growing one, or go back to considering MLX.
