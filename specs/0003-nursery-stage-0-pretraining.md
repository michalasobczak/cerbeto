# 0003 — Nursery: language learning (stage 0) and baseline measurement

Status: in progress (growing student training since 2026-10-31 20:46) · Depends on: 0001, 0002 · Unblocks: 0005

## 1. TL;DR
We teach the student to speak English on the stage 0 shelves (child-directed speech → simple stories), in parallel with a control student of fixed size 30M. Speech is kept as a **conversation with roles** (adult `<|human|>` ↔ child `<|student|>`), so turn-taking is learned from day one. Proof: the nursery exam (BLiMP + a conversation probe) and a recorded baseline measurement for all grades 1–3.

## 2. Abstract
The student starts with random weights, so before it begins lessons with a teacher it must master language. This is the most compute-heavy stage (≥ 250M tokens) and it happens once, centrally, on a 32 GB computer; people receive a student that already speaks (0007). The data order mimics development: first child-directed speech, then stories. A child learns language **in conversation**, so the speech shelf keeps who is speaking: in CHILDES the child's turns become `<|student|>` and the adults' turns `<|human|>` — the same chat format the student later uses with the teacher and in the admin panel. (In 0001 speaker tags were stripped; this specification rebuilds `s0_speech` with roles.) To prove the project's thesis, we train in parallel a control student without growth (30M from the start).

## 3. Objective
**Goals**
- Training loop: AdamW, cosine LR with warm-up, mixed precision where the device supports it, gradient accumulation, resuming after interruption.
- `s0_speech` rebuilt with conversation roles: CHILDES `*CHI` → `<|student|>`, every other speaker → `<|human|>`; Switchboard A/B → alternating `<|human|>`/`<|student|>`; subtitles and BNC (no reliable speaker info) stay plain text.
- Nursery curriculum: phase A `s0_speech` (~30% of tokens), phase B a mix of `s0_speech` + `s0_stories` (~70%).
- Growing student (start ~8M, growth according to 0002 on plateau) and control student (fixed 30M) on the same data in the same order.
- Nursery exam and baseline measurement: results of both students on the grade 1–3 exams (0004) **before** school.

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
- Growth in the nursery is allowed when the validation loss decreases by < 1% over 3 consecutive evaluation windows.
- Checkpoint every 30 min and at the end of each phase; every checkpoint is a version (manifesto §4).
- The nursery exam runs on a validation set the student has not seen.

**Resilience:** interruption (closing the laptop) → resume from the last checkpoint at the exact position in the data stream.

**Data:** ~2 × 300 MB of checkpoints, metrics in JSONL.

## 6. Critical Path
1. Training loop + resuming → 2. phase A on the growing student (short test run) → 3. BLiMP exam → 4. full run of both students in the background → 5. baseline measurement for grades 1–3.

**Completion condition:** both students have a "post-nursery" version with exam results and the baseline measurement recorded in `runs/nursery/`.

## 7. Control Gates
| # | Type | Condition | Verification | Blocks |
|---|---|---|---|---|
| G1 | Unit | Resuming gives the same next batch and the same loss (±1e-5) | `pytest tests/test_resume.py` | full run |
| G2 | Integration | BLiMP ≥ 65% (growing student) | `cerbeto exam nursery` | 0005 |
| G3 | Integration | Conversation probe ≥ 60% fitting replies | `cerbeto exam nursery --conversation` | 0005 |
| G6 | Unit | `s0_speech`: in CHILDES every child turn starts with `<|student|>`, every adult turn with `<|human|>`; the tokenizer keeps both as single tokens | `pytest tests/test_shelves.py` | full run |
| G4 | Integration | Baseline measurement for grades 1–3 recorded for both students (runs on the saved post-nursery checkpoints once 0004 exam pools exist) | file `runs/nursery/baseline.json` | 0005 |
| G5 | Manual | 20 random student utterances readable by a human | review | 0007 |
| N1 | Negative | No exam example (BLiMP, grades 1–3) appears in the training data | hash check | G2–G4 |

## 8. Expected Outcome
- **Direct:** a student speaking simple English.
- **Measurable:** BLiMP, perplexity, conversation probe, baseline measurement for grades 1–3.
- **Side effects:** a dozen or so hours of full GPU load.
- **Rollback:** any nursery checkpoint.

## 9. Additional Information
- **Rejected alternatives:** a single student without a control group (impossible to prove the benefit of growth).
- **Technical debt:** the conversation probe is graded by an LLM (soft); the hard gates are BLiMP and the code-based exams in 0004 (including the code-graded social skills).
- **Rejected alternatives (added):** story continuation as the nursery ability test — it does not show whether the student can hold a conversation.
- **Sources:** BabyLM Challenge; BLiMP (Warstadt et al., 2020).
- **Open questions:** BLiMP threshold after the first measurement; if the real time exceeds 24 h — shorten the control student's nursery to the same number of tokens as the growing one, or go back to considering MLX.
