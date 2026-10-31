# 0003 — Nursery: language learning (stage 0) and baseline measurement

Status: planned · Depends on: 0001, 0002 · Unblocks: 0005

## 1. TL;DR
We teach the student to speak English on the stage 0 shelves (child-directed speech → simple stories), in parallel with a control student of fixed size 30M. Proof: the nursery exam (BLiMP + story coherence) and a recorded baseline measurement for all grades 1–3.

## 2. Abstract
The student starts with random weights, so before it begins lessons with a teacher it must master language. This is the most compute-heavy stage (≥ 250M tokens) and it happens once, centrally, on a 32 GB computer; people receive a student that already speaks (0007). The data order mimics development: first child-directed speech, then stories. To prove the project's thesis, we train in parallel a control student without growth (30M from the start).

## 3. Objective
**Goals**
- Training loop: AdamW, cosine LR with warm-up, mixed precision where the device supports it, gradient accumulation, resuming after interruption.
- Nursery curriculum: phase A `s0_speech` (~30% of tokens), phase B a mix of `s0_speech` + `s0_stories` (~70%).
- Growing student (start ~8M, growth according to 0002 on plateau) and control student (fixed 30M) on the same data in the same order.
- Nursery exam and baseline measurement: results of both students on the grade 1–3 exams (0004) **before** school.

**Out of scope:** lessons with a teacher, hippocampus, sleep.

**Success criteria**
- BLiMP (BabyLM filtered version) ≥ 65% [TO BE VERIFIED after the first measurement].
- Story coherence: ≥ 70% of continuations rated by the teacher-examiner as grammatical and coherent (scale 1–5, threshold ≥ 3).
- The whole of stage 0 for both students ≤ 24 h on a MacBook Pro M1 Pro 32 GB.

**Time budget (estimate for M1 Pro, ~300M tokens, real GPU utilization ~1–1.5 TFLOPS):**
| Student | Estimated time |
|---|---|
| Growing (~8M + loop K) | 5–7 h |
| Control (fixed 30M) | 12–15 h |
| Total | ~20 h — fits within the limit, small margin |

The students are trained **sequentially** (first the growing one, then the control one), e.g. over two nights; simultaneous training would share the same GPU with no gain. Precision per 0002 (fp32 or fp16 after gate G6, no bf16).

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `brain/cerbeto/train/loop.py` | system | to create |
| `brain/cerbeto/train/nursery.py` (stage 0 curriculum) | system | to create |
| `brain/cerbeto/exams/blimp.py` | system | to create |
| `data/library/blimp/` (BabyLM-community/BabyLM-BLIMP-Filtered) | data | to create |
| `config/nursery.yaml` | data | to create |
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
| G3 | Integration | Story coherence ≥ 70% | `cerbeto exam nursery --stories` | 0005 |
| G4 | Integration | Baseline measurement for grades 1–3 recorded for both students | file `runs/nursery/baseline.json` | 0005 |
| G5 | Manual | 20 random student utterances readable by a human | review | 0007 |
| N1 | Negative | No exam example (BLiMP, grades 1–3) appears in the training data | hash check | G2–G4 |

## 8. Expected Outcome
- **Direct:** a student speaking simple English.
- **Measurable:** BLiMP, perplexity, coherence, baseline measurement for grades 1–3.
- **Side effects:** a dozen or so hours of full GPU load.
- **Rollback:** any nursery checkpoint.

## 9. Additional Information
- **Rejected alternatives:** a single student without a control group (impossible to prove the benefit of growth).
- **Technical debt:** coherence assessment by an LLM is soft — only as an auxiliary metric; the hard gates are BLiMP and the code-based exams.
- **Sources:** BabyLM Challenge; BLiMP (Warstadt et al., 2020).
- **Open questions:** BLiMP threshold after the first measurement; if the real time exceeds 24 h — shorten the control student's nursery to the same number of tokens as the growing one, or go back to considering MLX.
