# 0008 — Growth experiment: a 100M student and more training data

Status: in progress (2026-11-06) — training of the grown arm running; same-size arm and evaluation pending · Depends on: 0002, 0004, 0005 · Unblocks: decision on student size and library growth

## 1. TL;DR
We grow the school's best version `v0017` (34.6M) function-preservingly to **99.1M** and train it on 80M tokens: shelf windows plus request → reply pairs. A same-size arm (34.6M) gets the same stream. Proof: both arms and `v0017` measured the same way — shelf validation loss, stories on request, the nursery conversation probe, the preschool exam.

## 2. Abstract
Preschool stopped at ~68% (`v0017`, day 18), and the night bench found no night setting that moves it (0005 §9). Two explanations remain: the student lacks **capacity** (parameters), or it lacks **data** — it has seen 300M nursery tokens, mostly stories, and 10.7M in school. The experiment separates the two with two arms on one stream, and also tests whether request → reply data makes the student usable in a talk. It is a short, deliberate check (~18 h per arm), not training to the end; a longer run follows only if the result is promising. It runs outside the school: the school's 40M limit (`config/student.yaml`) stays.

## 3. Objective
**Goals**
- **Grown arm:** `v0017` grown to 99.1M — width 512 → 896, attention heads 16 → 28 (× 32), MLP 1792 → 3072; blocks unchanged (2 + 4 × K + 2). Right after growth it answers exactly like `v0017`.
- **Same-size arm:** `v0017` at 34.6M, same stream, same tokens.
- **Stream, 80M tokens per arm:**
  - two of three steps: windows of all five shelves — stories 40%, speech 30%, books 18%, edu 9%, Wikipedia 3%; every token learned;
  - every third step: request → reply pairs — a story on request 55%, preschool lessons 40% (never an exam prompt), who the student is 5%; only the student's turn learned.
- **Evaluation** of `v0017` and both arms: validation loss per shelf; stories on request (held-out phrasings, graded by the teacher); nursery conversation probe; preschool exam (0004).

**Budget:** 80M tokens ≈ 18 h per arm at ~1,200 tokens/s (99M, M1 Pro, fp32). What 80M shows from each shelf:

| Shelf | Has | Will see | Passes |
|---|---|---|---|
| stories (`s0_stories`) | 544M | ~21M | 0.04× |
| speech (`s0_speech`) | 83M | ~16M | 0.2× |
| books (`s1_books`) | 37M | ~10M | 0.3× |
| education (`s3_edu`) | 4.4M | ~5M | 1.1× |
| Wikipedia (`s2_world`) | 0.8M | ~1.6M | 2× |

The rest are request → reply pairs. One pass over all 670M would need a ~10× longer run (~7 days).

**Out of scope:** a full training of the 99M student (~2B tokens by the Chinchilla rule); new shelves or crawling; changing the school or its limit.

**Success criteria**
- Both arms finish 80M tokens; growth is function-preserving (Δ loss ≈ 0 at the start).
- A results table: `v0017` vs same-size arm vs grown arm on all four measures.
- A decision recorded in §9: longer training of the grown arm, more data, or neither.

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `brain/cerbeto/train/final_test.py` (`train grown|control`, `eval`) | system | created, not committed |
| Panel tab *Final test* (`brain/cerbeto/panel.py`, `panel.html`) | system | created, not committed |
| `checkpoints/final/{grown,control}/` | data | grown: in progress |
| `runs/final/{grown,control}/` (state, log, evaluation) | data | grown: in progress |

## 5. Functional Domain Changes
**Rules:**
- Growth uses the 0002 operators (`grow_model_dim`, `grow_width`); new units write zero, so the grown student starts with `v0017`'s behaviour.
- Both arms see the same stream in the same order (same seed); the only difference is size.
- Request → reply pairs never contain an exam or validation prompt (exam hashes excluded, as in 0005).
- The core passes K are sampled 1–6 in training, K = 4 in evaluation (as in 0002).
- Training is resumable and can be **extended** from the current step: the learning-rate schedule is stretched to a new token target without restarting.

**Resilience:** a checkpoint every 30 minutes; an interruption loses at most 30 minutes.

## 6. Critical Path
1. Grow `v0017` → 99.1M, check function preservation → 2. train the grown arm, 80M tokens (~18 h) → 3. train the same-size arm, 80M tokens → 4. evaluate `v0017` and both arms → 5. record the decision.

**Completion condition:** the results table and the decision are in §9.

## 7. Control Gates
| # | Type | Condition | Verification | Blocks |
|---|---|---|---|---|
| G1 | Unit | Growth to 99.1M changes the loss of `v0017` by < 0.01 | growth check in `final_test.py` | 2 |
| G2 | Integration | Both arms reach 80M tokens on the same stream | `runs/final/*/state.json` | 4 |
| G3 | Unit | No request → reply pair is an exam or validation prompt | exam hash check | 2, 3 |
| G4 | Integration | All three models evaluated with the same settings (K = 4, same prompts) | `runs/final/*/eval` | 5 |

## 8. Expected Outcome
- **Direct:** whether 3× parameters or the data stream moves the student, at a small budget.
- **Measurable:** shelf validation loss, story grades, conversation probe, preschool exam — per model.
- **Reading the result:** 80M tokens is ~4% of what a 99M model would need (~20 tokens per parameter). The 65M new parameters will be far from used, so **no gain at 80M does not show that size does not help** — it bounds what a short run gives. A gain of the same-size arm points to data; a larger gain of the grown arm points to capacity.
- **Side effects:** the machine is busy ~36 h; the school does not run meanwhile.
- **Rollback:** none needed — the school's checkpoints are not touched.

## 9. Additional Information
- **Decision (2026-11-06, the user's):** the 80M budget is deliberate — a quick check of the direction, not training to the end. A full pass over the library (~7 days) only if the result is promising.
- **If more data is needed**, before any crawling: more FineWeb-Edu shards (one gave 4.4M tokens at grade ≤ 4; there are hundreds); a reading-level limit rising with each passed grade; synthetic child-level textbooks (Cosmopedia *young children*, or texts by the local teacher for the curriculum). Every new shelf goes through the 0001 filters and the exam leak check. Repeating a shelf helps up to ~4 passes, then less (Muennighoff et al., 2023).
- **Sources:** Chinchilla scaling (Hoffmann et al., 2022); data-constrained scaling (Muennighoff et al., 2023); function-preserving growth (0002).
- **Results:** pending.
