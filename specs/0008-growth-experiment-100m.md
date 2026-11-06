# 0008 — Growth experiment: a 100M student and more training data

Status: closed (2026-11-06) — no gain: the wide 99M arm stopped at 4.8M tokens, the deep 283M arm at 18.3M tokens; same-size arm dropped; continued in [0009](0009-deep-100m-conversations.md) · Depends on: 0002, 0004, 0005 · Unblocks: 0009

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
| `brain/cerbeto/train/final_test.py` (`train grown|control|deep`, `eval`; bf16) | system | created |
| Panel tab *300M* (was *Final test*; `brain/cerbeto/panel.py`, `panel.html`) | system | created |
| `cerbeto chat /stats` (`brain/cerbeto/talk_stats.py`): certainty per token, attention per block | system | created |
| `checkpoints/final/deep/latest` (step 5,243) | data | kept; grown arm deleted, control never run |
| `runs/final/deep/` (state, log), `runs/final/eval_v0017.json` | data | kept |

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
- **Change (2026-11-06, the user's decision): deep instead of wide.** At ~100M, deep-and-thin models beat wide-and-shallow ones (MobileLLM, 2024: 30 blocks at d 576 over 12 at d 768; SmolLM-135M follows it), and the student's errors (question forms, multi-step) and the flat K curve point away from more width. The wide 99M arm (8 blocks, 20 effective at K = 4) was stopped at 4.8M tokens and its progress deleted. New **deep arm**, closer to Qwen2.5-0.5B (24 blocks at d 896):
  - `v0017` grown to d 896, 28 heads, MLP 3072 (as before), then the recurrent core **unrolled**: its 4 blocks run 4 times become 16 separate blocks with the same weights (`growth.unroll_core`), so at K = 1 it computes exactly what `v0017` did at K = 4; then 4 new core blocks with zero output (`grow_depth`): **2 + 20 + 2 = 24 blocks, 282.6M parameters**, Δ loss 0.0 at every step.
  - No loop: K = 1 in training and evaluation.
  - Measured on the M1 Pro: ~450 tokens/s with activation checkpointing, so 80M tokens take **~50 h**; GPU memory ~7 GB (driver), so it fits a 16 GB Mac.
  - The 34.6M same-size arm stays as the comparison.
- **Decision (2026-11-06, the user's):** the 80M budget is deliberate — a quick check of the direction, not training to the end. A full pass over the library (~7 days) only if the result is promising.
- **If more data is needed**, before any crawling: more FineWeb-Edu shards (one gave 4.4M tokens at grade ≤ 4; there are hundreds); a reading-level limit rising with each passed grade; synthetic child-level textbooks (Cosmopedia *young children*, or texts by the local teacher for the curriculum). Every new shelf goes through the 0001 filters and the exam leak check. Repeating a shelf helps up to ~4 passes, then less (Muennighoff et al., 2023).
- **Sources:** Chinchilla scaling (Hoffmann et al., 2022); data-constrained scaling (Muennighoff et al., 2023); function-preserving growth (0002).
- **Change (2026-11-06, the user's decision): no same-size arm** — only the target size runs.
- **Learning rate:** at 1e-3 the grown 99M student forgot within ~600 steps (speech validation 3.17 → 3.48, stories 1.38 → 1.80, lessons 0.09 → 0.23); restarted at **3e-4** (continued pretraining needs a low rewarm rate, Gupta et al., 2023).
- **New hardware (2026-11-06):** the deep arm moved at step 2,963 to a **Mac mini M6, 16 GB**, now the main machine (data, checkpoints, training, panel; code arrives by `git push`). **bf16 mixed precision** from step 3,171 (autocast; weights, optimizer and checkpoint stay fp32): M1 Pro fp32 ~810, M6 fp32 ~950, **M6 bf16 ~1,390 tokens/s**; ~1.4 GB swap on 16 GB, stable.
- **Results (deep arm stopped at step 5,243, 18.3M tokens):**

  | Validation loss | speech | stories |
  |---|---|---|
  | `v0017` | 3.17 | 1.38 |
  | step 1,000 (3.5M) | 3.23 | 1.50 |
  | step 2,000 (7.0M) | 3.23 | 1.50 |
  | step 3,000 (10.4M) | 3.20 | 1.51 |
  | step 4,000 (13.9M) | 3.20 | 1.50 |

  - The loss on request → reply pairs stayed at ~1.25 from step 100.
  - **Chat at step 3,875** (`v0017` → deep): "tell me a story" → "Thank you!" → a real story ("Once upon a time, there was a blue bird…"); "tell me a story about a dog named Max" → "A dog is white." → "Max was a big dog…". But after a story, "who are you?" → it goes on with the story ("Tom. Tom was very fast…"), and "what color is the sky?" in that conversation → "blue. The blue bird was red…": the pairs taught one turn, never a conversation.
  - **Where the capacity went** (`/stats` on "what color is the sky?", weight norms): both models answer "The sky is blue." at 98%. The 16 unrolled blocks were still copies of `v0017`'s four (q norms 38.50 / 38.52 / 38.36 / 38.30); four of them — core 3 and its copies — a **dead attention**: q and k ~3.5× smaller than in the other blocks (~10.8 vs ~37), attention spread evenly over every token (spread 1.00) in every copy. Of 672 heads, 3 pointed at one token (all at `<|answer|>`), none at "sky" or "color"; the most attended token was `<|human|>` (17.7%). The answer comes from memory in the MLPs, not from reading the question.
- **Decision (2026-11-06, the user's):** stop the 283M arm; grow `v0017` again to a deep, thin **~100M** student with copies that start apart, the dead attention renewed, and conversations of several turns in the data → [0009](0009-deep-100m-conversations.md).
