# 0018 — Twenty billion more tokens on the same shelves

Status: running (2026-11-20) · Depends on: 0016, 0017 · Unblocks: v7, 0013

## 1. TL;DR
After ten billion tokens (0016) the student was still not saturated: validation fell on every new shelf to the end, at
~117 tokens per parameter. Conversation tuning of that result (0017) did not give a v7. We let the same student read
**twenty billion more tokens** from `bil10/final`, on the same shelves in the same shares, so only the length changes.

## 2. Abstract
`bil10` read each shelf ~1.25 times; twenty billion more are ~2.5 passes, ~3.75 in all, still below the ~4 passes where
repetition starts to cost (Muennighoff et al., 2023). No new data has to be downloaded or built, so the four idle cards
start now. At the end the student will have read ~32.5B tokens, ~300 per parameter.

## 3. Objective
- **Arm `bil20`** (`final_test.py`): from `checkpoints/final/bil10/final`, **20B tokens**, `bil10`'s shelf shares,
  conversations every fourth step, learning rate re-warmed to 1e-4 over 2,000 steps, cosine to 1e-5; validation on
  `bil10`'s nine shelves.
- **Run** (`scripts/final/bil20_launch.sh`): copy `bil10/final` to every GPU machine, then DiLoCo as `bil10` (plain
  averaging, H = 500, three workers and a validator, checkpoints every 30 minutes). ~7 days at ~33k tokens/s.
- **Then:** conversation tuning (0017's recipe at the run's last learning rate) and the comparison with v6 and
  `bil10/final`.

**Out of scope:** new shelves (the next run after this one), a larger student, a different recipe.

## 4. Control Gates
| Gate | What | Pass |
|---|---|---|
| G1 | Re-warming, 1B | validation on every shelf back at or below `bil10/final` by 1B tokens |
| G2 | Result | validation below `bil10/final` on every new shelf (edu 12, textbooks, stories) by ≥ 0.05 |
| G3 | Answers | quick benchmark of `bil20/final` not below `bil10/final` (64); after tuning, above v6 (62) |

## 5. Additional Information
- **Why not new data first:** building new shelves takes a day or more of downloading and filtering; the shelves we
  have carry 20B more tokens within the repetition limit, and new data can be the next run.
- **If G1 fails** (validation still above `bil10/final` at 1B): stop and restart at a lower peak (3e-5).

## 6. Results
- (running)
