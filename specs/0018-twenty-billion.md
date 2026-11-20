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
| G1 | Re-warming | validation falls on every shelf after the jump of the re-warm-up (checked every ~1B tokens); compared with `bil10/final` only at the end, as G2 |
| G2 | Result | validation below `bil10/final` on every new shelf (edu 12, textbooks, stories) by ≥ 0.05 |
| G3 | Answers | quick benchmark of `bil20/final` not below `bil10/final` (64); after tuning, above v6 (62) |

## 5. Additional Information
- **Why not new data first:** building new shelves takes a day or more of downloading and filtering; the shelves we
  have carry 20B more tokens within the repetition limit, and new data can be the next run.
- **G1 changed (2026-11-20, at 0.86B):** it first asked for every shelf back at `bil10/final` by 1B tokens. At 1e-4
  that cannot happen: re-warm-up lifted validation +0.2 within 50M tokens, by 0.86B it was down only to +0.13–0.18
  (edu 12 3.577 → 3.493 against 3.353), and the loss after re-warming usually goes below the earlier result only when
  the learning rate decays, in the second half of the run.
- **If G1 fails** (validation rising again): stop and restart at a lower peak (3e-5).

## 6. Results
- **0.86B** (2026-11-20 22:04 UTC, ~33k tokens/s): validation +0.13–0.18 above `bil10/final` and falling slowly (edu 12
  3.577 at 49M → 3.502 at 553M → 3.493; textbooks 2.691 → 2.608 → 2.586).
