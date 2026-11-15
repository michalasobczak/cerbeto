# 0015 — DiLoCo: one model trained by several GPUs that rarely talk

Status: built and tested (2026-11-14) — used by 0016 · Depends on: 0014 (recipe, data, start checkpoint) · Unblocks: 0016

## 1. TL;DR
The M6 trains at ~2,550 tokens/s, so a billion tokens take ~4.5 days. Commodity GPUs in ordinary machines are faster,
but they share only a 1 GbE network. DiLoCo lets them train **one** model anyway: each GPU trains on its own for H
steps, then sends how far it moved; a small coordinator without a GPU averages the moves, takes one outer step and
sends the result back.

## 2. Abstract
Synchronous data parallelism averages gradients after every step and needs a fast interconnect. DiLoCo (Douillard
et al., 2023) syncs every H ≈ 500 inner steps: each worker keeps its own AdamW, the difference between the shared
weights and the worker's weights is a pseudo-gradient, and an outer optimizer applies their mean. The paper matches
synchronous training at ~500× less communication. The model has 112M parameters: a pseudo-gradient in bf16 is
~224 MB, ~2 s on 1 GbE per worker and round.

## 3. Objective
- **Coordinator** (`cerbeto/train/diloco.py coordinator`): small TCP server. Holds the shared weights θ, the outer
  momentum, the step counter and the global token and shelf-pass counts. Per round: waits for the workers'
  pseudo-gradients, averages them weighted by tokens, applies the outer step, sends the update (bf16; every copy of θ
  applies it the same way). Writes `runs/final/<name>/log.jsonl` and `state.json` in `final_test`'s format and saves
  θ + outer state every 30 minutes (`checkpoints/final/<name>/latest`), `final` at the end.
- **Worker** (`… worker --rank r`): the `final_test.train` loop unchanged in substance (same shelves, pairs, learning
  rate by **global** tokens, bf16), its own seed per rank. Every H steps: send θ − local weights, receive the update, go
  on. AdamW state stays local. A late worker gets the whole θ and only the rest of the round; the first round waits
  for every worker, later ones close a while after the first arrives.
- **Validator** (`… validator`): a GPU too weak to train fetches θ every few rounds and computes the arm's validation,
  so no worker stops for it; the coordinator logs the result.
- **Thermal pacer** (`final_test train --max-temp`): pauses between steps to keep a card with failing cooling under a
  temperature, without root.
- **Launch:** `scripts/gpu/diloco.sh` (machines in a local `hosts.env`), `scripts/gpu/setup.sh` per machine,
  `scripts/gpu/burn.py` a load test of one card.

**Out of scope:** synchronous DDP, overlapping communication (streaming DiLoCo), the M6 as a worker.

## 4. Tests (`brain/tests/test_diloco.py`)
- Round trip of a bf16 pseudo-gradient over the socket; the outer step against torch's Nesterov SGD.
- One worker, outer lr 1, momentum 0 → the same weights as plain training; the token-weighted mean of two workers.
- A round with two workers; a late worker gets θ and the rest of the round; finish and resume from `latest`; a step
  that is not a sync does nothing; the validator logs θ of the last round.
- The whole `final_test.train` loop saves under its run name; the thermal pacer; training after a sync moves the
  model, not θ.

## 5. Bugs found on the first runs (each now has a test)
- The workers ended the run themselves near the target and the last round hung: only the coordinator ends it.
- A late worker stayed late for good: it now does only the rest of the round, and the mean is weighted by tokens.
- In `final_test.train` a shelf name, then a parameter name, overwrote the run name at the first save.
- `vector_to_parameters` made the parameters views of the worker's θ, so after round 1 every pseudo-gradient was zero
  and the merged model only drifted on round 1's momentum: θ is now copied in (`load_vector`).

## 6. Outer step
With 1–3 workers moving the same way, Nesterov at lr 0.7 / momentum 0.9 overshot (up to ~7× each round's move; edu
validation 3.90 → 4.2 within 31M tokens). Plain averaging (lr 1, momentum 0) learns steadily; `diloco.sh` takes it
from `OUTER_LR=1 OUTER_M=0`.

## 7. Result
`bil2-avg`: the `bil2` recipe (0014) from `v16k/final`, 1B tokens with plain averaging, H = 500. At equal tokens its
validation stayed 0.09–0.15 nats per token below the M6's `bil2` on every shelf from 50M to 450M (the M6 run took a
bump at its warm start that it did not recover from), so DiLoCo costs no quality here. Measured on the M6 in bits per
byte at the end: level with v6 (edu 1.226 vs 1.258, textbooks 0.770 vs 0.753, dialogue 0.505 vs 0.518, speech and
stories within 0.006). It is the start of 0016.
