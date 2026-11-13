# 0015 — DiLoCo: one model on four GPUs in two machines

Status: draft (2026-11-13), awaiting the user's agreement · Depends on: 0014 (recipe, data, start checkpoint) · Unblocks: longer runs (2–5B tokens)

## 1. TL;DR
Two GPU machines with 2× RTX 3060 12 GB each train **one** model together. Each GPU trains on its own for H steps,
then sends how far it moved; a coordinator without a GPU averages the four moves, takes one outer optimizer step and
sends the result back. On 2026-11-13 one 3060 ran the `bil2` recipe at ~8,200 tok/s and two at ~16,400 (M6: ~2,550),
so four should give ~30,000: **1B tokens in ~10 hours instead of ~4.5 days**.

## 2. Abstract
Synchronous data parallelism averages gradients after every step and needs a fast interconnect; the machines share
only 1 GbE (~111 MB/s measured). DiLoCo (Douillard et al., 2023) syncs every H ≈ 500 inner steps: each worker keeps
its own AdamW, the difference between the shared weights and the worker's weights is a pseudo-gradient, and the outer
optimizer (Nesterov, lr 0.7, momentum 0.9) applies their mean. The paper matches synchronous training at ~500× less
communication. The model has 112M parameters: a pseudo-gradient in bf16 is ~224 MB, ~2 s on the link.

## 3. Machines (measured 2026-11-13)
| host | LAN | role | GPU | CPU, RAM |
|---|---|---|---|---|
| 10.10.12.13 | 172.31.115.13 | coordinator | none | 8 threads, 31 GB |
| 10.10.12.14 | 172.31.115.1 | 2 workers | 2× RTX 3060 12 GB | Xeon X5660, 31 GB |
| 10.10.12.15 | 172.31.115.2 | 2 workers | 2× RTX 3060 12 GB | Xeon X5690, 31 GB |

Ubuntu 24.04, driver 580, no sudo (everything in `~`), torch 2.15 + CUDA 13.2 works on .14 with bf16. The bil2 recipe
used 3.6 of 12 GB, so activation recomputation (needed on the 16 GB M6) can be off here.

## 4. Objective
- **Coordinator** (`cerbeto/train/diloco.py coordinator`): small TCP server on .13. Holds the shared weights θ, the
  outer momentum, the step counter and the global token and shelf-pass counts. Per round: waits for the workers'
  pseudo-gradients (timeout: a late or dead worker is left out of that round), averages, applies the outer step,
  serves the new θ. Writes `runs/final/<arm>/log.jsonl` and `state.json` in today's format and saves θ + outer state
  every 30 minutes (`checkpoints/final/<arm>/latest`), `final` at the end.
- **Worker** (`… worker --rank r`): the `final_test.train` loop unchanged in substance (same shelves, pairs, lr
  schedule by **global** tokens, bf16), its own seed per rank, so the four read different windows. Every H steps:
  send θ_start − θ_local (bf16), receive θ, go on. AdamW state stays local.
- **Validation** on a worker of rank 0 right after a sync, on θ, the 0014 shelves, every ~1,000 inner steps.
- **Launch:** `scripts/gpu/diloco.sh` starts the coordinator and four workers (`CUDA_VISIBLE_DEVICES` 0/1 per host).
- **Panel:** the coordinator's run folder is copied to the mini (via the MacBook) so the panel shows the run.

**Out of scope:** synchronous DDP, overlapping communication (streaming DiLoCo), a larger model, the M6 as a worker.

## 5. Tests
- One worker, outer lr 1.0, momentum 0 → bit-for-bit the same weights as plain training for the same steps (CPU).
- Two fake workers on CPU: the coordinator's mean and Nesterov step against a hand computation; a worker that misses
  the timeout is left out and catches up with the next θ.
- Round trip of a bf16 pseudo-gradient over the socket; resume of the coordinator from `latest`.

## 6. Experiment
- **D1, does DiLoCo cost quality?** From `v16k/final`, bil2 recipe, 100M tokens: (a) one GPU, plain (~3.5 h);
  (b) four GPUs, DiLoCo, H = 500 (~1 h). Pass: (b) within 0.02 of (a) in validation loss on edu and textbooks at
  100M tokens, and ≥ 3× the tokens per hour.
- **D2, speed knobs** (only if D1 passes): activation recomputation off; H = 250 / 1,000.
- **D3, the real run:** the user decides what (e.g. bil2's goal at 1–2B tokens) and what happens to bil2 on the M6.

## 7. Risks
- Shared machines: other jobs on the GPUs slow one worker; the timeout keeps the others going.
- A worker restart loses its AdamW state (a short bump, as at a warm restart); acceptable.
- 1 GbE into the coordinator: 4 × 224 MB in, 4 × 224 MB out ≈ 16 s per round, ~7% of a ~220 s round at H = 500;
  H = 1,000 halves it.
