# 0012 — A larger vocabulary: 8,192 → 16,384 tokens

Status: in progress (2026-11-12) — G1 passed (measured: 16k shortens the 1B mix by 7.1%); 16k tokenizer trained and committed; the 1B shelves being re-tokenized and arm `v16k` queued on the M6 · Depends on: 0010, 0011 · Unblocks: 0013, the next large pretraining

## 1. TL;DR
v6 (`talk2/final`) moves to a **16,384-token byte-level BPE** trained on the 1B shelf mix. Every new token's embedding
starts as the mean of the 8k pieces it is made of, the first 5M tokens train only the embedding, then 55M tokens of
talk2's stream (every other step the 1B shelves, re-tokenized) train everything. Text gets 7–11% shorter on the
knowledge shelves and words such as *photosynthesis* (6 pieces), *Jupiter* (3) or *temperature* (2) become one token.

## 2. Abstract
The 8k vocabulary was chosen for the nursery (spec 0001) and is the smallest among models of this size (SmolLM-135M 49k,
MobileLLM-125M 32k). Spec 0010 §9 planned to widen it after the 1B run, before any larger pretraining, so that rare
words of the edu, textbook and Wikipedia shelves are learnt as words and the 512-token context holds more text.
Changing the vocabulary of a trained model is done by re-initialising only the embedding from the old one and training
it first (as in vocabulary-extension work for Llama and Mistral models); the rest of the network is kept.

## 3. Objective
- **Measure** (`cerbeto.tokenizer.vocab measure`, 200 MB sampled by the mix weights + teacher questions): bytes per
  token on each shelf's validation text, 5 MB per shelf:

  | shelf | 8k | 16k | 32k |
  |---|---|---|---|
  | edu | 3.754 | 4.156 (−9.7%) | 4.360 (−13.9%) |
  | textbooks | 4.145 | 4.608 (−10.0%) | 4.762 (−13.0%) |
  | dialogue | 4.029 | 4.216 (−4.4%) | 4.262 (−5.5%) |
  | stories | 4.035 | 4.092 (−1.4%) | 4.112 (−1.9%) |
  | speech | 3.488 | 3.604 (−3.2%) | 3.709 (−6.0%) |
  | books | 3.801 | 3.981 (−4.5%) | 4.177 (−9.0%) |
  | wiki | 3.512 | 3.942 (−10.9%) | 4.240 (−17.2%) |
  | **mix** | 3.897 | **4.196 (−7.1%)** | 4.332 (−10.0%) |

  **16k chosen:** +5.2M parameters (embedding 5% → 9% of the model) for −7.1%; 32k would add 15.7M for 3 more points.
- **Tokenizer** `tokenizer_16384.json`: byte-level BPE, 400 MB of the mix + the 17.4k teacher questions, the same
  special tokens with the same ids. `tokenizer.json` (8k) stays for every older checkpoint; a model loads the tokenizer
  of its own `vocab_size`.
- **Shelves:** the seven 1B shelves re-tokenized into `train_16384.bin` / `valid_16384.bin` next to the 8k files.
- **Transfer** (`vocab.transfer`): every weight but the embedding copied; each new token's embedding is the mean of the
  embeddings of the 8k pieces the old BPE cuts it into (tied head: the output row too).
- **Arm `v16k`** (`final_test.py`): 60M tokens; the first 5M only the embedding learns, at 10× the learning rate;
  then everything at talk2's schedule (peak 3e-5, cosine to 3e-6) and stream (every other step request → reply, the
  talk2 mix; the other the 1B shelves).
- **Comparison** across vocabularies in **bits per byte** (`vocab.bpb`), on the first 300 KB of each validation shelf.

**Out of scope:** a new pretraining, a vocabulary larger than 16k, Polish.

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `brain/cerbeto/tokenizer/vocab.py`: measure, train, tokenize, transfer, bpb | system | created |
| `brain/cerbeto/tokenizer/artifacts/tokenizer_16384.json` | data | created |
| tokenizer, shelf bins, `Talker`, `final_test` chosen by vocabulary size (8k unchanged) | system | changed |
| `final_test.py`: arm `v16k`, embedding-first phase | system | created |
| `brain/tests/test_vocab.py` | test | created |
| `data/shelves/*/train_16384.bin` (git-ignored) | data | being written |

## 5. Critical Path
1. Measure → 2. train the 16k tokenizer → 3. re-tokenize the 1B shelves → 4. bits per byte of v6 and of the transferred
model → 5. `final_test train v16k --tokens 60e6` → 6. bits per byte and spot check against v6 → 7. publish if not worse
(then spec 0013 starts from it).

## 6. Control Gates
| Gate | What | Pass |
|---|---|---|
| G1 | Measurement | 16k shortens the mix by ≥ 5% (passed: 7.1%) |
| G2 | Transfer | the transferred model, before training, says sensible words; its bits per byte are recorded |
| G3 | Result | bits per byte on every shelf ≤ v6's + 0.01; spot check: chat, completing and memory not worse than v6 by > 5 / 5 / 3 |

## 7. Expected Outcome
Bits per byte equal to v6 or slightly better on edu, textbooks and wiki; the spot check unchanged within the noise;
generation ~7% faster per word; a base for 0013 and for any larger pretraining.

## 8. Additional Information
- **Why now (2026-11-12):** planned in 0010 §9 for after the 1B run; the user asked for it after v6 was published.
- **The estimate was too high:** 15–25% fewer tokens was said on 2026-11-07; measured 10% on edu and textbooks, 7% on the mix.
