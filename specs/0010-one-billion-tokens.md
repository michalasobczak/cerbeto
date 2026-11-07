# 0010 — One billion tokens: new data for vocabulary, knowledge and conversations

Status: planned (2026-11-07) — waits for the evaluation of 0009 · Depends on: 0001, 0008, 0009 · Unblocks: decision on training the student to the Chinchilla budget (~2B tokens)

## 1. TL;DR
The ~107M student of 0009 (`mid`) keeps training for **1B tokens — 10 tokens per parameter**, half of the Chinchilla rule of thumb (~20 per parameter, ~2.1B), as a trial. To make those tokens worth it we first build **new shelves that widen vocabulary and knowledge** and **many more multi-turn conversations**, so no source is repeated more than a few times. Expected: much better conversations, the student keeps a conversation of several turns going, and attention heads that pick out more complex relations.

## 2. Abstract
0009 gave the first growth that beats `v0017` (speech validation 3.09 vs 3.17 after 69.4M tokens), and the loss was still falling at the end: the student is short of data, not of capacity. The Chinchilla rule of thumb (Hoffmann et al., 2022) puts a compute-optimal model at ~20 training tokens per parameter; for 107M that is ~2.1B. We start with half, 1B, to see whether the gain keeps coming before we spend the whole budget. The current library cannot feed 1B well: with 0009's mix the education shelf would be repeated ~20× and Wikipedia ~37×, which teaches the text by heart. The knowledge and vocabulary the school is missing are exactly in those small shelves. So this spec is mostly about **data**: more educational text, child-level textbooks, a wider slice of Simple Wikipedia and conversations of several turns, all through the same filters and exam-leak check.

## 3. Objective
**Expectation (the user's, 2026-11-07)**
- Chinchilla: ~20 tokens per parameter; we start with **10 per parameter = 1B tokens** as a trial.
- To be efficient the 1B needs **new training data that widens vocabulary and knowledge**.
- Expected result: **much better conversations**, **continuing a multi-turn conversation**, **more capacity in attention to identify more complex relations**.

**Goals**
- **Start:** `checkpoints/final/mid/final` (0009, 106.8M, 69.4M tokens seen). Same architecture (2 + 16 + 2 blocks, width 640, 20 heads, K = 1), bf16, activation checkpointing, Mac mini M6.
- **Budget:** 1B new tokens at ~2,630 tokens/s ≈ **4.4 days**; a checkpoint every 30 minutes, resumable.
- **No source repeated more than ~4 times** in the 1B (beyond that repetition stops helping — Muennighoff et al., 2023); the target is ≤ 2× for the knowledge shelves.
- **New shelves** (sizes are estimates, measured when built):

| Shelf | Source | Filter | Expected tokens | Licence |
|---|---|---|---|---|
| `s3_edu` widened | FineWeb-Edu, ~20 shards of the 10BT sample instead of 1 | edu score ≥ 3, paragraphs up to grade 6 (FK ≤ 6, was 4) | 150–250M | ODC-By 1.0 |
| `s2_world` widened | Simple English Wikipedia (2023-11 dump + BabyLM) | paragraphs up to grade 6 (was 4: 99.6% rejected) | 30–50M | CC BY-SA 4.0 |
| `s2_textbooks` new | Cosmopedia, audience *young children* (synthetic textbooks and stories) | FK ≤ 6 | 100–200M | Apache 2.0 |
| `s0_dialogue` new | open multi-turn dialogue sets (candidate: SODA, social conversations of ~8 turns) | turns kept, roles mapped to `<\|human\|>` / `<\|student\|>` | 100–300M | to check per set |
| conversations from code | `Pairs.dialogue` of 0009 widened: longer conversations (up to 8 turns), more recall templates, two-step questions ("Anna has a dog named Rex. … What is the name of Anna's dog?") | exam hashes excluded | generated on the fly | own |
| conversations from the teacher | `gemma4:e4b-mlx` on the mini writes child-level conversations of 4–8 turns while no training runs; code checks the format and drops anything near an exam prompt | exam hashes excluded | 5–10M | own |

- **Stream, 1B tokens** (proposal, fixed after the shelves are measured): shelf windows two of three steps — stories 25%, speech 15%, dialogue 15%, edu 20%, textbooks 15%, books 5%, Wikipedia 5%; every third step request → reply data, half of it conversations of 2–8 turns.
- **Vocabulary:** the tokenizer stays at 8,192 tokens (changing it would throw away the trained embeddings); "widening vocabulary" means many more distinct **words** seen in context. Measured: distinct words per shelf and words of the probes the student has seen.
- **Evaluation** of `v0017`, `mid` (0009) and the new student with the same settings: the 0009 measures plus
  - a **longer conversation probe**: 30 held-out conversations of 4–6 turns, the last reply checked by code (a name or fact from an earlier turn, who the student is after a story, a switch of subject);
  - a **relation probe**: 50 questions that need two facts joined ("Tom is taller than Ben. Ben is taller than Sam. Who is the shortest?", "Anna's dog is called Rex. What is Rex?");
  - **`/stats` on both probes**: share of heads that point at the content token the answer needs (the name, the compared thing), per block.

**Out of scope:** the full Chinchilla budget (~2.1B, a later decision); a new tokenizer; mixture of experts; changing the school.

**Success criteria**
- Conversation probe of 0009 ≥ 8 of 10; the longer conversation probe clearly above `mid` (target ≥ +20 pp).
- Relation probe clearly above `mid` and `v0017` (target ≥ +15 pp).
- In `/stats`: more heads pointing at the needed content token than in `mid`, in more blocks.
- Validation loss on every old shelf at least as good as `mid`; preschool exam and nursery language not worse than `v0017` by more than 2 pp / 0.05 loss.
- No shelf repeated more than 4× (logged per shelf).

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `config/library.yaml`: widened `s2_world`, `s3_edu`; new `s2_textbooks`, `s0_dialogue` | config | planned |
| library download and shelf building for the new sources (FineWeb-Edu shards, Cosmopedia, dialogue sets) | system | planned |
| teacher-written conversations on the mini (`gemma4:e4b-mlx`), format check | system | planned |
| `Pairs.dialogue` widened (up to 8 turns, two-step questions) | system | planned |
| `final_test.py`: arm for 1B from `mid`, per-shelf pass counter | system | planned |
| longer conversation probe (30), relation probe (50) | system | planned |
| panel: the new arm and probe columns | system | planned |
| `data/library/`, `data/shelves/` on the mini (~30–50 GB downloads, 113 GB free) | data | planned |
| `checkpoints/final/<arm>/`, `runs/final/<arm>/` | data | planned |

## 5. Functional Domain Changes
**Rules:**
- Every new source passes the 0001 filters (reading level, minimum length, dedup) and the exam-leak check (exam and validation hashes); a source without a clear licence is not used.
- Each shelf keeps its `manifest.json` (source, licence, filter, tokens, rejected share).
- Teacher-written conversations are data, not truth: they are used for form (turns, staying on topic, recalling earlier turns), and code drops any conversation that states a fact the curriculum checks and gets it wrong.
- No shelf is repeated more than 4× within the run; the stream stops sampling a shelf that reaches its limit and says so in the log.
- The training format stays one: `<|eos|>` once, then per turn `<|human|> …\n<|student|><|answer|> …<|eos|>`; only the student's turns are learned in request → reply data.

**Resilience:** a checkpoint every 30 minutes; the run resumes and can be extended toward ~2.1B without a restart.

## 6. Critical Path
1. Evaluation of 0009 done (gate) → 2. new sources downloaded and shelves built, sizes measured → 3. mix fixed so no shelf passes 4× → 4. conversations widened (code + teacher) → 5. probes written → 6. 1B tokens on the M6 (~4.4 days) → 7. evaluate `v0017`, `mid`, new → 8. `/stats` → 9. decision in §9: go on to ~2.1B or not.

**Completion condition:** results table, both new probes and the attention analysis in §9.

## 7. Control Gates
| # | Type | Condition | Verification | Blocks |
|---|---|---|---|---|
| G0 | Decision | 0009's conversation probe shows the format works (`mid` clearly above `v0017`); otherwise fix the conversation format first | `runs/final/eval_mid.json` | 2 |
| G1 | Data | Every new shelf: licence recorded, filters applied, no exam prompt (leak check passes) | `cerbeto library report`, `cerbeto exam check-leak` | 6 |
| G2 | Data | The mix reaches 1B with no shelf above 4 passes | dry run of the stream | 6 |
| G3 | Unit | Longer conversations and two-step questions encode as in 0009 (student turns only) | `tests/school/test_day.py` | 6 |
| G4 | Integration | 1B tokens reached | `runs/final/<arm>/state.json` | 7 |
| G5 | Integration | `v0017`, `mid` and the new student evaluated with the same settings | `runs/final/eval_*.json` | 9 |

## 8. Expected Outcome
- **Direct:** much better conversations — the student stays on the subject, answers the last turn, uses an earlier turn when asked ("What was the dog's name?"), says who it is after a story; it knows more words and more facts.
- **Measurable:** both conversation probes, relation probe, story grades, preschool exam, validation loss per shelf (old and new), heads pointing at content in `/stats`.
- **Reading the result:** validation still falling at 1B → go on to ~2.1B; conversations better but relations not → the data has few relations, add them before more tokens; nothing better than `mid` → data quality, not amount, is the limit.
- **Side effects:** the M6 is busy ~4.4 days plus downloads; the school does not run meanwhile; ~30–50 GB more on the mini's disk.
- **Rollback:** none needed — `mid` and the school's checkpoints are not touched.

## 9. Additional Information
- **Why new data (2026-11-07):** with 0009's mix over 1B, the shelves we have would be seen: stories 0.7×, speech 3.6×, books 5×, edu 20×, Wikipedia 37×.
- **Sources:** Chinchilla (Hoffmann et al., 2022); data-constrained scaling, up to ~4 epochs about as good as new data (Muennighoff et al., 2023); MobileLLM (Liu et al., 2024); Cosmopedia (Hugging Face, 2024); SODA (Kim et al., 2023); FineWeb-Edu (Penedo et al., 2024).
- **Beyond 1B — the road to a proper small model (2026-11-07, plan, not in scope of this spec):**
  - **Target:** the level of SmolLM-135M / MobileLLM-125M — fluent English, basic world knowledge, sensible conversations, simple instructions; not the reasoning of large models (the ceiling of this size).
  - **Tokens:** small models keep gaining well past Chinchilla (SmolLM-135M: 600B, SmolLM2-135M: 2T, MobileLLM-125M: 1T). Ours: **5–10B tokens** (~3–6 weeks on the M6 at ~2,600 tokens/s), with ~3–5B unique tokens; data quality matters more than amount.
  - **Mix for the base phase:** FineWeb-Edu ~50% (knowledge, vocabulary), Cosmopedia ~20% (textbooks in plain language), stories and speech ~15% (keep the child's language), conversations ~10% (SODA, smol-smoltalk, ours), Wikipedia ~5%; no source more than 4×.
  - **Three phases:** (1) **base**, ~90% of tokens, the mix above; (2) **annealing**, the last ~10%: the best data (textbooks, top-scored edu, conversations) while the learning rate decays to zero — a large gain for its share; (3) **conversation tuning**, 50–200M tokens of conversations and instructions, only the student's turns learned — this makes it a partner in conversation.
  - **Tokenizer, decide before the 5–10B run:** 8,192 tokens fit a child's language, but wide knowledge splits technical words into many pieces (slower learning, shorter effective context). Widen it to 16–32k once, before the big run; it costs embedding parameters and retraining part of the model.
  - **Specialist data later:** possible at any time from a checkpoint, as a new shelf with its own filter and licence, mixed in at ~20–30% to avoid forgetting; best after the general knowledge base, since specialist text is written for adults.
- **Results:** pending.
