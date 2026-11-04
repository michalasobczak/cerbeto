# Changelog

What has actually been delivered. Intent of changes: `specs/`. Proof: tests.

## [Unreleased]

### Added
- CDD project structure: `README.md`, `MANIFESTO.md`, `CHANGELOG.md`, specifications 0001–0007 and the source document in `specs/`.
- `scripts/download_library.sh` — resumable download of library shelves into `data/library/` (outside git) — [0001].
- Project name: Cerbeto; specification refinements (fp32/fp16 precision without bf16, default teacher `nemotron-3.5-bolt:e4b`, nursery time budget on M1 Pro).
- Python brain project `brain/` (uv, Python 3.12, package `cerbeto`) with CLI `cerbeto library build|tokenize|report` and `cerbeto tokenizer train` — [0001].
- School library shelves built offline into `data/shelves/` (JSONL text + uint16 `.bin` tokens + `manifest.json` with licences): `s0_speech` 83.6M, `s0_stories` 544.5M, `s1_books` 37.3M, `s2_world` 0.8M, `s3_edu` 4.4M training tokens — [0001].
- Librarian: Flesch-Kincaid grade filter, clean-prose filter, paragraph/document deduplication, deterministic 0.5% validation split — [0001].
- Own byte-level BPE tokenizer, 8192 tokens incl. 9 special tokens (`brain/cerbeto/tokenizer/artifacts/tokenizer.json`); 1.241 tokens/word on TinyStories valid — [0001].
- Gate tests G1–G4, N1, N2 (27 tests, `uv run pytest` in `brain/`) — [0001].
- Student model `cerbeto.model.student`: transformer with prelude / recurrent core × K / coda, RoPE, SwiGLU, RMSNorm, tied embeddings, sampling; starts at 6.9M parameters (`config/student.yaml`) — [0002].
- Function-preserving growth operators (`grow_depth`, `grow_width`, `grow_model_dim`), MLP pruning, neuron renewal (continual backprop), optimizer-state transplant, growth log, 30M hard limit — [0002].
- EWC synaptic consolidation, safetensors checkpoints with architecture and optimizer state, activation checkpointing — [0002].
- `cerbeto train --smoke`: 30-min fp32 run on TinyStories cut validation loss 9.08 → 1.75 (−81%) at ~14k tokens/s; fp16 was 4.5% slower → fp32 stays — [0002].
- Gate tests for 0002 (G1–G3, G5, N1, N2): 53 tests; 30M student peaks at 3.05 GB with checkpointing — [0002].
- Specs 0003/0004: social skills training (conversation roles in nursery speech, polite routines, agree/disagree, opinions, honest "I don't know") — spec change only.
- Speech shelf rebuilt with conversation roles (CHILDES child → `<|student|>`, adults → `<|human|>`; Switchboard B/A) — [0003].
- Nursery training loop `cerbeto nursery run`: phased data mix (speech → speech + stories), gradient accumulation, growth on validation plateau along a fixed schedule, versioned checkpoints every 30 min, exact resume — [0003].
- Nursery exams `cerbeto exam nursery`: BLiMP (BabyLM filtered, 67 phenomena) and a 50-situation conversation probe graded by the local teacher; minimal Ollama teacher client — [0003].
- License: PolyForm Noncommercial 1.0.0 (`LICENSE.md`) — noncommercial use, modification and forks allowed; commercial use requires the author's permission.
- Nursery growth deadlines (60/120/180/240M tokens), learning-rate re-warm segment after each growth, 32 validation batches per shelf; nursery restarted (attempt 2) — [0003].
- Nursery completed for the growing student: 300M tokens in 15 h 42 min, grown 6.9M → 29.9M (4 growth steps), validation loss 9.07 → 2.25; BLiMP 65.9% (gate passed), conversation probe 80% (passed, but grading found too lenient — kept as a baseline only) — [0003].
- `cerbeto chat`: talk to a student checkpoint in the terminal, with probes `/story`, `/ask N` (stability), `/think` (answers at K = 1, 2, 4, 8), `/k`, `/temp`, one-shot `--say`, cursor-key line editing and history across sessions — [0003].
- School curriculum (stages 1–3: preschool, grade 1, grade 2): 16 task generators with known answers, a preschool mini knowledge base, three exam variants (`id`, `ood_hard`, `ood_combo`) and 10% unanswerable questions per pool — [0004].
- Grader by code (yes/no, numbers in digits or words, text with competitors, phrases, honesty rule) and `cerbeto exam build | check-leak | run`: 10,830 frozen exam tasks, 0 exam prompts in 1M training tasks — [0004].
- Baseline of the post-nursery student on the school exams: no stage passed (5–40% on code-graded skills) — [0004].
- `cerbeto school smoke`: short direct lessons from the generators with nursery replay, one stage at a time — a learnability check before 0005 — [0004].
- Specs 0003/0005 aligned on school order (preschool → grade 1 → grade 2); open question in 0005: the 30M limit leaves no room for growth in school — spec change only.
- `cerbeto top`: live terminal dashboard of both students (status, progress, ETA, validation loss trend, growth, exam) and the machine (RAM, swap, GPU, power, Ollama) — [0003].
- README: what the student can do after the nursery, and the next phases with their expected effects.
- `cerbeto teacher exam`: admission exam for an LLM teacher — solves and explains training tasks of every stage (never an exam prompt, every prompt logged), graded by code: correct answers and explanations consistent with the generator's truth; optional thinking per skill — [0005].
- Default teacher `gemma4:e4b-mlx` (97.5% correct, 94.8% consistent on 240 tasks; admitted by the owner's decision just under the 98% / 95% thresholds); `nemotron-3.5-bolt:e4b` not admitted (96.1% / 84.6%) — [0005].

- `cerbeto school run` / `status`: the automatic school — day (lessons with the teacher, rephrased questions checked for names, numbers, options and negation; on a mistake the teacher's explanation is used only when code verifies it, otherwise the generator's reference solution), hippocampus (SQLite, `data/brain.sqlite`), sleep (the day's experiences weighted by surprise up to ×3, replay of earlier stages and the nursery shelves, EWC), exam of the current and earlier stages, promotion or rollback (forgetting gate 2 pp), stage advance by the 0004 thresholds; every phase resumable, Ctrl+C safe — gates G1–G3 tested — [0005].
- Smoke check result (`smoke/v0001`, 876 steps, 28k preschool lessons, old exam set): preschool `id` 7–37% → 54–100%, `ood_combo` 5–37% → 11–93%; grades 1–2 unchanged — [0004].

### Fixed
- Curriculum knowledge base found wrong by the teacher exam: leaves can be yellow, red or brown; cats hiss and purr; birds chirp and sing; the sky can be grey; short has two opposites (tall, long); open ↔ close; no false statement uses a sound the animal really makes; ambiguous "Do you know …?" and "Can you do it now?" questions removed — exam set rebuilt (`d47409634843b5c83b42c61c`) — [0004].
- Grader: simple inflections of the accepted word count (roars, chirping, closed); a reason after "because"/"than" may name the other options — [0004].
