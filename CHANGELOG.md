# Changelog

What has actually been delivered, by date (newest first). Intent of changes: `specs/`. Proof: tests.

## 2026-11-06 — night bench: what a school night does

### Added
- `cerbeto school bench run|summary|list`: a school night replayed offline from a copy of a version and lessons already in the hippocampus, one setting per variant (`config/bench.yaml`), validation per skill at every check (`runs/bench/`); 13 variants of the night after day 17 — none beyond chance (a seed alone: ±3 pp) — [0005].
- Panel: Bench tab — validation of every variant along the night, hypotheses with best / step / end, per-skill start → best → end, the running variant's log — [0005].
- Skills consolidation (EWC on earlier right lessons, per-lesson Fisher normalised to mean 1, `ewc_skills_lambda`); off — 1e-3…1e-1 did not change the night — [0005].
- Sleep: passes of the recurrent core per step configurable (`k_min`, `k_max`) — [0005].
- School nights count the tokens they show the student by source (`diet` in `runs/school/cycles.jsonl` and in the version's metadata: today's lessons, earlier days, earlier stages, nursery shelves); days 1–20 estimated after the fact — [0005].
- README: tables of the datasets and of what the student has seen in each phase.

### Changed
- Growth experiment (0008): the wide 99M arm stopped and replaced by a deep arm like Qwen2.5-0.5B — core unrolled (`growth.unroll_core`, same function at K = 1) plus 4 blocks: 24 blocks, 282.6M, no loop; ~450 tokens/s, ~7 GB GPU memory.
- Control student dropped (0003, 0005): not needed; growth is judged by its own logs and the school's results.
- 0006 Rails panel on hold — the brain's `cerbeto panel` and `cerbeto chat` are enough for now; 0007 offline distribution: someday, maybe.
- Spec 0008: growth experiment — `v0017` grown to 99.1M vs the same size, 80M tokens of shelves + request → reply pairs, evaluation and how to read it.
- README: the 100M experiment (v0017 grown to 99.1M vs the same size, 80M tokens of shelves + request → reply pairs) described, with how to read its result and where more data would come from.
- A night keeps its end when its best validation check is within 2 pp of it (`keep_tolerance`): the small validation set only refuses clearly worse nights, the exam decides; before, 9 of 13 bench nights kept their start and nothing reached the exam — [0005].
- Sleep replays each earlier prompt once, its latest lesson (`dedup_memories`: 1,739 instead of 18,770 memories, the same night on the bench) — [0005].

### Fixed
- Panel: the exam history shows every cycle, scrollable — [0005].

### Results
- First night under the new rules (day 18): exam 60.5% → **68.5%**, promoted `v0017`; `ood_hard` opposites 25 → 85%, clarify 10 → 52%, sizes 54 → 69%; agree/disagree 67 → 57% — [0005] §9.

### Found
- The nursery EWC has no effect (its Fisher sums to 1.7 over 34.6M weights: a batch-mean gradient squared); every night trades agree/disagree and categories for clarify already by step 35, whatever holds the weights — [0005] §9.

## 2026-11-05 — school: harder lessons, reading, new passing rule, growth

### Added
- Validation of a night logged per skill, with the change against the start of the night — [0005].
- Harder school days after a plateau at 60.5% (days 9–16): weaker skills on the last exam come up more often (`focus`), and four lesson-only mixes of two preschool skills (never an exam form, never a rare item) make about 20% of a day — [0005].
- Reading lessons `read.facts` (rare facts told in a text, then asked about) and three more lesson-only mixes practising forms next to the exam's held-out ones (`mix.say_back`, `mix.not_statement`, `mix.which_category`); per-skill lesson weight overrides — [0005].
- `cerbeto school grow --mlp N --heads N`: function-preserving growth of the active school version into a new active version; parameter limit raised 30M → 40M; `v0015` grown into `v0016` (29.9M → 34.6M, +384 MLP neurons per block, Δ loss 0.0) — [0005].

### Fixed
- One format of thinking for yes/no: lessons always learn the code's comparison (the tutor is not asked on yes/no), and older yes/no memories without it are retired from replay (`retired` column; still in the database) — [0005].
- Yes/no lessons think claim → fact → decision ("The question says: … So that is wrong, and the answer is no."); categories had no thinking, agree/disagree thinking never compared the fact with the claim — the student answered "No. A dog says woof." — [0005].
- Lesson weights of a grown version come from the last promoted version's exam (they came from the nursery exam) — [0005].

### Changed
- README current state rebuilt from this changelog: where the student is, status of every specification, hypotheses upheld / refuted / open; this changelog reorganised by date (Rondo).
- Passing a grade: every skill `id` ≥ 85%, the stage's mean `ood_hard` ≥ 70% and every skill `ood_hard` ≥ 50% (before: `ood_hard` ≥ 70% in every skill) — [0004].
- A night whose best validation is before sleep creates no version and no exam; it counts as a cycle without progress (before, such copies were "promoted" with the same score) — [0005].

## 2026-11-04 — school runs: teacher exam, day/sleep/exam loop, panel

### Added
- Default teacher `gemma4:e4b-mlx` (97.5% correct, 94.8% consistent on 240 tasks; admitted by the owner's decision just under the 98% / 95% thresholds); `nemotron-3.5-bolt:e4b` not admitted (96.1% / 84.6%) — [0005].
- `cerbeto school run` / `status`: the automatic school — day (lessons with the teacher, rephrased questions checked for names, numbers, options and negation; on a mistake the teacher's explanation is used only when code verifies it, otherwise the generator's reference solution), hippocampus (SQLite, `data/brain.sqlite`), sleep (the day's experiences weighted by surprise up to ×3, replay of earlier stages and the nursery shelves, EWC), exam of the current and earlier stages, promotion or rollback (forgetting gate 2 pp), stage advance by the 0004 thresholds; every phase resumable, Ctrl+C safe — gates G1–G3 tested — [0005].
- Smoke check result (`smoke/v0001`, 876 steps, 28k preschool lessons, old exam set): preschool `id` 7–37% → 54–100%, `ood_combo` 5–37% → 11–93%; grades 1–2 unchanged — [0004].
- `cerbeto panel`: local read-only web panel of the school — phase progress with ETA, stages, exam per skill vs the nursery baseline, exam history, days, skills, lessons browser (filters, search, the student's thinking, what it learns at night, the teacher's raw explanation and the original question when rephrased), log, machine; the hippocampus now also stores the teacher's raw reply and the original question — [0005].
- Sleep replays earlier days of the current stage (as many as today's experiences per step, earlier mistakes ×2), so a new night does not wash out what earlier nights taught — night 2 without it lost clarify and categories `ood` and was rolled back — [0005].
- Panel: one screen without scrolling; Exams tab — every exam task of any version or the nursery baseline with the expected and the given answer, filters, pool sizes; the overview explains what the latest exam was and its decision — [0005].
- Sleep keeps the weights with the best result on a stage validation set (checked at the start and every 35 steps; disjoint from the exam, excluded from lessons) — early stopping against overfitting to lesson forms — [0005].
- Homeroom teacher rephrases 60% of lessons, now also yes/no statements and categories, kept only if every knowledge-base word survives — [0005].
- School restarted from `growing/v0032` under the new rules; the first run is archived as `school-attempt1` (checkpoints, runs, hippocampus rows) — [0005].
- README: Related work — what is new in Cerbeto and the closest systems (CurLL, LittleLearner, BabyLM interaction, LMs Need Sleep, WSCL, Growing to Looping, Huginn).
- `cerbeto chat` shows the student's thinking (`<|think|>`) above its answer.

### Fixed
- Curriculum knowledge base found wrong by the teacher exam: leaves can be yellow, red or brown; cats hiss and purr; birds chirp and sing; the sky can be grey; short has two opposites (tall, long); open ↔ close; no false statement uses a sound the animal really makes; ambiguous "Do you know …?" and "Can you do it now?" questions removed — exam set rebuilt (`d47409634843b5c83b42c61c`) — [0004].
- Exam pools balance "yes" and "no" in every pool (a pool had 226 "no" and 18 "yes"); more rare category members; no "Is a car a toy?" statements (toy cars exist) — exam set rebuilt (`c907cac02b624b535c7eb4f2`) — [0004].
- Grader: a "no" that restates the false statement ("No. A cloud is green.") is wrong — found while browsing lessons in the panel; the school's `v0001` exam was regraded (56.1% → 55.1%) — [0004].
- Grader: simple inflections of the accepted word count (roars, chirping, closed); a reason after "because"/"than" may name the other options — [0004].

## 2026-11-01 — nursery results, curriculum and exams

### Added
- Nursery completed for the growing student: 300M tokens in 15 h 42 min, grown 6.9M → 29.9M (4 growth steps), validation loss 9.07 → 2.25; BLiMP 65.9% (gate passed), conversation probe 80% (passed, but grading found too lenient — kept as a baseline only) — [0003].
- `cerbeto chat`: talk to a student checkpoint in the terminal, with probes `/story`, `/ask N` (stability), `/think` (answers at K = 1, 2, 4, 8), `/k`, `/temp`, one-shot `--say`, cursor-key line editing and history across sessions — [0003].
- School curriculum (stages 1–3: preschool, grade 1, grade 2): 16 task generators with known answers, a preschool mini knowledge base, three exam variants (`id`, `ood_hard`, `ood_combo`) and 10% unanswerable questions per pool — [0004].
- Grader by code (yes/no, numbers in digits or words, text with competitors, phrases, honesty rule) and `cerbeto exam build | check-leak | run`: 10,830 frozen exam tasks, 0 exam prompts in 1M training tasks — [0004].
- Baseline of the post-nursery student on the school exams: no stage passed (5–40% on code-graded skills) — [0004].
- `cerbeto school smoke`: short direct lessons from the generators with nursery replay, one stage at a time — a learnability check before 0005 — [0004].
- `cerbeto top`: live terminal dashboard of both students (status, progress, ETA, validation loss trend, growth, exam) and the machine (RAM, swap, GPU, power, Ollama) — [0003].
- README: what the student can do after the nursery, and the next phases with their expected effects.
- `cerbeto teacher exam`: admission exam for an LLM teacher — solves and explains training tasks of every stage (never an exam prompt, every prompt logged), graded by code: correct answers and explanations consistent with the generator's truth; optional thinking per skill — [0005].

### Changed
- Specs 0003/0005 aligned on school order (preschool → grade 1 → grade 2); open question in 0005: the 30M limit leaves no room for growth in school — spec change only.

## 2026-10-31 — library, tokenizer, student, nursery loop

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
- Speech shelf rebuilt with conversation roles (CHILDES child → `<|student|>`, adults → `<|human|>`; Switchboard B/A) — [0003].
- Nursery training loop `cerbeto nursery run`: phased data mix (speech → speech + stories), gradient accumulation, growth on validation plateau along a fixed schedule, versioned checkpoints every 30 min, exact resume — [0003].
- Nursery exams `cerbeto exam nursery`: BLiMP (BabyLM filtered, 67 phenomena) and a 50-situation conversation probe graded by the local teacher; minimal Ollama teacher client — [0003].
- License: PolyForm Noncommercial 1.0.0 (`LICENSE.md`) — noncommercial use, modification and forks allowed; commercial use requires the author's permission.
- Nursery growth deadlines (60/120/180/240M tokens), learning-rate re-warm segment after each growth, 32 validation batches per shelf; nursery restarted (attempt 2) — [0003].

### Changed
- Specs 0003/0004: social skills training (conversation roles in nursery speech, polite routines, agree/disagree, opinions, honest "I don't know") — spec change only.
