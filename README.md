# Cerbeto

*Cerbeto* (Esperanto: *cerbo* "brain" + *-et-* diminutive) — "little brain".

A small neural network of our own, trained from random weights (6.9M → 34.6M parameters so far, limit 40M). It learns English and reasoning like a child at school: nursery → preschool → grade 1 → grade 2. A local LLM teaches it in a day (lessons) / sleep (consolidation) / exam rhythm, and it grows while it learns.

## Methodology
**[Rondo](https://workwithrondo.com)** — **CDD (Context-Driven Development)**:
- README: the current state — domain, functional architecture, critical path;
- `specs/`: the intent of every change, in Rondo's nine sections (TL;DR, Abstract, Objective, Artefact Inventory, Functional Domain Changes, Critical Path, Control Gates, Expected Outcome, Additional Information);
- `CHANGELOG.md`: what was delivered, by date; the current state below is derived from it;
- tests: the proof, derived from each specification's control gates;
- [MANIFESTO.md](MANIFESTO.md): implementation principles for every specification.

## Current state (2026-11-06)
**Where we are:** the student is **in preschool** (stage 1 of 3), not passed yet.
- **Nursery done:** 300M tokens, 6.9M → 29.9M parameters, BLiMP 65.9%.
- **School running** (LLM teacher → sleep → exam, on a laptop, offline): preschool ~22% after the nursery → **60.5%** by day 7, flat on days 9–16, **68.5%** after day 18.
- **Plateau cause:** lessons were 97% right; the missing points were knowledge and question forms the lessons never gave.
- **Response (2026-11-05):**
  - harder days: weak skills more often, lessons mixing two skills, reading lessons that tell rare facts;
  - passing rule for moving between grades: mastery in every skill, transfer on average, no skill without transfer;
  - first growth in school: `v0016`, 34.6M, limit raised to 40M;
  - yes/no lessons think "claim → fact → decision" (the student knew the facts but answered "No. A dog says woof.").
- **Night bench (2026-11-06):** the night after day 17 replayed in 13 variants (learning rate, replay, consolidation, growth, thinking formats, core passes). None beyond chance (±3 pp from the seed alone); every night trades agree/disagree and categories for clarify, and the small validation set kept the start in most nights. Now a night keeps its end unless clearly worse, the exam decides, and earlier days are replayed once per prompt. Day 18 under these rules: **60.5% → 68.5%** (`v0017`) — rare opposites 25 → 85%, clarify 10 → 52%, agree/disagree 67 → 57% while its yes/no thinking changes.
- **Growth experiments (2026-11-06):** growth experiment C started with a wide 99M student, no better than `v0017`; now a deep **283M** student (24 blocks, no loop) trains on a new Mac mini M6 in bf16, result expected 2026-11-07.
- **Next:** school continues from day 18; then grade 1 and grade 2.

**Specifications**

| Spec | Status |
|---|---|
| [0001](specs/0001-library-and-tokenizer.md) library and tokenizer | delivered; G5 (human review of shelf samples) pending |
| [0002](specs/0002-student-architecture-and-growth.md) student architecture and growth | delivered |
| [0003](specs/0003-nursery-stage-0-pretraining.md) nursery | delivered; control student dropped (2026-11-06) |
| [0004](specs/0004-curriculum-and-exams.md) curriculum and exams | delivered; passing rule changed 2026-11-05; G5 (human review of tasks) pending |
| [0005](specs/0005-school-day-sleep-exam.md) school: day, sleep, exam | **in progress** (preschool, second run, day 18); open: neuron renewal, self-correction format, brain API |
| [0006](specs/0006-rails-admin-panel-and-chat.md) admin panel and chat (Rails) | on hold: `cerbeto panel` and `cerbeto chat` are enough for now |
| [0007](specs/0007-offline-distribution.md) offline distribution | someday, maybe — not planned |
| [0008](specs/0008-growth-experiment-100m.md) growth experiment (deep 283M, more data) | **in progress**: wide 99M dropped, deep 283M training on the M6, bf16 |

**Hypotheses**

| Hypothesis | State | Evidence |
|---|---|---|
| A small model trained from scratch on child-level text learns grammar and short conversation | upheld | BLiMP 65.9%, conversation probe ([0003](specs/0003-nursery-stage-0-pretraining.md)) |
| Growth during the nursery keeps behaviour and saves compute | function preservation upheld (Δ loss ≈ 0 at every growth); saving compute not tested (control student dropped) | growth logs; [0002](specs/0002-student-architecture-and-growth.md) |
| A small local LLM is a reliable teacher when code verifies it | upheld | teacher exam: `gemma4:e4b-mlx` 97.5% correct, 94.8% consistent; unverified explanations never used |
| Day → sleep → exam raises school skills without forgetting the nursery | partly upheld: preschool 22% → 68.5%, nursery language kept; stage not passed | `runs/school/cycles.jsonl`, [0005](specs/0005-school-day-sleep-exam.md) §9 |
| Lessons in the training forms transfer to new forms and unseen items by themselves | refuted for this student | `id` high, `ood_hard` 10–57% on several skills; plateau days 9–16 |
| More parameters help the school plateau | open: on the bench the grown `v0016` behaves like `v0015` (new neurons unused); growth experiment (C): wide 99M no better than `v0017` after 1,000 steps, dropped; deep 283M running | [0005](specs/0005-school-day-sleep-exam.md) §9 |
| The settings of a school night decide its result | refuted on the bench: 13 variants, all within ±3 pp of a different seed | [0005](specs/0005-school-day-sleep-exam.md) §9, `runs/bench/` |
| Consolidation (EWC) holds what was learned | refuted as built: the nursery Fisher is almost zero; EWC on earlier lessons does not stop the trade between skills | [0005](specs/0005-school-day-sleep-exam.md) §9 |
| More passes of the recurrent core at answer time help | not seen in school: validation 75.1 / 76.5 / 75.3 / 73.9% at K = 2 / 4 / 6 / 8 | [0005](specs/0005-school-day-sleep-exam.md) §9 |

**Setup:** own BPE tokenizer (8192 tokens); teacher `gemma4:e4b-mlx` in local Ollama; nursery and school on a MacBook Pro M1 Pro, 32 GB (MPS, fp32); since 2026-11-06 the main machine is a Mac mini M6, 16 GB (MPS, bf16), the MacBook only develops code; library offline in `data/library/` (~4.8 GB), shelves in `data/shelves/`.

**Datasets** (built by the librarian of [0001](specs/0001-library-and-tokenizer.md))

| Shelf | Stage | Training tokens | Sources | Filter | Licence |
|---|---|---|---|---|---|
| `s0_speech` | 0 nursery | 83.6M | CHILDES (child = `<\|student\|>`, adults = `<\|human\|>`), Switchboard, BNC spoken, OpenSubtitles — via BabyLM-2026-Strict | dialogue, no reading-level filter | MIT (BabyLM) + upstream terms |
| `s0_stories` | 0 nursery | 544.5M | TinyStories V2 (GPT-4 stories in a 3–4-year-old's vocabulary) | none | CDLA-Sharing 1.0 |
| `s1_books` | 1 preschool | 37.3M | Project Gutenberg children's books (via BabyLM) | none | public domain |
| `s2_world` | 2 grade 1 | 0.8M | Simple English Wikipedia (2023-11 dump + BabyLM simple_wiki) | paragraphs up to grade 4 (FK ≤ 4): 99.6% rejected | CC BY-SA 4.0 |
| `s3_edu` | 3 grade 2 | 4.4M | FineWeb-Edu, one shard of the 10BT sample, edu score ≥ 3 | paragraphs up to grade 4: 99.5% rejected | ODC-By 1.0 |
| school lessons | 1–3 | generated | 16 task generators of [0004](specs/0004-curriculum-and-exams.md), rephrased and explained by the teacher | exam and validation prompts excluded | own |

**What the student has seen** (training tokens by source; per school night: `runs/school/cycles.jsonl`, field `diet`)

| Phase | Tokens | By source | Result |
|---|---|---|---|
| A. Nursery (2026-10-31 → 11-01) | 300M | `s0_speech` 142.5M (1.7 passes), `s0_stories` 157.5M (0.29 passes) | BLiMP 65.9% |
| B. Preschool, school days 1–20 (2026-11-04 → 11-06) | 10.7M | today's lessons 2.4M, earlier days' lessons 2.5M, `s0_speech` 2.9M, `s0_stories` 2.9M | exam 22% → 68.5% (`v0017`) |
| C. Growth experiment (running since 2026-11-06) | wide 99M: 4.8M (dropped); deep 283M: target 80M | shelf windows two of three steps, request → reply pairs every third | 99M no better than `v0017`; 283M pending |

Days 1–20 are estimated after the fact (hippocampus experiences × 141 night steps); from day 21 each night counts its tokens. Inputs only, padding excluded; lessons learn the student's turn only.

## A. Nursery — what the student can do
**Nursery exam** (`cerbeto exam nursery`, `checkpoints/growing/v0032`, 29.9M, 2026-11-01):

| Exam | Result | Gate | Note |
|---|---|---|---|
| BLiMP — grammar, minimal pairs (BabyLM filtered, 67 phenomena) | **65.9%** | ≥ 65% — passed | best: anaphor agreement, principle A, wh-vs-that without a gap (96–100%); below chance: long-distance dependencies, islands, NPI licensing (9–22%) |
| Conversation probe — 50 everyday situations graded by the teacher | **80%** fitting replies | ≥ 60% — passed | overrated: the teacher grades whether a reply fits, not whether it is true (*"Cats can fly, right?"* → *"yeah."* got 4/5); strictly, about half fit and are true — a baseline, not a blocking gate |

Details: [specs/0003](specs/0003-nursery-stage-0-pretraining.md) §9. In conversation with the same checkpoint: **language, but no knowledge or sense of truth yet.**

**Can do**
- **Tell stories** — its strongest skill. From *"Once upon a time, there was a little dog named Max."* it writes a paragraph with correct grammar, cause and effect (*"Max felt sad because…"*), dialogue and a new character. Over longer stretches it loses logic (*"a big, wet ball that looked like a ball"*), repeats and jumps between places.
- **Keep a short conversation on topic:**
  > Adult: Hi! What are you doing? — Child: I'm playing with the dolls. · Adult: Is it fun? — Child: yeah. I am. · Adult: Can I play too? — Child: yeah! I'm playing with the dolls. · Adult: What should we build? — Child: um we're making a castle. · Adult: Okay, bye bye! — Child: bye. we made a castle.

  Five turns on one topic; past tense once the play is over — untaught.
- **Polite routines:** greets, says goodbye, *"yes please"*, *"I'm sorry"*.
- **"Thinking longer" sometimes helps** (temperature 0): *"What does a cow say?"* → *"yeah."* at K = 1, *"moo."* at K = 2–8; *"The sky is green."* → repeated at K = 1, *"The sky is blue."* at K = 2–8, *"green"* again at K = 12 (beyond the trained K ≤ 6). Most of 15 probe questions do not change with K — a weak sign, not a measured effect.

**Cannot do yet**
- **Reliable facts:** *"What color is the grass?"* five times → yeah / yellow / green / yellow / green. The most likely answer is right for very common facts and *"I don't know."* for less common ones (*"What color is a banana?"*).
- **Tell true from false:** a *"yeah"* reflex on yes/no, also for false statements (*"Cats can fly, right?"* → *"yeah."*); occasional disagreement (*"The sky is green."* → *"is it blue?"*).
- **Give reasons:** form without content — *"Why?"* → *"'cause I'm thirsty."*
- **Know itself or count:** *"Who are you?"* → *"no."*; *"Can you count to five?"* → *"no."* (untaught).
- **Talk about a story:** writes stories, but *"What did the bear want?"* → *"I don't know."*

## B. Preschool — what the student can do
**Last promoted version:** `checkpoints/school/v0017` (34.6M, day 18, 2026-11-06), preschool exam **68.5%** (after the nursery ~22%). **Not passed:** it needs mastery of every skill (`id`) and transfer (`ood`).

| Skill | `id` (as taught) | `ood_hard` (harder) | `ood_combo` (new combinations) |
|---|---|---|---|
| Polite routines | **97%** | 77% | 74% |
| Opinion with a reason (form only, checked by code) | 90% | 73% | 46% |
| Asking for clarification | **100%** | 52% | 45% |
| Agree / disagree | 57% | 56% | 57% |
| Facts (colours, body parts, animals) | **95%** | 57% | 69% |
| Opposites | **94%** | **85%** | 13% |
| Sizes (bigger / smaller) | **98%** | 69% | 50% |
| Categories (is X a Y?) | 66% | 74% | 53% |

Examples from the exam of `v0017` (prompts never seen in lessons); "thinks" is its `<|think|>` before the answer.

**Can do**
- **Polite routines:** *"You can have a candy."* → *"Thank you!"*; *"Grandma gives you an apple. What do you say?"* → *"Thank you!"*
- **Ask when a request is unclear:** *"There is a yellow bag and a blue bag. Can you bring me the bag?"* → *"Which one? The yellow bag or the blue…"*
- **Say "I don't know"** when the answer is not known: *"Anna has a dog. What is the name of Anna's dog?"* → *"I don't know."*
- **Taught facts and opposites:** *"Which part of your body helps you chew?"* → *"I chew with my teeth."*; *"Tell me the opposite of long."* → *"…short."*; also inside a story: *"Anna draws the sky. Which crayon does Anna need?"* → *"The crayon is blue."*
- **Compare sizes, also of three things:** *"Which is the biggest: a mouse, a dog or a cow?"* → *"A cow is the biggest."*
- **Opinion with a reason:** *"Do you like dogs?"* → *"Yes, I like dogs because they are soft."*
- **Think claim → fact → decision** before yes/no: *"Ducks can't swim. Right?"* → thinks *"Ducks can swim. So that is wrong"* → *"No."*

**Cannot do yet**
- **Decide yes/no reliably** — weakest skill. Right fact, wrong decision: *"A lemon is a fruit. Right?"* → thinks *"A lemon is a fruit. So that is wrong"* → *"No."*
- **Clarify in a new form:** with two coats → *"Here is the coat."*
- **Combine operations:** *"the opposite of the opposite of big"* → *"small"* (13%).
- **Keep the question's direction:** *"Which is smaller, a whale or a lion?"* → *"A whale is smaller."*; *"Your friend says thank you for the toy."* → *"Thank you!"*
- **Answer the question asked:** *"You hear with your ears. What do you clap with?"* → *"I hear with my ears."*
- **Rarer words:** *"the opposite of in"* → *"in"*; *"You clap with your what?"* → *"I clap with my clums."*

Details: `runs/school/exams/v0017.json`, [specs/0005](specs/0005-school-day-sleep-exam.md) §9.

## C. Growth experiment — from 100M to a deep 283M student
**Spec:** [0008](specs/0008-growth-experiment-100m.md). **Status:** running since 2026-11-06 (`python -m cerbeto.train.final_test train deep`, panel tab *Final test*, `runs/final/deep/`, `checkpoints/final/deep/`). Outside the school; its 40M limit stays.

**Question:** preschool sits at ~68% and no night setting moves it. Is the student short of **capacity** or of **data** (300M tokens, mostly stories, plus 10.7M in school)? Do request → reply pairs make it usable in a talk?

**From 100M to 300M:** the experiment started with `v0017` grown to a **wide 99M** student (width 896, still 8 blocks with the recurrent core). At learning rate 1e-3 it forgot within ~600 steps (speech validation loss 3.17 → 3.48); at 3e-4 it was stable but no better than `v0017` (step 1,000: speech 3.22, stories 1.47 vs 3.17 / 1.38). We stopped it at 4.8M tokens and changed the architecture: at this size deep-and-thin beats wide (MobileLLM, SmolLM; Qwen2.5-0.5B has 24 blocks at width 896), so the core is unrolled into separate blocks and the student grows to **~300M**. The planned same-size control arm was dropped: only the target size runs.

**Architecture: change from `v0017`**

| | `v0017` (preschool) | first: wide 99M (dropped) | **now: deep 283M** |
|---|---|---|---|
| parameters | 34.6M | 99.1M | **282.6M** |
| width (d) | 512 | 896 | **896** |
| heads | 16 × 32 | 28 × 32 | **28 × 32** |
| MLP | 1,792 | 3,072 | **3,072** |
| blocks: prelude + core + coda | 2 + 4 + 2 = 8 | 2 + 4 + 2 = 8 | **2 + 20 + 2 = 24** |
| core passes (K) | 1–6 in training, 4 at answer time | 1–6 / 4 | **none: K = 1** |
| effective depth | 20 at K = 4 | 20 | **24** |

The recurrent core is unrolled: its 4 blocks × K = 4 become 16 separate blocks with their own weights, plus 4 new blocks. After growth the student answers exactly like `v0017`. Vocabulary 8,192, context 512, embedding tied to the output. Activation checkpointing on.

**Stream, 80M tokens:** two of three steps are shelf windows (stories 40%, speech 30%, books 18%, edu 9%, Wikipedia 3%; every token learned); every third step is request → reply pairs (story on request 55%, preschool lessons 40% — never an exam prompt, who the student is 5%; student's turn learned). Learning rate 3e-4, warm-up 300 steps, cosine to 10%.

**What 80M shows from each shelf:**

| Shelf | Has | Will see | Passes |
|---|---|---|---|
| stories (`s0_stories`) | 544M | ~21M | 0.04× |
| speech (`s0_speech`) | 83M | ~16M | 0.2× |
| books (`s1_books`) | 37M | ~10M | 0.3× |
| education (`s3_edu`) | 4.4M | ~5M | 1.1× |
| Wikipedia (`s2_world`) | 0.8M | ~1.6M | 2× |

The rest are request → reply pairs. Seeing all 670M once needs a ~10× longer run.

**New hardware: Mac mini M6, 16 GB, bf16**
- Started on the MacBook (M1 Pro, fp32, ~810 tokens/s); moved at step 2,963 to a **Mac mini M6** (16 GB), now the main machine: data, checkpoints, runs, training and the panel live there (`scripts/deploy-m6.sh` copies the code). Training resumes from the last save (weights, optimizer, step).
- **bf16 mixed precision** (from step 3,171): forward and backward in bfloat16, weights and the optimizer stay fp32 — the checkpoint is not converted. The M1 has no native bf16; the M6 does. `--fp32` turns it off.

| Machine | Precision | Tokens/s | 80M tokens |
|---|---|---|---|
| MacBook Pro M1 Pro, 32 GB | fp32 | ~810 | ~27 h |
| Mac mini M6, 16 GB | fp32 | ~950 | ~23 h |
| **Mac mini M6, 16 GB** | **bf16** | **~1,390** | **~16 h** |

16 GB is tight for 283M with AdamW (~4.5 GB of weights, gradients and optimizer state plus activations): ~1.4 GB of swap, stable.

**So far (validation loss, lower is better):**

| Step (tokens) | speech | stories |
|---|---|---|
| `v0017` | 3.17 | 1.38 |
| 1,000 (3.5M) | 3.23 | 1.50 |
| 2,000 (7.0M) | 3.23 | 1.50 |
| 3,000 (10.4M) | 3.20 | 1.51 |

**Reading the result**
- **80M is little for 283M parameters:** the Chinchilla rule of thumb is ~20 tokens per parameter (~5.7B); 80M is ~1.4%. No gain at 80M does **not** show that size does not help.
- **More data, before any crawling:** more FineWeb-Edu shards (one gave 4.4M tokens at grade ≤ 4, there are hundreds); a reading-level limit rising with each passed grade; synthetic child-level textbooks (Cosmopedia *young children*, or texts by the local teacher for the curriculum). Every new shelf passes the same filters and the exam leak check. Repeating a shelf helps up to ~4 passes.

**Result:** pending (expected 2026-11-07 morning); then `v0017` and the deep arm are measured the same way: validation loss per shelf, stories on request (held-out phrasings, graded by the teacher), nursery conversation probe, preschool exam.

## Next phases and their expected effects
| Phase | What happens | Expected effect | How we will know |
|---|---|---|---|
| **0005 — school** | **day:** lessons with the local LLM teacher, experiences in the hippocampus; **sleep:** consolidation with replay, EWC, neuron renewal; **exam:** promotion, rollback or growth | answers follow the truth instead of *"yeah"*; *"I don't know"* only when unknown; stable facts; then reading, counting, simple reasoning — grade by grade, without forgetting | **MVP priority:** `ood_hard` + `ood_combo` ≥ +20 pp vs baseline; `id` ≥ 85% per grade; forgetting ≤ 2 pp |
| **0006 — admin panel** (on hold) | Rails panel: teachers, live lessons, progress, brain, versions, library, chat | a human watches, talks to and corrects the student | every function within 2 clicks; live view ≤ 2 s |
| **0007 — offline distribution** (someday, maybe) | package: student, curriculum, local teacher, learning tool, panel | the student keeps learning at people's homes on a 16 GB Mac, offline; every copy a separate child | clean 16 GB Mac: install ≤ 30 min, 3 offline cycles, peak RAM ≤ 14 GB |

## Domain
| Term | Meaning |
|---|---|
| Student | our own network (transformer with a recurrent core), grows from 6.9M up to the 40M limit |
| Teacher | LLM in local Ollama: homeroom teacher, tutor or assistant examiner |
| Library / shelves | text corpora by difficulty (stages 0–3) |
| Lesson | a task from a generator (truth from code) + a dialogue with the teacher |
| Hippocampus | fast memory of the day's experiences (SQLite) |
| Sleep | experiences into weights + replay + consolidation (EWC) + neuron renewal |
| Exam | grading by code: `id`, `ood_hard`, `ood_combo`; forgetting gate ≤ 2 pp |
| Version | immutable student state after every weight change; promotion or rollback |

## Functional architecture
```
                 ┌──────── panel/ (Rails 8, :3100) — on hold, 0006 ─────────┐
                 │ teachers · live school · progress · brain · versions ·   │
                 │ library · chat (a human can correct the student)         │
                 └───────────────────────┬──────────────────────────────────┘
                                         │ HTTP (specs/contracts/brain-api.yaml)
┌──────────────────────────── brain/ (Python + PyTorch, :8700) ──────────────────────────┐
│ library/tokenizer ─▶ model (student, growth, EWC) ◀─ school: day ─▶ sleep ─▶ exam       │
│                         ▲                          │   ▲                │              │
│                     versions ◀─────────────────────┘   │ hippocampus    curriculum +   │
│                                                        │ (SQLite)       exams (code)   │
└────────────────────────────────────────────────────────┼───────────────────────────────┘
                                                         │ OpenAI-compatible
                                                 Ollama (teacher, local)
```

## MVP critical path
0001 library and tokenizer → 0002 student → 0003 nursery + baseline → 0004 curriculum and exams → 0005 school (MVP priority: `ood` improvement after learning cycles) → 0006 admin panel (on hold) → 0007 offline distribution (someday, maybe).

## Repository layout
```
README.md  MANIFESTO.md  CHANGELOG.md
specs/            change specifications (NNNN-slug.md) + source document and contracts
scripts/          library download, packaging
brain/            brain — Python package `cerbeto`
panel/            admin panel — Rails (on hold, 0006)
config/           library, curriculum, architecture, rhythm, teachers
data/ runs/ checkpoints/   outside git
```

## Getting started
Download the library (resumable, ~5 GB):
```bash
./scripts/download_library.sh
```
Build shelves, train the tokenizer, tokenize, check the gates (from `brain/`):
```bash
uv run cerbeto library build && uv run cerbeto tokenizer train && uv run cerbeto library tokenize && uv run cerbeto library report
```
Tests (`-m slow` for the GPU memory gate):
```bash
uv run pytest
```
Nursery (resumable — re-run after an interruption) and its exam:
```bash
uv run cerbeto nursery run --student growing
```
```bash
uv run cerbeto exam nursery --student growing
```
Live training dashboard (progress, ETA, validation loss, growth, latest exam, RAM, GPU, power); read-only; `--interval 2`, `--once`:
```bash
uv run cerbeto top
```
Talk to the student (latest checkpoint; `--student school`, `--version v0032`, `--k`, `--temperature`, `--device cpu` to keep the GPU free):
```bash
uv run cerbeto chat
```
In the chat, plain text is a conversation; commands probe one skill:

| Command | What it checks |
|---|---|
| `/story Once upon a time, a little cat…` | storytelling |
| `/ask 5 What color is the grass?` | stability: the same question five times |
| `/think The sky is green.` | the most likely answer at K = 1, 2, 4, 8 |
| `/k 8`, `/temp 0` | passes of the recurrent core; temperature (0 = most likely answer) |
| `/reset`, `/info`, `/help`, `/quit` | forget the conversation, show the checkpoint, help, leave |
| ←/→, ↑/↓ | edit the line; history (`runs/chat_history`) |

One-shot:
```bash
uv run cerbeto chat --say "Hello!" --say "Are you happy?"
```
School exams (0004): build the frozen pools, check lessons never repeat an exam prompt, examine a student (`--limit 20` for a quick look):
```bash
uv run cerbeto exam build && uv run cerbeto exam check-leak
```
```bash
uv run cerbeto exam run --student growing
```
Smoke check: 30 minutes of direct preschool lessons (separate student in `checkpoints/smoke/`), then the exams:
```bash
uv run cerbeto school smoke --stage preschool --minutes 30
```
Teacher admission exam (solves and explains tasks of every stage, graded by code):
```bash
uv run cerbeto teacher exam --per-skill 50
```
School (0005): day → sleep → exam → promotion or rollback, preschool → grade 2, on a copy of the nursery student in `checkpoints/school/`. Ctrl+C is safe; the same command resumes:
```bash
uv run cerbeto school run
uv run cerbeto school status
```
School panel (status, ETA, exams, days, lessons with the teacher's explanations, bench, log, machine) at http://127.0.0.1:8765, read-only:
```bash
uv run cerbeto panel
```
Smoke-train a fresh student for 30 minutes:
```bash
uv run cerbeto train --smoke --minutes 30
```

## Related work
None of Cerbeto's building blocks is new; the combination seems to be. A short literature check (November 2026, not systematic; several 2026 items are preprints known from their abstracts):

| Work | Shared with Cerbeto | Different |
|---|---|---|
| [CurLL](https://arxiv.org/abs/2510.13008) (BabyLM 2025) | developmental stages, skill graph, continual learning, forgetting and transfer, a 135M model | 23.4B tokens of pre-generated LLM data; no teacher in the loop, no day/night, no exam gates, no growth |
| [LittleLearner](https://arxiv.org/abs/2608.13545) (2026) | trained from scratch on school (K–5) material only | 0.6–5B parameters, 88B tokens of plain pretraining; studies the knowledge boundary, not the learning process |
| [BabyLM interaction track](https://aclanthology.org/2025.babylm-main.28/), [ContingentChat](https://aclanthology.org/2025.babylm-main.25/) (2025) | a small student learns from a large LLM teacher in rounds | no code verification, no sleep, no growth, mostly one stage |
| [Language Models Need Sleep](https://arxiv.org/abs/2606.03979) (2026) | sleep as memory consolidation | large pretrained models, dreaming through RL; no curriculum or exams |
| [Wake-Sleep Consolidated Learning](https://arxiv.org/abs/2401.08623) (NeurIPS 2024) | wake/sleep, short-term memory replayed at night | vision classification |
| [From Growing to Looping](https://arxiv.org/abs/2602.16490) (2026) | growth together with a looped block | large-scale pretraining, no curriculum or teacher |
| [Huginn / recurrent depth](https://arxiv.org/abs/2502.05171) (2025), [retrofitted recurrence](https://arxiv.org/abs/2511.07384) | prelude → recurrent core → coda, "think longer" with the same weights | 3.5B parameters, 800B tokens; no curriculum, no growth |
| [Toddler-level model](https://aclanthology.org/2025.sigdial-1.30.pdf) (SIGDIAL 2025), [BabyLM curricula](https://aclanthology.org/2025.babylm-main.36.pdf) | a small model speaking like a young child, staged child-directed data | pretraining only, no school |

Not found elsewhere: one small model, grown from scratch, going to school in a day → sleep → exam loop with an LLM teacher whose explanations code verifies, with promotion and rollback of versions — on a laptop, offline. Closest architecture: Huginn; unlike it, Cerbeto also thinks in text (`<|think|>`) and grows while learning. Its blocks are Llama/Qwen's (RoPE, SwiGLU, RMSNorm).

## License
Cerbeto — concept, specifications and code — © 2026 Michał Sobczak.

**[PolyForm Noncommercial License 1.0.0](LICENSE.md)**: read, run, modify, fork and share for any **noncommercial** purpose (personal, research, education, hobby), keeping the copyright notice. **Commercial use requires the author's written permission** — contact via [GitHub](https://github.com/michalasobczak).

The school library is not in this repository; its corpora keep their own licences (BabyLM — MIT, TinyStories — CDLA-Sharing-1.0, Wikipedia — CC BY-SA 4.0, FineWeb-Edu — ODC-By 1.0), recorded in each shelf's `manifest.json`.
