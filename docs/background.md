# Background

Reference material moved from the README on 2026-11-16: the school path (nursery → preschool → school with a local LLM teacher, paused since 2026-11-06), its domain and architecture, hypotheses, datasets, commands, and related work. Results by date: [CHANGELOG.md](../CHANGELOG.md).

## Summary as of 2026-11-16

**Where we are:** the deep, thin D student (106.8M) **passed preschool** on 2026-11-07 (stage 1 of 3), read **one billion tokens** of textbooks, Wikipedia and conversations, and was tuned to answer questions: **v6** on Hugging Face answers 117 of 160 sampled chat questions (v5: 55) and remembers every name it is told. It read **a second billion** (`bil2-avg`, level with v6) and will next read **ten billion** tokens of a new data set. The school student `v0017` (34.6M) is still in preschool.
- **Nursery done:** 300M tokens, 6.9M → 29.9M parameters, BLiMP 65.9%.
- **School running** (LLM teacher → sleep → exam, on a laptop, offline): preschool ~22% after the nursery → **60.5%** by day 7, flat on days 9–16, **68.5%** after day 18.
- **Plateau cause:** lessons were 97% right; the missing points were knowledge and question forms the lessons never gave.
- **Response (2026-11-05):**
  - harder days: weak skills more often, lessons mixing two skills, reading lessons that tell rare facts;
  - passing rule for moving between grades: mastery in every skill, transfer on average, no skill without transfer;
  - first growth in school: `v0016`, 34.6M, limit raised to 40M;
  - yes/no lessons think "claim → fact → decision" (the student knew the facts but answered "No. A dog says woof.").
- **Night bench (2026-11-06):** the night after day 17 replayed in 13 variants (learning rate, replay, consolidation, growth, thinking formats, core passes). None beyond chance (±3 pp from the seed alone); every night trades agree/disagree and categories for clarify, and the small validation set kept the start in most nights. Now a night keeps its end unless clearly worse, the exam decides, and earlier days are replayed once per prompt. Day 18 under these rules: **60.5% → 68.5%** (`v0017`) — rare opposites 25 → 85%, clarify 10 → 52%, agree/disagree 67 → 57% while its yes/no thinking changes.
- **Growth experiments (2026-11-06):** C — a wide 99M student, then a deep **283M** one on a new Mac mini M6 in bf16: no gain over `v0017` after 18M tokens (the unrolled blocks stayed copies, four with a dead attention). D — a deep, thin **~107M** student trained on conversations of several turns: after 69.4M tokens it **beats `v0017` on every measure** — conversation probe 2 → 8 of 10, stories on request 1.5 → 4.3 of 5, **preschool exam 68.5% → 76.3%, preschool passed** without a school night. Next: [0010](../specs/0010-one-billion-tokens.md), 1B tokens with new data.
- **One billion tokens (2026-11-07 → 11-11):** [0010](../specs/0010-one-billion-tokens.md) — D read textbooks, Wikipedia and conversations; every validation shelf fell, stopped at 982M when the loss was flat. Public since 2026-11-09 on Hugging Face as versions `v1`–`v5` with a chat in the browser. The chat did not grow with the tokens (v4 69, v5 55 of 160): the knowledge was there in "complete the sentence" form, not as answers — see [CHANGELOG](../CHANGELOG.md).
- **Conversation tuning (2026-11-11 → 11-12):** [0011](../specs/0011-conversation-tuning.md) — 50M tokens of questions with short answers (17,400 written by a local teacher), conversations that keep facts, "I don't know" about made-up words, every other step the 1B shelves. **v6-1032M:** chat 55 → 117 of 160, memory 19 → 40 of 40, nothing forgotten. At learning rate 1e-4 the first try forgot (shelf validation +0.2–0.3); 3e-5, where the 1B run ended, did not.
- **A larger vocabulary did not pay off yet (2026-11-12):** [0012](../specs/0012-larger-vocabulary.md) — moving v6 to 16,384 tokens (text 7% shorter) or extending its 8k by 4,096 frequent merges (5% shorter) and training 60M tokens left it behind v6 in bits per byte; the loss sat in the new, rarer words — see [CHANGELOG](../CHANGELOG.md).
- **A second billion tokens (2026-11-12 → 11-15):** [0014](../specs/0014-second-billion.md) — from v6 on the 16k vocabulary, mostly text the first billion did not reach (a new shelf of grade 7–8 educational paragraphs, ~440M tokens). Stopped on the M6 at 459M tokens because the way data is fed changes; the same recipe trained with DiLoCo ([0015](../specs/0015-diloco.md)) to 1B ends level with v6 — see [CHANGELOG](../CHANGELOG.md).
- **Ten billion tokens (running since 2026-11-16):** [0016](../specs/0016-ten-billion.md) — a new training on other commodity hardware, non-stop for ~3.5 days, from a new data set built where the training runs: 6.3B tokens of text (FineWeb-Edu grade 8–12 2.41B, Cosmopedia stories 1.12B and textbooks 0.88B, earlier shelves 1.87B), each read ~1.25 times — see [CHANGELOG](../CHANGELOG.md).
- **Next:** conversation tuning on the ten-billion result → v7; then honest answers ([0013](../specs/0013-honest-answers.md): answer what it knows, "I don't know" about the rest, instead of guessing).

## Hypotheses, datasets and what the student has seen

**Hypotheses**

| Hypothesis | State | Evidence |
|---|---|---|
| A small model trained from scratch on child-level text learns grammar and short conversation | upheld | BLiMP 65.9%, conversation probe ([0003](../specs/0003-nursery-stage-0-pretraining.md)) |
| Growth during the nursery keeps behaviour and saves compute | function preservation upheld (Δ loss ≈ 0 at every growth); saving compute not tested (control student dropped) | growth logs; [0002](../specs/0002-student-architecture-and-growth.md) |
| A small local LLM is a reliable teacher when code verifies it | upheld | teacher exam: `gemma4:e4b-mlx` 97.5% correct, 94.8% consistent; unverified explanations never used |
| Day → sleep → exam raises school skills without forgetting the nursery | partly upheld: preschool 22% → 68.5%, nursery language kept; stage not passed | `runs/school/cycles.jsonl`, [0005](../specs/0005-school-day-sleep-exam.md) §9 |
| Lessons in the training forms transfer to new forms and unseen items by themselves | refuted for this student | `id` high, `ood_hard` 10–57% on several skills; plateau days 9–16 |
| More parameters help the school plateau | open: on the bench the grown `v0016` behaves like `v0015` (new neurons unused); growth experiment C: wide 99M and deep 283M no better than `v0017` (283M after 18M tokens); **D: a deep, thin 107M with conversations beats it and passes preschool (76.3% vs 68.5%), the gain from data in the conversation format as much as from size** | [0005](../specs/0005-school-day-sleep-exam.md) §9 |
| The settings of a school night decide its result | refuted on the bench: 13 variants, all within ±3 pp of a different seed | [0005](../specs/0005-school-day-sleep-exam.md) §9, `runs/bench/` |
| Consolidation (EWC) holds what was learned | refuted as built: the nursery Fisher is almost zero; EWC on earlier lessons does not stop the trade between skills | [0005](../specs/0005-school-day-sleep-exam.md) §9 |
| More passes of the recurrent core at answer time help | not seen in school: validation 75.1 / 76.5 / 75.3 / 73.9% at K = 2 / 4 / 6 / 8 | [0005](../specs/0005-school-day-sleep-exam.md) §9 |

**Setup and datasets**

**Setup:** own BPE tokenizer (8192 tokens); teacher `gemma4:e4b-mlx` in local Ollama; nursery and school on a MacBook Pro M1 Pro, 32 GB (MPS, fp32); since 2026-11-06 the main machine is a Mac mini M6, 16 GB (MPS, bf16), the MacBook only develops code; library offline in `data/library/` (~4.8 GB), shelves in `data/shelves/`.

**Datasets** (built by the librarian of [0001](../specs/0001-library-and-tokenizer.md))

| Shelf | Stage | Training tokens | Sources | Filter | Licence |
|---|---|---|---|---|---|
| `s0_speech` | 0 nursery | 83.6M | CHILDES (child = `<\|student\|>`, adults = `<\|human\|>`), Switchboard, BNC spoken, OpenSubtitles — via BabyLM-2026-Strict | dialogue, no reading-level filter | MIT (BabyLM) + upstream terms |
| `s0_stories` | 0 nursery | 544.5M | TinyStories V2 (GPT-4 stories in a 3–4-year-old's vocabulary) | none | CDLA-Sharing 1.0 |
| `s1_books` | 1 preschool | 37.3M | Project Gutenberg children's books (via BabyLM) | none | public domain |
| `s2_world` | 2 grade 1 | 0.8M | Simple English Wikipedia (2023-11 dump + BabyLM simple_wiki) | paragraphs up to grade 4 (FK ≤ 4): 99.6% rejected | CC BY-SA 4.0 |
| `s3_edu` | 3 grade 2 | 4.4M | FineWeb-Edu, one shard of the 10BT sample, edu score ≥ 3 | paragraphs up to grade 4: 99.5% rejected | ODC-By 1.0 |
| school lessons | 1–3 | generated | 16 task generators of [0004](../specs/0004-curriculum-and-exams.md), rephrased and explained by the teacher | exam and validation prompts excluded | own |

**What the student has seen**

Training tokens by source (per school night: `runs/school/cycles.jsonl`, field `diet`).

| Phase | Tokens | By source | Result |
|---|---|---|---|
| A. Nursery (2026-10-31 → 11-01) | 300M | `s0_speech` 142.5M (1.7 passes), `s0_stories` 157.5M (0.29 passes) | BLiMP 65.9% |
| B. Preschool, school days 1–20 (2026-11-04 → 11-06) | 10.7M | today's lessons 2.4M, earlier days' lessons 2.5M, `s0_speech` 2.9M, `s0_stories` 2.9M | exam 22% → 68.5% (`v0017`) |
| C. Growth experiment (2026-11-06, stopped) | wide 99M: 4.8M; deep 283M: 18.3M | shelf windows two of three steps, request → reply pairs every third | no better than `v0017`; stories on request |
| D. ~100M with conversations (2026-11-06 → 07) | 69.4M | the same, a third of the pairs as conversations of 2–5 turns | better than `v0017` on every measure; preschool exam 76.3% |

Days 1–20 are estimated after the fact (hippocampus experiences × 141 night steps); from day 21 each night counts its tokens. Inputs only, padding excluded; lessons learn the student's turn only.

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

## Repository layout

```
README.md  MANIFESTO.md  CHANGELOG.md
specs/            change specifications (NNNN-slug.md) + source document and contracts
scripts/          library download, packaging; hf/ publishing to Hugging Face
brain/            brain — Python package `cerbeto`
panel/            admin panel — Rails (on hold, 0006)
config/           library, curriculum, architecture, rhythm, teachers
data/ runs/ checkpoints/   outside git
```

## Commands (school path and tools)

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
