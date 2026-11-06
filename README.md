# Cerbeto

*Cerbeto* (Esperanto: *cerbo* "brain" + *-et-* diminutive) — "little brain".

A small neural network of our own (6.9M → 34.6M parameters so far, limit 40M, starting from random weights) that learns English and reasoning "like a child at school": nursery → preschool → grade 1 → grade 2. It is taught by a large local LLM teacher, follows a day (lessons) / sleep (consolidation) / exam rhythm, and grows while it learns.

## Methodology
The project uses **[Rondo](https://workwithrondo.com)** as its reference engineering and project methodology — **CDD (Context-Driven Development)**:
- this README describes the "here and now" state: domain, functional architecture, critical path;
- `specs/` holds the intent of every change, each written in Rondo's nine-section structure (TL;DR, Abstract, Objective, Artefact Inventory, Functional Domain Changes, Critical Path, Control Gates, Expected Outcome, Additional Information);
- `CHANGELOG.md` records what was actually delivered, by date; the README's current state is derived from it;
- tests are the proof, derived from each specification's control gates;
- [MANIFESTO.md](MANIFESTO.md) is the engineering manifesto — implementation principles applied to every specification.

## Current state (2026-11-06)
Derived from [CHANGELOG.md](CHANGELOG.md) (what was delivered, by date) and the status line of each specification.

**Where we are:** the student is **in preschool** (school stage 1 of 3) and has not passed it yet.
- **Nursery done:** the growing student finished it (300M tokens, 6.9M → 29.9M parameters, BLiMP 65.9%).
- **School running:** the automatic school (day with the LLM teacher → sleep → exam) runs on a laptop, offline. The preschool score went from ~22% after the nursery to **60.5%** by day 7, then stayed flat for days 9–16; **68.5%** after day 18.
- **Cause of the plateau:** lessons were 97% right, and the missing points were knowledge and question forms that lessons never gave.
- **Response (2026-11-05):**
  - harder days: weak skills come up more often, lessons mix two skills, and new reading lessons tell the rare facts;
  - a passing rule fitted to moving between grades: mastery of what was taught in every skill, transfer on average, no skill without transfer;
  - the first growth in school: `v0016`, 34.6M parameters, limit raised to 40M;
  - yes/no lessons now think "claim → fact → decision": the student knew the facts but answered "No. A dog says woof.", which held agree/disagree and categories below 85%.
- **Night bench (2026-11-06):** the night after day 17 replayed offline in 13 variants (learning rate, replay, consolidation, growth, thinking formats, core passes). None moved the night beyond chance (±3 pp from the seed alone); every night trades agree/disagree and categories for clarify, and the small validation set kept the start in most nights, so their learning never reached the exam. Now a night keeps its end unless it is clearly worse, and the exam decides; earlier days are replayed once per prompt. The first such night (day 18) raised the exam **60.5% → 68.5%** (`v0017`): rare opposites 25 → 85%, clarify 10 → 52%; agree/disagree fell 67 → 57% while it changes its yes/no thinking.
- **Next:** the school runs again from day 18 with these rules; the next nights show whether the student gets through the yes/no format change toward passing; then grade 1 and grade 2.

**Specifications**

| Spec | Status |
|---|---|
| [0001](specs/0001-library-and-tokenizer.md) library and tokenizer | delivered; G5 (human review of shelf samples) pending |
| [0002](specs/0002-student-architecture-and-growth.md) student architecture and growth | delivered |
| [0003](specs/0003-nursery-stage-0-pretraining.md) nursery | delivered for the growing student; control student run open |
| [0004](specs/0004-curriculum-and-exams.md) curriculum and exams | delivered; passing rule changed 2026-11-05; G5 (human review of tasks) pending |
| [0005](specs/0005-school-day-sleep-exam.md) school: day, sleep, exam | **in progress**: the school runs (preschool, second run, day 18); open: neuron renewal, self-correction format, brain API, control student |
| [0006](specs/0006-rails-admin-panel-and-chat.md) admin panel and chat (Rails) | planned (a read-only school panel `cerbeto panel` exists in the brain) |
| [0007](specs/0007-offline-distribution.md) offline distribution | planned |

**Hypotheses**

| Hypothesis | State | Evidence |
|---|---|---|
| A small model trained from scratch on child-level text learns grammar and short conversation | upheld | BLiMP 65.9%, conversation probe ([0003](specs/0003-nursery-stage-0-pretraining.md)) |
| Growth during the nursery keeps behaviour and saves compute | upheld for function preservation (Δ loss ≈ 0 at every growth); saving vs a fixed student: open (control student not run) | growth logs; [0002](specs/0002-student-architecture-and-growth.md) |
| A small local LLM can be a reliable teacher when code verifies it | upheld | teacher exam: `gemma4:e4b-mlx` 97.5% correct, 94.8% consistent; unverified explanations are never used |
| Day → sleep → exam raises school skills without forgetting the nursery | partly upheld: preschool 22% → 68.5% (day 18), nursery language kept (EWC + replay); stage not passed yet | `runs/school/cycles.jsonl`, [0005](specs/0005-school-day-sleep-exam.md) §9 |
| Lessons in the training forms transfer to new forms and unseen items by themselves | refuted for this student | `id` high, `ood_hard` 10–57% on several skills; plateau days 9–16 |
| More parameters help the school plateau | open: tested since 2026-11-05 together with harder lessons; on the bench the grown `v0016` and `v0015` behave the same (new neurons not used yet) | [0005](specs/0005-school-day-sleep-exam.md) §9 |
| The settings of a school night (learning rate, replay share, consolidation, core passes) decide its result | refuted on the bench: 13 variants, all within the ±3 pp of a different seed | [0005](specs/0005-school-day-sleep-exam.md) §9, `runs/bench/` |
| Consolidation (EWC) holds what was learned | refuted as built: the nursery Fisher is almost zero (no effect); EWC on earlier lessons does not stop the trade between skills | [0005](specs/0005-school-day-sleep-exam.md) §9 |
| More passes of the recurrent core at answer time help | not seen in school: validation 75.1 / 76.5 / 75.3 / 73.9% at K = 2 / 4 / 6 / 8 | [0005](specs/0005-school-day-sleep-exam.md) §9 |

**Setup**
- **Data:** library offline in `data/library/` (BabyLM-2026-Strict, TinyStories V2, Simple English Wikipedia, FineWeb-Edu shard, ~4.8 GB). Shelves in `data/shelves/`, stage 0 with 628M tokens.
- **Tokenizer:** own BPE, 8192 tokens.
- **Teacher:** default `gemma4:e4b-mlx` in local Ollama.
- **Hardware:** MacBook Pro M1 Pro, 32 GB (MPS, fp32).

## What the student can do after the nursery
**Nursery exam** (`cerbeto exam nursery`, `checkpoints/growing/v0032`, 29.9M parameters, 2026-11-01):

| Exam | Result | Gate | Note |
|---|---|---|---|
| BLiMP — grammar, minimal pairs (BabyLM filtered, 67 phenomena) | **65.9%** | ≥ 65% — passed | best: anaphor agreement, principle A, wh-vs-that without a gap (96–100%); worst, below chance: long-distance dependencies, islands, NPI licensing (9–22%) |
| Conversation probe — 50 everyday situations graded by the teacher | **80%** fitting replies | ≥ 60% — passed | overrated: the teacher grades whether a reply fits as a turn, not whether it is true (*"Cats can fly, right?"* → *"yeah."* got 4/5); a strict re-reading counts about half the replies as fitting and true — kept as a baseline, not a blocking gate |

Details: [specs/0003](specs/0003-nursery-stage-0-pretraining.md) §9.

Observed outside the formal exam, in conversation with the same checkpoint. In short: **it has language, but not yet knowledge or a sense of truth.**

**Can do**
- **Tell stories** — its strongest skill. Given *"Once upon a time, there was a little dog named Max."* it continues for a whole paragraph with correct grammar, a protagonist, cause and effect (*"Max felt sad because…"*), quoted dialogue and a new character who asks *"what was wrong"*. It loses logic over longer stretches (*"a big, wet ball that looked like a ball"*), repeats sentences and jumps between places.
- **Hold a short conversation and keep its topic:**
  > Adult: Hi! What are you doing? — Child: I'm playing with the dolls. · Adult: Is it fun? — Child: yeah. I am. · Adult: Can I play too? — Child: yeah! I'm playing with the dolls. · Adult: What should we build? — Child: um we're making a castle. · Adult: Okay, bye bye! — Child: bye. we made a castle.

  Five turns on one topic, and it switches to the past tense once the play is over — nobody taught it that.
- **Polite routines:** greets, says goodbye, *"yes please"*, *"I'm sorry"*.
- **"Thinking longer" helps sometimes.** With the most likely answer (temperature 0) and more passes of the recurrent core: *"What does a cow say?"* → *"yeah."* at K = 1, *"moo."* at K = 2–8; *"The sky is green."* → repeated back at K = 1, *"The sky is blue."* at K = 2–8, *"The sky is green."* again at K = 12 (beyond the trained K ≤ 6 it starts to drift). For most of 15 probe questions the answer does not change with K. A first, weak sign that the recurrent core (0002) is used — not yet a measured effect.

**Cannot do yet**
- **Reliable facts.** Sampled answers are unstable: *"What color is the grass?"* asked five times → yeah / yellow / green / yellow / green. Its single most likely answer is often right for very common facts (sky blue, grass green) and *"I don't know."* for less common ones (*"What color is a banana?"*, *"How many legs does a dog have?"*); *"What does a dog say?"* → *"no."* or *"I don't know."*.
- **Tell true from false.** It has a strong *"yeah"* reflex on yes/no questions, including false statements (*"Cats can fly, right?"* → *"yeah."*). First signs of disagreement exist — *"The sky is green."* → *"is it blue?"* or *"blue."* — but not consistently.
- **Give reasons.** It has the form, not the content: *"Do you like dogs?"* → *"yeah."*, *"Why?"* → *"'cause I'm thirsty."*
- **Know itself or count:** *"Who are you?"* → *"no."*; *"Can you count to five?"* → *"no."* (true — it has not been taught).
- **Talk about a story.** It writes stories well, but asked *"What did the bear want?"* it answers *"I don't know."* — writing a story and answering questions about one are still separate skills.

## Next phases and their expected effects
| Phase | What happens | Expected effect on the student | How we will know |
|---|---|---|---|
| **0003 — control student** | The same 300M-token nursery for a fixed 30M student, no growth (~25 h, run separately) | Nothing new for the student itself — a reference point for the project's thesis | Its BLiMP and validation loss next to the growing student's; later the same school results |
| **0005 — school: day, sleep, exam** | The automatic loop: **day** — lessons with the local LLM teacher, experiences in the hippocampus; **sleep** — consolidation into the weights with replay, EWC and neuron renewal; **exam** — promotion, rollback or growth | The *"yeah"* reflex replaced by answers that follow the truth; *"I don't know"* only where the answer really is unknown; stable facts from the knowledge base; then reading, counting and simple reasoning — grade by grade, without forgetting the previous ones | **MVP priority:** `ood_hard` + `ood_combo` up by ≥ 20 pp versus the baseline; ≥ 85% on `id` per grade; forgetting ≤ 2 pp; growing vs control comparison |
| **0006 — admin panel and chat** | Local Rails panel: teachers, live lessons, progress, the student's brain, versions with compare and rollback, library, chat | You can watch lessons live, talk to the student and correct it — a human becomes an extra teacher | Every function within 2 clicks; live view ≤ 2 s delay |
| **0007 — offline distribution** | A package with the post-nursery student, curriculum, local teacher, learning tool and panel | The student keeps learning and growing at people's homes on a 16 GB Mac, fully offline; every copy becomes a separate child | Clean 16 GB Mac: install ≤ 30 min, 3 offline cycles, peak RAM ≤ 14 GB |

## Domain
| Term | Meaning |
|---|---|
| Student | our own network (transformer with a recurrent core), grows from ~8M to 30M parameters |
| Control student | same learning, fixed size of 30M — proof of whether growth helps |
| Teacher | large LLM (locally via Ollama) acting as homeroom teacher, tutor or assistant examiner |
| Library / shelves | text corpora matched to difficulty level (stages 0–3) |
| Lesson | a task from a generator (ground truth from code) + a dialogue with the teacher |
| Hippocampus | fast memory of the day's experiences (SQLite) |
| Sleep | consolidating experiences into weights + review + consolidation (EWC) + neuron renewal |
| Exam | independent grading by code: `id`, `ood_hard`, `ood_combo`; forgetting gate ≤ 2 pp |
| Version | immutable student state after every weight change; promotion or rollback |

## Functional architecture
```
                 ┌──────────────── panel/ (Rails 8, :3100) ────────────────┐
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
0001 library and tokenizer → 0002 student → 0003 nursery + baseline → 0004 curriculum and exams → 0005 school (MVP priority: `ood` improvement after learning cycles) → 0006 admin panel → 0007 offline distribution.

## Repository layout
```
README.md  MANIFESTO.md  CHANGELOG.md
specs/            change specifications (NNNN-slug.md) + source document and contracts
scripts/          scripts (library download, packaging)
brain/            brain — Python package `cerbeto`
panel/            admin panel — Rails (to be created, 0006)
config/           library, curriculum, architecture, rhythm, teachers
data/ runs/ checkpoints/   outside git
```

## Getting started (current state)
Download the library (resumable, ~5 GB):
```bash
./scripts/download_library.sh
```
Build shelves, train the tokenizer, tokenize and check the gates (from `brain/`):
```bash
uv run cerbeto library build && uv run cerbeto tokenizer train && uv run cerbeto library tokenize && uv run cerbeto library report
```
Run the tests (add `-m slow` for the GPU memory gate):
```bash
uv run pytest
```
Nursery training (resumable — re-run the same command after an interruption) and its exam:
```bash
uv run cerbeto nursery run --student growing
```
```bash
uv run cerbeto exam nursery --student growing
```
Watch training live — a `top`-like dashboard: progress and ETA, validation loss with its trend, growth steps, latest exam, and the machine (RAM, swap, GPU, power, whether Ollama is sharing the GPU). Read-only, so it never slows training down; `--student control`, `--interval 2`, `--once` for a single snapshot:
```bash
uv run cerbeto top
```
Talk to the student yourself (latest checkpoint; `--version v0032`, `--k`, `--temperature`, `--device cpu` to keep the GPU free during training):
```bash
uv run cerbeto chat
```
Inside the chat, plain text is a conversation the student remembers; commands probe one skill at a time:

| Command | What it checks |
|---|---|
| `/story Once upon a time, a little cat…` | storytelling: continues the beginning |
| `/ask 5 What color is the grass?` | stability: the same question five times |
| `/think The sky is green.` | "thinking longer": the most likely answer at K = 1, 2, 4, 8 |
| `/k 8`, `/temp 0` | passes of the recurrent core; temperature (0 = always the most likely answer) |
| `/reset`, `/info`, `/help`, `/quit` | forget the conversation, show the checkpoint, help, leave |
| ←/→, ↑/↓ | edit the line; browse earlier lines (history kept in `runs/chat_history`) |

One-shot, for scripts:
```bash
uv run cerbeto chat --say "Hello!" --say "Are you happy?"
```
School exams (spec 0004) — build the frozen pools once, check that lessons never repeat an exam prompt, then examine a student (`--limit 20` for a quick look):
```bash
uv run cerbeto exam build && uv run cerbeto exam check-leak
```
```bash
uv run cerbeto exam run --student growing
```
Smoke check: direct preschool lessons from the generators for 30 minutes (a separate student in `checkpoints/smoke/`), then the exams:
```bash
uv run cerbeto school smoke --stage preschool --minutes 30
```
Teacher admission exam (solves and explains tasks of every stage, graded by code):
```bash
uv run cerbeto teacher exam --per-skill 50
```
School (spec 0005): day with the LLM teacher → sleep → exam → promotion or rollback, preschool → grade 1 → grade 2,
on a copy of the nursery student in `checkpoints/school/`. Ctrl+C is safe at any moment; the same command resumes:
```bash
uv run cerbeto school run
uv run cerbeto school status
```
Web panel of the school (status, progress with ETA, exams vs the nursery baseline, days, lessons browser with the
teacher's explanations, log, machine) at http://127.0.0.1:8765 — read-only:
```bash
uv run cerbeto panel
```
Smoke-train a fresh student for 30 minutes:
```bash
uv run cerbeto train --smoke --minutes 30
```

## Related work

None of Cerbeto's building blocks is new; the combination seems to be. A short literature check (November 2026,
not a systematic review; several 2026 items are preprints known from their abstracts) found each axis studied
separately:

| Work | Shared with Cerbeto | Different |
|---|---|---|
| [CurLL](https://arxiv.org/abs/2510.13008) (BabyLM 2025) | developmental stages (~a year of childhood each), skill graph, continual learning, forgetting and transfer, a 135M model | 23.4B tokens of pre-generated LLM data; no teacher in the loop, no day/night, no exam gates, no growth |
| [LittleLearner](https://arxiv.org/abs/2608.13545) (2026) | a model trained from scratch on school (K–5) material only | 0.6–5B parameters, 88B tokens of plain pretraining; studies the knowledge boundary, not the learning process |
| [BabyLM interaction track](https://aclanthology.org/2025.babylm-main.28/), [ContingentChat](https://aclanthology.org/2025.babylm-main.25/) (2025) | a small student learns from a large LLM teacher in rounds | no code verification of explanations, no sleep, no growth, mostly one stage |
| [Language Models Need Sleep](https://arxiv.org/abs/2606.03979) (2026) | sleep as memory consolidation in language models | large pretrained models, dreaming through RL; no curriculum or exams |
| [Wake-Sleep Consolidated Learning](https://arxiv.org/abs/2401.08623) (NeurIPS 2024) | wake/sleep, short-term memory replayed at night | vision classification |
| [From Growing to Looping](https://arxiv.org/abs/2602.16490) (2026) | network growth together with a looped (recurrent) block | large-scale pretraining, no school curriculum or teacher |
| [Huginn / recurrent depth](https://arxiv.org/abs/2502.05171) (2025), [retrofitted recurrence](https://arxiv.org/abs/2511.07384) | prelude → recurrent core → coda, "think longer" with the same weights | 3.5B parameters and 800B tokens; no curriculum, no growth |
| [Toddler-level model](https://aclanthology.org/2025.sigdial-1.30.pdf) (SIGDIAL 2025), [BabyLM curricula](https://aclanthology.org/2025.babylm-main.36.pdf) | a small model speaking like a young child, staged child-directed data | pretraining only, no school with a teacher |

What we did not find elsewhere: one small model, grown from scratch, going to school in a day → sleep → exam loop
with an LLM teacher whose explanations are verified by code, with promotion and rollback of versions — on a laptop,
offline. Closest architecture: Huginn (recurrent depth); unlike it, Cerbeto also thinks in text (`<|think|>`) and
grows during learning. Its blocks are those of Llama/Qwen (RoPE, SwiGLU, RMSNorm).

## License
Cerbeto — its concept, specifications and code — © 2026 Michał Sobczak.

Licensed under the **[PolyForm Noncommercial License 1.0.0](LICENSE.md)**: you may read, run, modify, fork and share it for any **noncommercial** purpose (personal projects, research, education, hobby), keeping the copyright notice. **Commercial use requires the author's written permission** — please get in touch via [GitHub](https://github.com/michalasobczak).

The school library is not part of this repository; its corpora keep their own licences (BabyLM — MIT, TinyStories — CDLA-Sharing-1.0, Wikipedia — CC BY-SA 4.0, FineWeb-Edu — ODC-By 1.0), recorded in each shelf's `manifest.json`.
