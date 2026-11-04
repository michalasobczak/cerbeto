# Cerbeto

*Cerbeto* (Esperanto: *cerbo* "brain" + *-et-* diminutive) — "little brain".

A small neural network of our own (8 → 30M parameters, starting from random weights) that learns English and reasoning "like a child at school": nursery → preschool → grade 1 → grade 2. It is taught by a large local LLM teacher, follows a day (lessons) / sleep (consolidation) / exam rhythm, and grows while it learns.

## Methodology
The project uses **[Rondo](https://workwithrondo.com)** as its reference engineering and project methodology — **CDD (Context-Driven Development)**:
- this README describes the "here and now" state: domain, functional architecture, critical path;
- `specs/` holds the intent of every change, each written in Rondo's nine-section structure (TL;DR, Abstract, Objective, Artefact Inventory, Functional Domain Changes, Critical Path, Control Gates, Expected Outcome, Additional Information);
- `CHANGELOG.md` records what was actually delivered;
- tests are the proof, derived from each specification's control gates;
- [MANIFESTO.md](MANIFESTO.md) is the engineering manifesto — implementation principles applied to every specification.

## Current state (2026-11-01)
- Specifications 0001–0007 written; **0001** (school library and tokenizer) and **0002** (student architecture and growth) delivered — 0001 G5 (human review of shelf samples) pending.
- **0003** nursery delivered for the growing student: 300M tokens in 15 h 42 min on the M1 Pro, grown 6.9M → 29.9M, validation loss 9.07 → 2.25. BLiMP 65.9% (gate 65%), conversation probe 80% (gate 60%, grading too lenient). It takes short, toddler-like conversational turns ("How old are you?" → "I got two.") but answers "yeah" even to false statements — learning truth and honest "I don't know" is the job of school (0004–0005). Still open: the control student run and the school baseline (preschool, grade 1, grade 2).
- **0004** curriculum and exams delivered: 16 skills across preschool, grade 1 and grade 2, 10,830 frozen exam tasks graded by code. Baseline after the nursery: no stage passed (5–40%), with a "yeah" and "I don't know" reflex. A first preschool smoke run (direct lessons, no teacher) is trained; its exam is pending. Open: the 30M limit leaves no room for growth in school (0005).
- School library downloaded offline into `data/library/` (outside git): BabyLM-2026-Strict, TinyStories V2, Simple English Wikipedia, FineWeb-Edu (1 shard) — about 4.8 GB.
- Shelves built (`data/shelves/`): stage 0 has 628M training tokens (speech 84M, stories 544M); stages 1–3: books 37M, world 0.8M, edu 4.4M.
- Own BPE tokenizer: 8192 tokens, 1.24 tokens/word on TinyStories.
- Student: 6.9M parameters at start, can grow to 30M without changing its behaviour; 30 min of training from scratch already writes simple coherent stories (validation loss 9.08 → 1.75). Training precision fp32; a 30M student trains within ~3 GB.
- Curriculum direction: preschool starts with **social skills** — conversation, politeness, agreeing/disagreeing, opinions and an honest "I don't know".
- Admin panel code: not written yet.
- Reference hardware: MacBook Pro M1 Pro, 32 GB — PyTorch uses the GPU via MPS (fp32/fp16, no bf16).
- Default teacher: `gemma4:e4b-mlx` in local Ollama, admitted after the teacher exam (`cerbeto teacher exam`: 97.5% correct answers, 94.8% consistent explanations).

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
| **0004 — curriculum and exams** | Task generators whose answers are known to the code: preschool social skills (agree/disagree by truth, honest "I don't know", opinions with a reason), a mini knowledge base (colours, animals, body parts, opposites, categories), then reading comprehension and counting (grade 1), order of events, cause–effect, numbers to 100 and one-step word problems (grade 2); frozen exam pools in three variants (`id`, `ood_hard`, `ood_combo`), ~10% unanswerable questions | No training yet — the **baseline**: an honest measurement of what both nursery students score on preschool, grade 1 and grade 2 before school (expected: low, with "yeah" and random facts graded as errors) | Exam tables per skill × variant; `runs/nursery/baseline.json` |
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
Smoke-train a fresh student for 30 minutes:
```bash
uv run cerbeto train --smoke --minutes 30
```

## License
Cerbeto — its concept, specifications and code — © 2026 Michał Sobczak.

Licensed under the **[PolyForm Noncommercial License 1.0.0](LICENSE.md)**: you may read, run, modify, fork and share it for any **noncommercial** purpose (personal projects, research, education, hobby), keeping the copyright notice. **Commercial use requires the author's written permission** — please get in touch via [GitHub](https://github.com/michalasobczak).

The school library is not part of this repository; its corpora keep their own licences (BabyLM — MIT, TinyStories — CDLA-Sharing-1.0, Wikipedia — CC BY-SA 4.0, FineWeb-Edu — ODC-By 1.0), recorded in each shelf's `manifest.json`.
