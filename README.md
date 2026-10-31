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

## Current state (2026-10-31)
- Specifications 0001–0007 written; **0001** (school library and tokenizer) and **0002** (student architecture and growth) delivered — 0001 G5 (human review of shelf samples) pending. **0003** nursery in progress: the growing student is learning to talk (300M tokens, growing 6.9M → 29.9M on the way; ~15 h on the M1 Pro).
- School library downloaded offline into `data/library/` (outside git): BabyLM-2026-Strict, TinyStories V2, Simple English Wikipedia, FineWeb-Edu (1 shard) — about 4.8 GB.
- Shelves built (`data/shelves/`): stage 0 has 628M training tokens (speech 84M, stories 544M); stages 1–3: books 37M, world 0.8M, edu 4.4M.
- Own BPE tokenizer: 8192 tokens, 1.24 tokens/word on TinyStories.
- Student: 6.9M parameters at start, can grow to 30M without changing its behaviour; 30 min of training from scratch already writes simple coherent stories (validation loss 9.08 → 1.75). Training precision fp32; a 30M student trains within ~3 GB.
- Curriculum direction: preschool starts with **social skills** — conversation, politeness, agreeing/disagreeing, opinions and an honest "I don't know".
- Admin panel code: not written yet.
- Reference hardware: MacBook Pro M1 Pro, 32 GB — PyTorch uses the GPU via MPS (fp32/fp16, no bf16).
- Default teacher: `nemotron-3.5-bolt:e4b` in local Ollama (probe: 5/5 correct, ~36 tok/s).

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
Smoke-train a fresh student for 30 minutes:
```bash
uv run cerbeto train --smoke --minutes 30
```

## License
Cerbeto — its concept, specifications and code — © 2026 Michał Sobczak.

Licensed under the **[PolyForm Noncommercial License 1.0.0](LICENSE.md)**: you may read, run, modify, fork and share it for any **noncommercial** purpose (personal projects, research, education, hobby), keeping the copyright notice. **Commercial use requires the author's written permission** — please get in touch via [GitHub](https://github.com/michalasobczak).

The school library is not part of this repository; its corpora keep their own licences (BabyLM — MIT, TinyStories — CDLA-Sharing-1.0, Wikipedia — CC BY-SA 4.0, FineWeb-Edu — ODC-By 1.0), recorded in each shelf's `manifest.json`.
