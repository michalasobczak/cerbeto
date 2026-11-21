# Cerbeto

*Cerbeto* (Esperanto: *cerbo* "brain" + *-et-* diminutive) — "little brain".

A small English language model trained from random weights: 106.8M parameters, raised like a child (nursery → preschool) and then fed
more and more text. Try it in the browser: **[chat on Hugging Face](https://huggingface.co/spaces/michalasobczak/cerbeto-chat)** (pick a
model above the chat) · weights: **[michalasobczak/cerbeto](https://huggingface.co/michalasobczak/cerbeto)**.

## Checkpoints (2026-11-20)

| # | Spec | Checkpoint | Date | Tokens: run / in all | What it is | In the chat | Results |
|---|---|---|---|---|---|---|---|
| 1 | [0002](specs/0002-student-architecture-and-growth.md)–[0003](specs/0003-nursery-stage-0-pretraining.md) | `growing/v0032` | 11-01 | 300M / 300M | nursery, 6.9M → 29.9M parameters | – | BLiMP 65.9% |
| 2 | [0005](specs/0005-school-day-sleep-exam.md) | `school/v0017` | 11-06 | 11M / 311M | preschool at school with an LLM teacher, 34.6M | – | preschool exam 68.5%, not passed |
| 3 | [0009](specs/0009-deep-100m-conversations.md) | `final/mid` (D) | 11-07 | 69M / 391M | deep, thin 106.8M student, conversations of several turns | – | preschool exam 76.3%, passed |
| 4 | [0010](specs/0010-one-billion-tokens.md) | `v1-439M` … `v5-704M` | 11-09 → 11-10 | 439–704M / 0.83–1.10B | first billion, checkpoints on the way (HF tags) | – | chat 69 (v4), 55 (v5) of 160 |
| 5 | [0010](specs/0010-one-billion-tokens.md) | `final/bil` | 11-11 | 982M / 1.37B | end of the first billion | – | knows facts as sentences, not as answers |
| 6 | [0011](specs/0011-conversation-tuning.md) | **`v6-1032M`** (`final/talk2`) | 11-12 | 50M / 1.42B | conversation tuning of `bil` — **the published version** | ✅ | chat 117 of 160 · bench **62**/100 |
| 7 | [0012](specs/0012-larger-vocabulary.md) | `final/v16k`, `final/vext` | 11-12 | 60M / 1.48B | v6 on a 16k / 12k vocabulary | – | behind v6, dropped |
| 8 | [0015](specs/0015-diloco.md) | **`final/bil2-avg`** | 11-14 | 1.0B / 2.49B | second billion, trained by DiLoCo on three GPUs | – | level with v6 in bits per byte · bench **73**/100 |
| 9 | [0014](specs/0014-second-billion.md) | `final/bil2` | 11-15 | 460M / 1.94B | second billion on the Mac mini, stopped | – | below `bil2-avg` at equal tokens |
| 10 | [0016](specs/0016-ten-billion.md) | `bil10-5014M` | 11-17 | 2.5B / 5.0B | preview of `bil10` at 2.5B, before conversation tuning | – | bench 59/100: more school text, conversation drifted |
| 11 | [0016](specs/0016-ten-billion.md) | `bil10-5676M` | 11-17 | 3.2B / 5.7B | preview of `bil10` at 3.2B (`checkpoints/final/bil10/preview-3191M` on the mini) | – | – |
| 12 | [0016](specs/0016-ten-billion.md) | `bil10-6520M` | 11-17 | 4.0B / 6.5B | preview of `bil10` at 4B | – | bench 61/100 |
| 13 | [0016](specs/0016-ten-billion.md) | `bil10-8326M` | 11-18 | 5.8B / 8.3B | preview of `bil10` at 5.8B | – | bench 66/100 |
| 14 | [0016](specs/0016-ten-billion.md) | `bil10-8507M`, `bil10-10494M` | 11-18 → 11-19 | 6.0B, 8.0B / 8.5B, 10.5B | previews of `bil10` at 6B and 8B | – | – |
| 15 | [0016](specs/0016-ten-billion.md) | **`final/bil10`** (`bil10-12488M`) | 11-19 | 10.0B / 12.5B | ten billion tokens of a new data set, from `bil2-avg` | ✅ default | bench **64**/100: validation down on every new shelf (edu 3.64 → 3.35, textbooks 3.24 → 2.40), dialogue up (1.41 → 1.51) |
| 16 | [0017](specs/0017-talk10.md) | `final/talk10` | 11-20 | 50M / 12.5B | conversation tuning of `bil10` at v6's learning rate (3e-5) | – | bench **68**/100, but shelf validation +0.07–0.15: forgot, not v7 |
| 17 | [0017](specs/0017-talk10.md) | `final/talk10b` | 11-20 | 50M / 12.6B | the same at 1e-5, the rate `bil10` ended with | – | bench **66**/100, shelf validation +0.03–0.06: less forgetting, still not v7 |
| 18 | [0018](specs/0018-twenty-billion.md) | `final/bil20` | 11-20 → 11-21 | 2.16B / 14.6B | twenty billion more on `bil10`'s shelves ([0018](specs/0018-twenty-billion.md)) | – | **cancelled** at 2.16B: more text did not move the answers |

**Tokens: run / in all** — tokens read in that run / in the model's whole life. **Chat** — 160 sampled questions, counted right by a
pattern (`scripts/hf/spot_check.py`). **Bench** — 100 questions asked once, greedy (`scripts/hf/bench.py`). Validation is compared in bits
per byte, which does not depend on the vocabulary.

**What ten billion tokens showed (2026-11-20):**
- The student is not saturated: validation still fell on every new shelf to the end of the run, at ~117 tokens per
  parameter (small open models are trained on thousands per parameter).
- Reading school text does not make it answer better: the quick benchmark went 73 (`bil2-avg`) → 59 at 2.5B → 64 at
  10B. The jumps come from conversation tuning (v5 → v6: chat 55 → 117 of 160 at the same size).
- Conversation tuning must start at the learning rate the run ended with: `talk10` at 3e-5 with a fresh optimizer lifted
  validation by +0.2 within 3M tokens and only partly came back ([0017](specs/0017-talk10.md)); `talk10b` at 1e-5 was
  +0.04 at the same point and ended at +0.03–0.06 with benchmark 66; neither passed, so there is no v7 yet.
- Facts are still missing (the capital of France, the largest planet; "Which is smaller, an ant or a horse?" → "A horse"):
  partly the answering form, partly perhaps capacity.

**The 1000-question benchmark (2026-11-21):** `scripts/hf/bench1000.py`, `bench.py`'s 100 questions (core) and 900 more
from fact tables, none of them in the tuning data; between two models a difference of ~25 of 1000 is beyond chance
(with 100 questions it was ~10 of 100).

| Model | Tokens in all | **Bench1000** |
|---|---|---|
| v1-439M | 0.83B | **312** |
| v3-523M | 0.91B | **274** |
| v5-704M | 1.10B | **316** |
| v6-1032M | 1.42B | **488** |
| `bil2-avg` | 2.49B | **498** |
| `bil10` at 10.5B | 10.5B | **509** |
| `bil10/final` | 12.5B | **503** |
| `talk10` | 12.5B | **500** |
| `talk10b` | 12.6B | **494** |

- The one big step is conversation tuning (v5 → v6, +172); everything after it is within ~20 points, about the size of chance.
- Ten billion tokens of school text raised completing (71 → 79–83) and yes/no (118 → 124), but not knowledge asked as a
  question (70 → 69–81, best at `bil2-avg`): the facts read as text are not reached by the answering form.
- Arithmetic stays at 3–9 of 100 in every model.

**Next:**
1. ~~`talk10b` → v7~~: failed [0017](specs/0017-talk10.md) G1 (shelves +0.03–0.06); `bil10` stays the chat's default.
2. ~~More data at the same size~~: `bil20` ([0018](specs/0018-twenty-billion.md)) cancelled at 2.16B of 20B (2026-11-21).
   **Instead, fact questions** ([0020](specs/0020-fact-questions.md)): a teacher writes hundreds of thousands of question
   → answer pairs from the shelf passages the student has read, each checked twice; then tuning of `bil10/final` on them,
   measured on the 1000-question benchmark.
3. **More parameters only if knowledge stays flat:** a ~300M student *grown* from this one (not from scratch), the most
   that fits our training cards; the browser chat still fits (~350 MB in int8).
4. Honest answers ([0013](specs/0013-honest-answers.md)).

## More
- [CHANGELOG.md](CHANGELOG.md) — what was done and measured, by date (every phase in detail).
- [specs/](specs/README.md) — one specification per change, with status.
- [docs/background.md](docs/background.md) — the school path (teacher, day → sleep → exam), hypotheses, datasets, commands, related work.
- [scripts/hf/README.md](scripts/hf/README.md) — publishing to Hugging Face; [MANIFESTO.md](MANIFESTO.md) — implementation principles.

**Methodology:** [Rondo](https://workwithrondo.com) — CDD (Context-Driven Development): README the current state, `specs/` the intent of
every change, `CHANGELOG.md` what was delivered, tests the proof.

## Getting started
From `brain/` ([uv](https://docs.astral.sh/uv/)):
```bash
uv run pytest
```
```bash
uv run cerbeto chat
```
```bash
uv run python -m cerbeto.train.final_test train bil2
```
```bash
uv run cerbeto panel
```
Library download and shelves: `scripts/download_library.sh`, `uv run cerbeto library build`; training on several GPUs: `scripts/gpu/diloco.sh`;
the quick benchmark: `uv run --with huggingface_hub python ../scripts/hf/bench.py v6-1032M checkpoints/final/bil2-avg/final`.

## License
Cerbeto — concept, specifications and code — © 2026 Michał Sobczak.

**[PolyForm Noncommercial License 1.0.0](LICENSE.md)**: read, run, modify, fork and share for any **noncommercial** purpose (personal, research, education, hobby), keeping the copyright notice. **Commercial use requires the author's written permission** — contact via [GitHub](https://github.com/michalasobczak).

The library is not in this repository; its corpora keep their own licences (BabyLM — MIT, TinyStories — CDLA-Sharing-1.0, Wikipedia — CC BY-SA 4.0, FineWeb-Edu — ODC-By 1.0, Cosmopedia — Apache 2.0), recorded in each shelf's `manifest.json`.
