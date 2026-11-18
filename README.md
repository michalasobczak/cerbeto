# Cerbeto

*Cerbeto* (Esperanto: *cerbo* "brain" + *-et-* diminutive) — "little brain".

A small English language model trained from random weights: 106.8M parameters, raised like a child (nursery → preschool) and then fed
more and more text. Try it in the browser: **[chat on Hugging Face](https://huggingface.co/spaces/michalasobczak/cerbeto-chat)** (pick a
model above the chat) · weights: **[michalasobczak/cerbeto](https://huggingface.co/michalasobczak/cerbeto)**.

## Checkpoints (2026-11-16)

| Checkpoint | Date | Tokens: run / in all | What it is | In the chat | Results |
|---|---|---|---|---|---|
| `growing/v0032` | 11-01 | 300M / 300M | nursery, 6.9M → 29.9M parameters | – | BLiMP 65.9% |
| `school/v0017` | 11-06 | 11M / 311M | preschool at school with an LLM teacher, 34.6M | – | preschool exam 68.5%, not passed |
| `final/mid` (D) | 11-07 | 69M / 391M | deep, thin 106.8M student, conversations of several turns | – | preschool exam 76.3%, passed |
| `v1-439M` … `v5-704M` | 11-09 → 11-10 | 439–704M / 0.83–1.10B | first billion, checkpoints on the way (HF tags) | – | chat 69 (v4), 55 (v5) of 160 |
| `final/bil` | 11-11 | 982M / 1.37B | end of the first billion | – | knows facts as sentences, not as answers |
| **`v6-1032M`** (`final/talk2`) | 11-12 | 50M / 1.42B | conversation tuning of `bil` — **the published version** | ✅ | chat 117 of 160 · bench **62**/100 |
| `final/v16k`, `final/vext` | 11-12 | 60M / 1.48B | v6 on a 16k / 12k vocabulary | – | behind v6, dropped |
| `final/bil2` | 11-15 | 460M / 1.94B | second billion on the Mac mini, stopped | – | below `bil2-avg` at equal tokens |
| **`final/bil2-avg`** | 11-14 | 1.0B / 2.49B | second billion, trained by DiLoCo on three GPUs | – | level with v6 in bits per byte · bench **73**/100 |
| `final/bil10` | running, ends ~11-19 | → 10B / → 12.5B | ten billion tokens of a new data set, from `bil2-avg` | newest preview only | – |
| `bil10-5014M` | 11-17 | 2.5B / 5.0B | preview of `bil10` at 2.5B, before conversation tuning | – | bench 59/100: more school text, conversation drifted |
| `bil10-5676M` | 11-17 | 3.2B / 5.7B | preview of `bil10` at 3.2B (`checkpoints/final/bil10/preview-3191M` on the mini) | – | – |
| `bil10-6520M` | 11-17 | 4.0B / 6.5B | preview of `bil10` at 4B | – | – |
| `bil10-8326M` | 11-18 | 5.8B / 8.3B | preview of `bil10` at 5.8B | ✅ default | – |

**Tokens: run / in all** — tokens read in that run / in the model's whole life. **Chat** — 160 sampled questions, counted right by a
pattern (`scripts/hf/spot_check.py`). **Bench** — 100 questions asked once, greedy (`scripts/hf/bench.py`). Validation is compared in bits
per byte, which does not depend on the vocabulary.

**Next:** conversation tuning on the `bil10` result → v7 if it beats v6; then honest answers ([0013](specs/0013-honest-answers.md)).

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
