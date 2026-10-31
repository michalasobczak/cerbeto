# Engineering Manifesto — Cerbeto

Implementation rules binding on every specification (`specs/`). A specification may depart from them only explicitly, in the "Additional Information → Technical debt" section.

## 1. The student is ours from scratch
- No pre-trained weights and no pre-built tokenizers as a starting point. The student's weights are initialized randomly with a recorded `seed`.
- Ready-made **data** (corpora, benchmarks) are allowed if they have a known licence recorded in the shelf manifest.
- Ready-made **models** may be used only as teachers and assistant examiners — never as part of the student.

## 2. Biology at the system level (level B)
- A single neuron is simplified (standard network layers); we reproduce biology in the mechanisms of development and learning: growth, pruning, sleep/consolidation, hippocampus + cortex, synaptic consolidation, neuron renewal.
- Every "biological" mechanism must have a counterpart described in the ML literature (source cited in the specification) and its own metric in the admin panel.
- We do not reinvent the wheel: first we look for a known technique, and only then design our own.

## 3. Truth comes from code
- The correctness of the student's answers is decided by a program (the generator knows the answer), not by an LLM.
- The LLM teacher explains, asks and corrects — its content enters the data only after verifying agreement with the truth from the generator.
- Exam sets are generated from reserved seeds (`seed`) and never reach the teacher or the training data (checked by hash).

## 4. Every weight change is a version
- Every sleep, growth, pruning and rollback creates a new, immutable version of the student (weights + architecture + optimizer state + event log).
- A new version becomes active only after passing the exam gates; otherwise there is an automatic return to the previous one.
- Everything is reproducible: configuration, `seed`, data hash and code version (commit) are recorded with every version.

## 5. Local and offline
- The whole system works without internet after a one-time download of the library and the teacher.
- Budgets: training and inference of the student must fit into 16 GB RAM together with the system; hard student limit of 30M parameters.
- Data (`data/`), runs (`runs/`) and checkpoints (`checkpoints/`) never go into git.

## 6. Technology split
- **Brain** (student, training, school, exams): Python 3.12 + PyTorch, `uv` environment. Device: MPS → CUDA → CPU (detected automatically).
- **Admin panel and chat**: Ruby on Rails 8 (Hotwire), communicating with the brain exclusively via the brain's HTTP API.
- **Teachers**: any endpoint compatible with OpenAI Chat Completions; local Ollama by default.
- Weights in `safetensors` format (portable; enables a later MLX backend for inference).
- Compute precision: fp32 or mixed fp16 (autocast). **No bf16** — Apple M1 has no hardware bf16 (measured on M1 Pro: bf16 2.1 TFLOPS vs fp16 4.2 TFLOPS).
- The Neural Engine (Core ML) is not used in the MVP — training only on GPU (MPS) / CPU.

## 7. Code
- Code, identifiers, comments and all project documentation (`README`, `specs/`, `CHANGELOG`, commit messages) are written in English.
- The simplest solution that satisfies the specification; abstraction only at the second use.
- Configuration in YAML files in the repo (curriculum, architecture, daily rhythm), secrets in `.env` outside git.
- Tests: `pytest` (brain), `rails test` (admin panel). A gate of type "Unit/Integration" in a specification = an automated test in the repo.

## 8. Process (CDD)
- Every change starts with a specification in `specs/` (9 sections) and ends with an entry in `CHANGELOG.md` and an update of `README.md`.
- Tests are the proof of what the CHANGELOG describes.
- Specifications are numbered `NNNN-slug.md` and are never deleted; a change of direction = a new specification referring to the old one.
