# 0005 — School: day, sleep, exam, growth

Status: planned · Depends on: 0002, 0003, 0004 · Unblocks: 0006, 0007

## 1. TL;DR
An automatic learning loop: **day** (lessons with a large LLM teacher, experiences recorded in the hippocampus) → **sleep** (consolidation into the weights with replay and synaptic consolidation) → **exam** (gates) → version promotion / rollback / growth. Proof: after N cycles the student's result on the grade's `ood` exams rises, and forgetting stays within 2 pp.

## 2. Abstract
This is the core of the MVP priority: a measurable improvement in reasoning on new tasks after automatic learning cycles with a teacher. The teacher is a **large LLM run locally** (Ollama, OpenAI-compatible endpoint), so learning works offline and without API costs; the interface allows plugging in external providers. The day/sleep rhythm reproduces complementary learning systems (hippocampus + cortex, McClelland 1995). A human chatting with the student is an additional teacher, but lessons are led by the LLM.

## 3. Objective
**Goals**
- **Teachers** with roles: *homeroom teacher* (leads the lesson, asks questions, grades the difficulty), *tutor* (explains a mistake), *assistant examiner* (grades open answers; never sees the exam pools). Configuration: provider, endpoint, model, role, instructions, schedule, limits (tokens/day, cost/day for paid providers).
- **Teacher exam** before admission: solves the tasks of stages 1–3 (preschool, grade 1, grade 2) and generates explanations; required ≥ 98% correct answers and ≥ 95% of explanations consistent with the answer from the generator.
- **Lesson:** the generator provides a task + truth → the homeroom teacher phrases it naturally (linguistic variants) → the student answers (with `<|think|>`) → code grades → on a mistake the tutor explains and the student tries again → the experience is recorded.
- **Hippocampus:** a database of experiences (SQLite) with origin (`teacher|human`), verification state, result, surprise (student loss).
- **Sleep:** training on the day's verified experiences + replay (library + earlier grades, 50/50 by default) + EWC penalty; neuron renewal; optionally pruning.
- **Exam after sleep** + the 0004 gates → promotion or automatic rollback.
- **Growth controller:** no improvement of ≥ 1 pp on the current grade's exam for 3 cycles and a result below the pass threshold → growth operator (0002) → new version.
- **Rhythm:** automatic; settings: lessons per day, sleep start (after N lessons or at a given time), pause, "learn now", "sleep now".
- The **control student** goes through the same curriculum without growth.

**Out of scope:** fast plasticity in chat, dendrites, RL with a reward signal (project stage 2), knowledge sharing between copies.

**Success criteria (MVP priority)**
- The growing student passes stages 1–3 in school order — preschool, grade 1, grade 2 — according to the 0004 thresholds.
- The `ood_hard` + `ood_combo` result of every grade rises by ≥ 20 pp relative to the baseline measurement (0003).
- Forgetting: no earlier grade loses > 2 pp in an accepted version.
- Comparison report: growing student vs control student (result, number of cycles, time, parameters).

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `brain/cerbeto/teachers/` (OpenAI-compatible client, roles, prompts, teacher exam, limits) | system | to create |
| `brain/cerbeto/school/day.py`, `sleep.py`, `scheduler.py`, `growth_controller.py` | system | to create |
| `brain/cerbeto/memory/hippocampus.py` (SQLite) | system | to create |
| `brain/cerbeto/versions/` (version registry, promotion, rollback, comparison) | system | to create |
| `brain/cerbeto/api/` (FastAPI — brain API for the admin panel) | system | to create |
| `config/teachers.yaml`, `config/rhythm.yaml`, `config/prompts/*.md` | data | to create |
| Ollama 0.40 + **default teacher `nemotron-3.5-bolt:e4b`** (9B, Gemma 4 family, nvfp4, MLX runner, 7.2 GB) | external system | exists — admission after the teacher exam |
| Fallback teacher `qwen3.8:27b-mlx` (18 GB) | external system | exists |
| `data/brain.sqlite` (hippocampus, lessons, metrics, versions) | data | to create |

## 5. Functional Domain Changes
**Student states:** `idle → day → sleep → exam → (promoted | rolled_back) → [growth] → day…`; `paused` from any state.

**Rules:**
- An experience goes into sleep only when `verified = true`: the final answer from the teacher's explanation agrees with the truth from the generator; a human's utterance is verified by the assistant examiner + code (if it concerns a generator task).
- The correct trace (explanation → answer) goes into training; the student's wrong answer only as context in a self-correction format.
- Surprise (high student loss on a correct trace) increases the example's weight in sleep (counterpart of a dopamine signal, capped at ×3).
- The curriculum itself raises the difficulty within a grade when lesson accuracy is > 80% at a given level, and lowers it at < 40%.
- Teacher limits are enforced strictly; after exceeding them the teacher is suspended until the next day.
- The teacher and student training share the same GPU, so **by default the rhythm is sequential** (day, then sleep) on any hardware; on 32 GB parallelism is allowed only as an option.
- **Default teacher:** `nemotron-3.5-bolt:e4b`. Probe on M1 Pro (2026-10-31): 5/5 correct (word problem, order of events, relations, addition up to 100, explanation of a mistake), ~36 tok/s, ~36 tokens per explanation → ~1–2 s per lesson, several tens of thousands of lessons per day. The same model fits within the 16 GB profile.
- Fallback `qwen3.8:27b-mlx`: estimated ~10–12 tok/s on M1 Pro (memory bandwidth limit) — used if nemotron fails the teacher exam or as a second voice for consensus.

**Resilience:** no teacher (Ollama off) → day paused, sleep and exams work; interrupted sleep → no version is created, the student stays on the previous one.

**Data:** experiences ~1–3 KB each; 10k lessons/day ≈ 30 MB/day.

## 6. Critical Path
1. Teacher client + teacher exam → 2. a single lesson end-to-end (preschool, agree/disagree — truth from the knowledge base) → 3. hippocampus → 4. sleep + exam + gates + versions → 5. full automatic cycle for preschool → 6. growth controller → 7. brain API → 8. preschool → grade 1 → grade 2 for both students (a grade starts after the previous one is passed).

**Completion condition:** 3 automatic day→sleep→exam cycles without intervention, with at least one version promotion and an improvement in preschool `ood`.

## 7. Control Gates
| # | Type | Condition | Verification | Blocks |
|---|---|---|---|---|
| G1 | Unit | The verifier rejects 100% of explanations with a contradictory answer (set of 100 cases) | `pytest tests/school/test_verify.py` | sleep |
| G2 | Unit | The forgetting gate rejects a version on a drop > 2 pp | `pytest tests/school/test_gates.py` | promotion |
| G3 | Integration | An end-to-end lesson with a mock teacher records an experience | `pytest tests/school/test_day.py` | G5 |
| G4 | Contract | Brain API conforms to `specs/contracts/brain-api.yaml` (OpenAPI) | contract test | 0006 |
| G5 | E2E | 3 automatic cycles with a real teacher; ≥ 1 promotion; preschool `ood` rises | `cerbeto school run --cycles 3` | 0007 |
| G6 | E2E | Rollback: after a forced regression the previous version is active, logits identical | test | 0006 |
| N1 | Negative | The teacher never receives a task from the exam pool (prompt log vs hashes) | `cerbeto exam check-leak --prompts` | always |
| N2 | Negative | A teacher that has exceeded its limit receives no calls | test | always |

## 8. Expected Outcome
- **Direct:** the student learns on its own, cyclically, with a large local LLM.
- **Measurable:** curve of `id/ood` results per grade over time, forgetting, number of growths, comparison with the control student.
- **Side effects:** prolonged GPU load (27B teacher + student training).
- **Rollback:** any earlier version from the registry; pausing the school.

## 9. Additional Information
- **Rejected alternatives:** external API as the default teacher (costs, no offline — remains an option); LLM as the judge of correctness for verifiable tasks.
- **Technical debt:** a single lesson queue (no multiple teachers in parallel) in the first version.
- **Sources:** Complementary Learning Systems (McClelland et al., 1995); STaR (Zelikman et al., 2022); EWC; continual backprop (Dohare et al., 2024).
- **Open questions:** **parameter limit in school** — the hard limit is 30M (0002, `config/student.yaml`) and the growing student already has 29.9M after the nursery, so the growth controller has no room: every school growth would be rejected. Proposed: raise the limit to ~50–60M and, on a plateau, prune first and grow second; decide after the 0004 smoke check shows whether 30M saturates (needs a G5-style memory measurement for the 16 GB profile). Teacher: none regarding the choice of teacher — `nemotron-3.5-bolt:e4b` is the default on 32 GB and 16 GB, provided it passes the teacher exam (a 5/5 probe is not an exam).
