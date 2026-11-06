# 0005 — School: day, sleep, exam, growth

Status: in progress (2026-11-05) — teacher exam, day, hippocampus, sleep, exam gates, versions, `cerbeto school run` and `cerbeto school grow` delivered; the school runs (preschool, second run); open: automatic growth controller, neuron renewal, self-correction format, brain API, control student · Depends on: 0002, 0003, 0004 · Unblocks: 0006, 0007

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
| Ollama 0.40 + **default teacher `gemma4:e4b-mlx`** (MLX runner, 9.5 GB) | external system | exists — admitted 2026-11-04 (teacher exam: 97.5% correct, 94.8% consistent; admitted by owner decision) |
| Fallback teacher `qwen3.8:27b-mlx` (18 GB) | external system | exists |
| `data/brain.sqlite` (hippocampus, lessons, metrics, versions) | data | to create |

## 5. Functional Domain Changes
**Student states:** `idle → day → sleep → exam → (promoted | rolled_back) → [growth] → day…`; `paused` from any state.

**Rules:**
- An experience goes into sleep only when `verified = true`: the final answer from the teacher's explanation agrees with the truth from the generator; a human's utterance is verified by the assistant examiner + code (if it concerns a generator task).
- The correct trace (explanation → answer) goes into training; the student's wrong answer only as context in a self-correction format.
- Surprise (high student loss on a correct trace) increases the example's weight in sleep (counterpart of a dopamine signal, capped at ×3).
- **Roles in a lesson:** the *homeroom teacher* acts **before** the student answers — it rephrases the generator's question in its own words (code decides which lessons, by `rephrase_share`; the rephrasing is kept only if names, numbers, options and the knowledge-base words survive and no negation appears); the *tutor* acts **after** a wrong answer — it solves and explains the task, and its explanation is used only when code verifies it, otherwise the generator's reference solution is used.
- **Sleep replays earlier days of the stage**, not only today's lessons: per step as many experiences of earlier days as of today, an earlier mistake twice as often as an earlier right answer. Without it a night washed out what earlier nights taught (2026-11-04: night 2 lost `clarify` and `categories` `ood`).
- **Early stopping on a validation set:** during sleep the student is checked on a fixed validation set of the current stage (`id`, `ood_hard`, `ood_combo` forms; disjoint from the exam pools and excluded from lessons) at the start and after every epoch; the night keeps the weights with the best validation result, not the last ones. Reason: on small models the result on new combinations rises early and then falls while the training forms keep improving (task-specific overfitting, "Not All LLM Reasoners Are Created Equal", 2024) — observed here: every night after the first raised `id` and lowered `ood`.
- Lessons never contain an exam prompt nor a validation prompt (prompt hashes excluded from the lesson stream).
- The curriculum itself raises the difficulty within a grade when lesson accuracy is > 80% at a given level, and lowers it at < 40%.
- Teacher limits are enforced strictly; after exceeding them the teacher is suspended until the next day.
- The teacher and student training share the same GPU, so **by default the rhythm is sequential** (day, then sleep) on any hardware; on 32 GB parallelism is allowed only as an option.
- **Default teacher:** `gemma4:e4b-mlx`. Teacher exam (2026-11-04, 240 tasks, 15 per skill, few-shot prompt): 97.5% correct answers, 94.8% consistent explanations — just under the 98% / 95% thresholds; **admitted by the owner's decision**, because the remaining errors are mostly ambiguous curriculum items ("A car is a toy?" — toy cars exist), one counting slip (12 vs 13) and explanations that state the order of events without the word before/after. `nemotron-3.5-bolt:e4b` (the earlier choice) was not admitted: 96.1% / 84.6% on 800 tasks with a short prompt, 84% / 79% with the few-shot prompt (it answers "no" to unanswerable questions). Both teachers think (`think`) only on counting, order and comparisons: counting long lists without thinking is ~50%, with thinking ~90%, at ~7 s instead of ~1–2 s per task.
- Fallback `qwen3.8:27b-mlx`: estimated ~10–12 tok/s on M1 Pro (memory bandwidth limit) — used as a second voice for consensus or if the default teacher regresses.

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
- **Technical debt:** a single lesson queue (no multiple teachers in parallel) in the first version; not yet built: growth controller (blocked by the 30M limit), neuron renewal, self-correction format, assistant examiner, brain API, control student.
- **First cycles (2026-11-04, growing/v0032 → school, gemma4 teacher, 1500 lessons/day):** night 1 raised preschool from ~20% to 61.9% (promoted `v0001`; lesson accuracy next day 18% → 80%); nights 2 and 3 raised the `id` forms further (opposites 32 → 79%, sizes 52 → 81%) but lost `ood` forms the lessons never show (clarify in context 100 → 22%: the student answered "Thank you!" to "There is a red ball and a blue ball. Give me the ball.") and were rolled back (58.0%, 54.6%). Part of `v0001`'s score came from habits ("Which one?" to everything, "No" on an unbalanced pool) — the exam pools were rebalanced (0004).
- **Next cheap steps (from the literature):** early stopping on validation (above); more surface diversity — rephrase most lessons, including yes/no statements, under the code checks (instruction diversity drives generalization, 2024); before grade 1, lessons that combine two skills in pairs **different** from the exam's `ood_combo` (SKILL-MIX: training on skill combinations improves held-out combinations).
- **Plateau of the second run (2026-11-05, days 9–16):** preschool stuck at 60.5% (`v0007`/`v0009`); in most nights validation was best at step 0, so early stopping kept the weights from before sleep and the "promoted" versions `v0010`–`v0013`, `v0015` were copies with the same score. Lessons were 97% right (the teacher explained 35 of 1500), so a day had almost nothing to teach. The missing points are mostly `ood_hard` items that lessons never give by design (rare opposites push/pull, win/lose; rare sounds owl, wolf; rare sizes ant, whale, bear: the student answers "The opposite of push is closed") and held-out forms (clarify with two similar things in context: 10%, opposite of the opposite: 18%). More passes of the core do not help: validation of `v0015` at K = 2/4/6/8 is 75.1/76.5/75.3/73.9%.
- **Harder lessons (after the plateau):** (1) a skill comes up in a day with weight 1 + `focus` × (1 − its score on the active version's last exam; the nursery exam on day 1); (2) four **lesson-only mixes** of two preschool skills at `mix_weight` each (`mix.size_opposite`, `mix.opposite_statement`, `mix.sound_size`, `mix.clarify_kind`, about 20% of a day) — never an exam form, never a rare item, never in exams or validation, so `ood_hard` and `ood_combo` still measure transfer to what lessons did not show; (3) **a night whose best moment is its start creates no version**: no exam, no promotion, a cycle without progress (`night changed nothing` in the log and in `cycles.jsonl`).
- **Reading and neighbouring forms (same day):** the rare items are now *told* in lessons — `read.facts` gives the fact in a short text and asks about it ("Push and pull are opposites. What is the opposite of push?"); lessons never ask about a rare item without its text, so `ood_hard` asks from memory what the student only read. Three more lesson-only mixes practise a form next to the one the exam holds out: `mix.say_back` (reported speech; the exam describes a situation), `mix.not_statement` ("is not" about colours; the exam negates abilities), `mix.which_category` (a choice "Which one is a fruit, a lemon or a car?"; the exam asks colours this way).
- **Passing rule fitted to moving between grades (0004):** every code-graded skill `id` ≥ 85% (mastery of what was taught), the stage's **mean** `ood_hard` ≥ 70% and every skill `ood_hard` ≥ 50% (transfer on the whole, no skill without it). Before: `ood_hard` ≥ 70% in every skill. `v0015` passes neither.
- **The "No" reflex (found 2026-11-05):** `id` of agree/disagree (67%) and categories (64%) was lost on yes/no, not on knowledge: "A dog says woof. Right?" → thinking "A dog says woof. So the answer is no." → "No. A dog says woof." — the thinking stated the fact but never compared it with the claim, and categories had no thinking at all. Every preschool yes/no lesson now thinks claim → fact → decision: "The question says: a shirt is a vehicle. A shirt is a piece of clothing. So that is wrong, and the answer is no." Lesson weights of a grown (not yet examined) version come from the last promoted version's exam, not from the nursery exam.
- **Night after day 17 (first with the new lessons):** lesson loss fell, validation fell 76.5 → 72–74%, the night was kept out (no version). Hypotheses: lesson prompts of small task spaces are memorised (lessons 96–100% right, the same forms on held-out exam items 64–79%), and the yes/no thinking was inconsistent — 5,874 replayed memories in the old format and the teacher's own explanations next to the new comparison. Changes (diagnostic first): validation is logged **per skill** at every check of the night; yes/no lessons always learn the code's comparison (the tutor is not asked on yes/no); yes/no memories without the comparison are **retired** from replay (kept in the database, column `retired`).
- **Growth (2026-11-05, the user's decision):** the parameter limit was raised 30M → 40M and `v0015` was grown into `v0016` with `cerbeto school grow --mlp 384`: 1408 → 1792 MLP neurons in every block, 29.9M → 34.6M parameters, function-preserving (Δ loss 0.0; growth log `runs/school/growth_log.jsonl`). The evidence above did not point to capacity, so growth and harder lessons start together: if the stage moves, the two causes are not separated — a control run (v0015 with the same lessons) would separate them.
- **Night bench (2026-11-05/06, `cerbeto school bench`, `config/bench.yaml`, results `runs/bench/`):** the night after day 17 replayed offline from a copy of `v0016`, one setting changed per variant, validation per skill at every check (≈10 min per night on a free machine). Start 76.5% in every variant.

  | Variant (hypothesis) | Best (step) | End |
  |---|---|---|
  | base | 76.5 (0) | 76.5 |
  | seed2 — chance alone (H7) | 79.7 (70) | 74.6 |
  | inconsistent — mixed yes/no thinking (H1) | 78.6 (105) | 73.7 |
  | dedup — each memory prompt once, 1,739 instead of 18,770 (H2) | 76.9 (126) | 76.9 |
  | lr_low — 5e-5 instead of 2e-4 (H3) | 76.5 (0) | 72.5 |
  | ewc_off — no nursery consolidation (H4) | 76.5 (0) | 76.5 (identical to base) |
  | memory_low — replay share 0.3 (H5) | 76.5 (0) | 72.7 |
  | v0015 — before growth (H6) | 76.5 (0) | 76.2 |
  | k4 — core passes fixed at 4 (H8) | 76.9 (126) | 76.9 |
  | skills EWC λ 1e-3 / 1e-2 / 1e-1 (H9) | 76.5 (0) each | 76.5 / 74.6 / 73.2 |
  | dedup + k4 | 76.5 (0) | 76.2 |

  Findings: (1) **no setting of the night moves it beyond chance** — a different seed alone moves the best check by 3 pp, every hypothesis stays inside that; (2) every night trades skills in the same way, already by step 35 and whatever holds the weights: agree/disagree −13…−25, categories −10…−15, opinion −5…−12, clarify +2…+37; (3) the nursery EWC does nothing — its Fisher (squared gradient of a batch mean) sums to 1.7 over 34.6M weights, so ewc_off is identical to base to the last digit; EWC on earlier right lessons (per-lesson Fisher, mean 1) at 1e-3…1e-1 does not stop the trade either, so the trade is not drift of old weights but what the day's lessons teach (the new yes/no comparison format costs agree/disagree first — the variant with the old mixed formats lost least, −8); (4) a smaller learning rate learns less (lesson loss 0.21 vs 0.12), less replay is worse; (5) the grown `v0016` and `v0015` behave the same: the new neurons are not used yet; (6) **the validation set (8 skills × 3 variants × ≤20 tasks) is too small to choose the weights of a night**: in 9 of 13 variants the night kept its start and its learning never reached the exam.
  Changes (2026-11-06): the night keeps its **end** when the best check is within `keep_tolerance` = 2 pp of it — the exam (thousands of frozen tasks) decides, the promotion gate still refuses any drop; earlier days are replayed **once per prompt** (`dedup_memories`); skills EWC is in the code but off (`ewc_skills_lambda: 0`). First night under these rules (day 18): validation 76.5 → 75.1% at the end (kept), **exam 60.5 → 68.5% — promoted `v0017`**, the first progress since day 7. `ood_hard` rose where reading and the mixes taught: opposites 25 → 85%, clarify 10 → 52%, sizes 54 → 69%, routines 62 → 77%; agree/disagree fell 67 → 57% (`id` and `ood_hard`), categories `ood_combo` 64 → 53% — the yes/no format change. The old rule would have kept the start and examined nothing. Open: whether the next nights get agree/disagree through the format change; a larger validation set; a nursery Fisher per example.
- **Sources:** Complementary Learning Systems (McClelland et al., 1995); STaR (Zelikman et al., 2022); EWC; continual backprop (Dohare et al., 2024); Ibrahim et al., Simple and Scalable Strategies to Continually Pre-train LLMs (2024, replay + re-warming); Instruction Diversity Drives Generalization to Unseen Tasks (2024); Not All LLM Reasoners Are Created Equal (2024); SKILL-MIX (2023); FLAN (Wei et al., 2021: very small models lose on many tasks at once — capacity will matter). Related systems (CurLL, LittleLearner, BabyLM interaction track, Language Models Need Sleep, Wake-Sleep Consolidated Learning, From Growing to Looping, Huginn): see README "Related work".
- **Open questions:** the parameter limit is 40M (raised from 30M, see above); further growth needs a G5-style memory measurement for the 16 GB profile. Teacher: whether `gemma4:e4b-mlx` (9.5 GB) fits the 16 GB profile next to student training — to measure.
