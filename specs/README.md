# Specifications

One file per change, in Rondo's sections (TL;DR, Abstract, Objective, … Control Gates). Status as of 2026-11-16.

| Spec | Status |
|---|---|
| [0001](0001-library-and-tokenizer.md) library and tokenizer | delivered; G5 (human review of shelf samples) pending |
| [0002](0002-student-architecture-and-growth.md) student architecture and growth | delivered |
| [0003](0003-nursery-stage-0-pretraining.md) nursery | delivered; control student dropped (2026-11-06) |
| [0004](0004-curriculum-and-exams.md) curriculum and exams | delivered; passing rule changed 2026-11-05; G5 (human review of tasks) pending |
| [0005](0005-school-day-sleep-exam.md) school: day, sleep, exam | paused since 2026-11-06 (preschool, day 18, `v0017` 68.5%); the D student passed preschool without school |
| [0006](0006-rails-admin-panel-and-chat.md) admin panel and chat (Rails) | on hold: `cerbeto panel` and `cerbeto chat` are enough for now |
| [0007](0007-offline-distribution.md) offline distribution | someday, maybe — not planned |
| [0008](0008-growth-experiment-100m.md) growth experiment, 99M → 283M (C) | closed: no gain over `v0017` |
| [0009](0009-deep-100m-conversations.md) deep 100M student with conversations (D) | closed: success, beats `v0017` on every measure; `/stats` analysis pending |
| [0010](0010-one-billion-tokens.md) one billion tokens: new data for vocabulary, knowledge and conversations | done: stopped at 982M (2026-11-11), loss flat; public versions v1–v5 on Hugging Face |
| [0011](0011-conversation-tuning.md) conversation tuning: answer the question, keep the facts | done: v6-1032M (2026-11-12), chat 55 → 117 of 160 |
| [0012](0012-larger-vocabulary.md) a larger vocabulary, 8k → 16k | closed without a release: `v16k` and `vext` stayed behind v6 after 60M tokens; the 16k vocabulary goes on in 0014 |
| [0013](0013-honest-answers.md) honest answers: "I don't know" instead of guessing | paused (2026-11-12): facts test written; waits for 0014 |
| [0014](0014-second-billion.md) a second billion tokens | stopped on the M6 at 459M (2026-11-15); continued as `bil2-avg` (0015): level with v6 in bits per byte, 72 vs 62 of 100 in the quick benchmark |
| [0015](0015-diloco.md) DiLoCo: one model trained by several GPUs that rarely talk | built and tested (2026-11-14); runs 0016 |
| [0016](0016-ten-billion.md) ten billion tokens: a new data set on other commodity hardware | done (2026-11-19 22:33): 10.0B tokens, `bil10/final` = `bil10-12488M`, benchmark 64 |
| [0017](0017-talk10.md) conversation tuning of the ten-billion student → v7 | `talk10` (3e-5): benchmark 68 but shelves +0.07–0.15, failed G1; `talk10b` (1e-5) running |

Source document of the MVP: [source-mvp-specification.md](source-mvp-specification.md).
