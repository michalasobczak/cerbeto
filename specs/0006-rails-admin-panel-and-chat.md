# 0006 — Admin panel (Rails) and chat with the student

Status: planned · Depends on: 0005 (brain API) · Unblocks: 0007

## 1. TL;DR
A local Rails 8 application for managing the school: teachers, live lesson view, progress, the student's "brain", versions with comparison and rollback, the library, and a chat with the student in which a human can correct it. Proof: Rails system tests against a mock brain API and an E2E test with the real brain.

## 2. Abstract
The source specification requires a local admin panel for configuring multiple teachers, observing dialogues, monitoring reasoning, training and resources, comparing versions and rolling back. Ruby on Rails was chosen. The brain (Python) is the single source of truth about the student, lessons and versions; the admin panel communicates with it exclusively via the HTTP API (manifesto §6), so the student also works without the admin panel.

## 3. Objective
**Views:**
| View | Contents |
|---|---|
| Teachers | list, add/edit (provider, endpoint, model, role, instructions, schedule, limits), "teacher exam" with result |
| School live | rhythm state (day/sleep/exam/pause), current lessons streamed (Turbo Streams), buttons pause / learn now / sleep now |
| Progress | exam results per grade × variant over time, forgetting, transfer, growing student vs control student |
| Brain | parameter count, history of growth/pruning/renewal, loss, RAM/GPU/CPU usage |
| Versions | list, comparison of two versions on the same tasks (answer differences), rollback |
| Library | shelves, tokens read, licences |
| Chat | chat with a selected student version, "think longer" control (K), `<|think|>` preview, "correct" button → `human` experience in the hippocampus |

**Out of scope:** access from outside localhost, multiple users, authentication.

**Success criteria:** every admin panel function from the source specification reachable in ≤ 2 clicks from the menu; "live" refresh with ≤ 2 s latency.

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `panel/` — Rails 8.1, Ruby 3.4, SQLite (admin panel preferences only), Hotwire, Propshaft | system | to create |
| `panel/app/services/brain_client.rb` | system | to create |
| `specs/contracts/brain-api.yaml` (OpenAPI, shared with 0005) | data | to create |
| Admin panel port: 3100; brain API port: 8700 (localhost) | system | to create |

## 5. Functional Domain Changes
**Rules:**
- The admin panel does not store student data; teacher configuration is saved via the brain API (`config/teachers.yaml` on the brain side).
- Rolling back a version and deleting a teacher require confirmation in the UI.
- A human's correction in chat creates an experience `origin=human, verified=pending`.
- Brain unavailable → the admin panel shows a "brain offline" state and works in read-only mode on the last cached data.

**Resilience:** API calls with a 5 s timeout; the lesson stream resumes after a disconnect.

## 6. Critical Path
1. API contract → 2. Rails skeleton + `BrainClient` → 3. Chat (the fastest proof that the student works) → 4. School live → 5. Teachers → 6. Progress and Brain → 7. Versions → 8. Library.

**Completion condition:** from the admin panel one can add a teacher, examine it, start a day, watch a lesson live, see the exam result after sleep and roll back a version.

## 7. Control Gates
| # | Type | Condition | Verification | Blocks |
|---|---|---|---|---|
| G1 | Unit | `BrainClient` maps all responses and errors of the contract | `rails test test/services` | G3 |
| G2 | Contract | Mock API generated from `brain-api.yaml`; admin panel tests pass against the mock | `rails test` | G3 |
| G3 | E2E | The scenario from the completion condition against the real brain | `rails test:system` | 0007 |
| G4 | Manual | UI review by the owner | acceptance session | 0007 |
| N1 | Negative | The admin panel listens only on 127.0.0.1 | configuration test | always |
| N2 | Negative | A rollback without confirmation does not call the API | system test | always |

## 8. Expected Outcome
- **Direct:** full operation of the school from the browser.
- **Measurable:** all 0003–0005 metrics visible on charts.
- **Side effects:** Ruby required on users' computers (see 0007).
- **Rollback:** the admin panel is optional — the brain works without it (CLI).

## 9. Additional Information
- **Rejected alternatives:** admin panel in FastAPI/HTMX (Rails chosen); a shared Rails + Python database (dual schema ownership).
- **Technical debt:** no authentication (localhost only).
- **Sources:** —
- **Open questions:** charting library (Chartkick vs Chart.js via importmap).
