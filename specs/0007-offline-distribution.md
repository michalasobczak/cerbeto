# 0007 — Distribution package: further development of the student at people's homes (offline)

Status: **someday, maybe (2026-11-06, the user's decision)** — not planned; it may never happen · Depends on: 0003, 0005, 0006 · Unblocks: —

## 1. TL;DR
A package with the "post-nursery" student, the curriculum, generators, replay portions, the learning tool and the admin panel, which on a computer with 16 GB RAM lets the student continue developing (lessons, sleep, growth) completely offline with a local teacher. Proof: installation on a clean 16 GB computer and 3 learning cycles without internet.

## 2. Abstract
Stage 0 is done once, centrally. Further development — including growing the network — is to take place at people's homes, without API costs. It was agreed that copies are independent students ("separate children"); the lesson journal format prepares the ground for future sharing of experiences (not weights).

## 3. Objective
**Goals**
- Package: weights + architecture + growth log, `curriculum.yaml`, generators, exam pools, replay shelves (tokens, ~a few hundred MB), licence manifest, brain (Python), admin panel (Rails).
- Installer: `uv` (Python 3.12 + PyTorch), Ollama + teacher `nemotron-3.5-bolt:e4b` (7.2 GB — the same as the author's), Ruby/Rails for the admin panel.
- 16 GB profile: sequential rhythm (teacher and sleep alternating), sleep at night.
- Export of the lesson journal in a portable format (JSONL, without weights).

**Out of scope:** merging weights between copies, federated learning, online updates.

**Success criteria:** clean 16 GB Mac: installation ≤ 30 min (excluding download time), 3 offline cycles, peak RAM ≤ 14 GB.

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `scripts/package.sh`, `scripts/install.sh` | system | to create |
| `dist/adaptivemind-zero-<version>.tar.zst` | data | to create |
| `LICENSES/` (shelf licence manifest) | data | to create |
| Profile `config/profiles/16gb.yaml` | data | to create |

## 5. Functional Domain Changes
**Rules:** the starting version from the package is read-only (one can always return to it); the 30M student limit is enforced; no network calls whatsoever outside `localhost` after installation.

**Resilience:** no Ollama → school in "sleep and exams only" mode; no Ruby → CLI.

## 6. Critical Path
1. 16 GB profile and RAM measurement → 2. packaging script → 3. installer → 4. test on a clean offline machine.

**Completion condition:** G3 passed.

## 7. Control Gates
| # | Type | Condition | Verification | Blocks |
|---|---|---|---|---|
| G1 | Integration | The package contains all artefacts from the manifest and the licences | `scripts/package.sh --verify` | G3 |
| G2 | Integration | Peak RAM in the 16 GB profile ≤ 14 GB (teacher + student + admin panel) | measurement | G3 |
| G3 | E2E | Clean 16 GB computer, network disconnected after installation, 3 cycles with a version promotion | manual test with a checklist | release |
| N1 | Negative | No network traffic outside localhost during the cycles | network monitor (`nettop`/`lsof`) | release |

## 8. Expected Outcome
- **Direct:** anyone can develop their own student offline.
- **Measurable:** installation time, RAM, exam results after cycles at the user's.
- **Side effects:** ~6–8 GB on disk (teacher + package).
- **Rollback:** uninstall; the starting version is always available.

## 9. Additional Information
- **Rejected alternatives:** a shared model fed by the copies (merging the weights of differently grown networks — unsolved).
- **Technical debt:** installation requires two environments (Python and Ruby).
- **Open questions:** Windows/Linux in the MVP [TO BE DETERMINED]; 16 GB RAM budget: teacher 7.2 GB + student ≤ 3 GB + admin panel + system — to be confirmed by gate G2.
