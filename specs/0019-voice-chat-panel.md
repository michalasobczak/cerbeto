# 0019 — Voice chat panel: the admin talks to Cerbeto

Status: implemented, E2E pending (2026-11-21) — the panel is served by `cerbeto serve` (decision change in §9) · Depends on: 0011 (conversation-tuned student), Agentaro spec 017 (voice mode) · Unblocks: —

## 1. TL;DR
A panel in which the admin **talks** to Cerbeto: their speech is turned into text by local Whisper, the text goes to
the student, and the answer appears as text and is read aloud at once by local Piper. Cerbeto gets one small thing it
does not have yet: **`cerbeto serve`**, a thin local HTTP server around `Talker` that loads the checkpoint once. The
rest (microphone, end-of-speech detection, playback, STT, TTS, chat bubbles) is reused from Agentaro's voice mode. All
of it runs on the admin's machine; nothing leaves `127.0.0.1`.

## 2. Abstract
Today the student is reached only through the CLI (`cerbeto chat`) and the `Talker` class; the read-only panel
(`cerbeto panel`, port 8765) has no inference endpoint and the brain API of 0006 (port 8700) is on hold. Agentaro
already has a working, local voice mode: `voice(app, chat)` in `chat.js` (Web Audio, silence detection, WAV upload,
sentence-by-sentence playback), `Voice.transcribe` (`whisper-cli`) and `Voice.speak` (Piper ONNX). Rebuilding that in
Cerbeto would duplicate tested code, so the panel is an Agentaro view, and Cerbeto exposes the student over a minimal
local HTTP contract. A reply takes 0.3–3.5 s on CPU (v6, k = 4, ≤ 60 tokens; step 1 of the epic), so a synchronous
request is enough for the first version and token streaming is left out.

## 3. Objective
**Cerbeto side — `cerbeto serve`** (`brain/cerbeto/serve.py`, standard library only, like `panel.py`):
- `uv run cerbeto serve --student final --version v6-1032M [--device cpu|mps|cuda] [--k 4] [--temperature 0]
  [--port 8766]`; binds **127.0.0.1 only**; loads the checkpoint once (`talk.load_student`) before it starts listening.
- Endpoints (JSON, UTF-8):

| Method, path | Request | Response |
|---|---|---|
| `GET /v1/health` | — | `200 {"ready": true, "student", "version", "params", "device", "k", "temperature", "busy"}` |
| `POST /v1/reply` | `{"conversation_id": str, "utterance_id": str, "text": str (1–500 chars), "memory": bool = false}` | `200 {"utterance_id", "answer", "thought", "seconds", "tokens"}` |
| `POST /v1/reset` | `{"conversation_id": str}` | `200 {"ok": true}` |

- Errors: `400` bad JSON / empty or too long text, `409 {"error": "busy"}` while another reply is generated (one model,
  one lock, no queue), `500 {"error": …}` on an inference exception (the server keeps running).
- History per `conversation_id` in memory (dict of `Talker.history` strings, swapped into the one `Talker` under the
  lock); `memory` is off by default, as in `cerbeto chat`, because the student does not yet tell turns apart.
  History is lost when the server restarts; that is accepted.
- `answer` is `Talker.parts()` as today (first line, special tokens removed): the student was tuned on one-line
  answers and the CLI and benchmarks read it the same way. `thought` is returned for display, never spoken.

**Built instead (2026-11-21): the panel in Cerbeto** — the Agentaro repository was not reachable from the Cerbeto
space, so `cerbeto serve` also serves the panel (`brain/cerbeto/voice_panel.html` at `/`) and two speech endpoints
that run **Agentaro's programs and weights** (`brain/cerbeto/voice.py`, a port of `Voice`):
`POST /voice/transcribe` (WAV ≤ 2 MiB → `{"text"}`, `whisper-cli -l en`, same arguments and phantom clean-up) and
`POST /voice/speak` (`{"text"}` ≤ 600 chars → WAV, Piper with the English voice). The microphone, silence detection,
WAV encoding and sentence-ordered playback are ported from `voice(app, chat)`. States, failures and rules below are
the same; `/v1/*` stays the contract an Agentaro view can use later. Paths: `docs/voice-panel.md`.

**Agentaro side — panel "Cerbeto"** (optional now; the code would go to the Agentaro repository):
- Route `/cerbeto` with the shared chat partial and `voice(app, chat)`; a **separate conversation** from the director's
  history (own `conversation_id`), never routed to the director.
- `POST /cerbeto/voice/transcribe` → `Voice.transcribe(audio, language: "en")` only — **not** the director's
  `/director/voice/transcribe`, which saves a message and starts the director.
- `POST /cerbeto/reply` → `CerbetoClient#reply` (Net::HTTP to `127.0.0.1:8766`, open timeout 1 s, read timeout 30 s).
- Speech: the existing `POST /director/voice/speak` flow (`Voice.speak`, ≤ 600 characters per chunk, playback in
  sentence order) with the **English** Piper voice; started as soon as the answer arrives.
- `GET /cerbeto/health` → proxies `/v1/health`, polled every 5 s while the panel is open.

**Panel states:**

| State | Shown | Mic | Enters on | Leaves to |
|---|---|---|---|---|
| `offline` | banner "Cerbeto is not running" + the command to start it | off | health fails / connection refused | `ready` when health answers |
| `ready` | mic button | off | panel opened and health ok; end of a turn | `listening` (button or hands-free) |
| `listening` | level meter, "listening…" | on | button / end of `speaking` in hands-free | `processing` at end of speech (silence) |
| `processing` | admin's bubble with the transcript, "thinking…" | off | WAV sent | `speaking` with the answer; `ready` on empty transcript |
| `speaking` | Cerbeto's bubble, the sentence being read is marked; stop button | off (no echo into STT) | answer received | `ready` / `listening` after the last sentence or stop |
| `error` | the reason (STT, model, TTS, microphone permission) and "try again" | off | any failure below | `ready` on "try again"; `offline` if health fails |

**Failure handling:**
- **Model not running / not reachable:** `offline`, the mic disabled, the text of the last turns kept; recovery is
  automatic when health answers again. A reply request refused mid-turn → `error` "Cerbeto stopped answering",
  transcript kept, then health decides `offline` or `ready`.
- **Model busy (409):** `error` "still answering" — not queued.
- **Timeout (30 s) or 500:** `error` with the reason; the admin's text stays and can be re-sent with one click.
- **Empty transcript (silence):** back to `ready` with the hint "didn't catch that", nothing sent to the model.
- **TTS fails or Piper/voice missing:** the answer is still shown; `error` "voice unavailable" only as a note on the
  bubble, the panel returns to `ready` (text is the source of truth, sound an addition).
- **Microphone denied:** `error` with the browser hint; typed input still works (the chat's text field stays).

**Out of scope:** token streaming to the panel, multi-line / Markdown answers, Polish conversation (the student and its
tokenizer are English), access from outside localhost, authentication, barge-in while speaking (stop button only), the
Rails panel of 0006, changes to training or to the student.

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `brain/cerbeto/serve.py` — HTTP server around `Talker` (+ the panel, `/voice/*`) | system | done |
| `brain/cerbeto/voice.py` — Whisper / Piper as local programs (port of Agentaro's `Voice`) | system | done |
| `brain/cerbeto/voice_panel.html` — the panel (port of Agentaro's voice mode, Cerbeto panel style) | system | done |
| `brain/cerbeto/talk.py` — `load_checkpoint` (a folder anywhere, `--checkpoint`) | system | done |
| `brain/cerbeto/cli.py` — subcommand `serve` | system | done |
| `brain/tests/test_serve.py` — contract tests on a scripted talker and a tiny student (as `test_talk.py`) | test | done |
| `brain/tests/test_voice.py` — speech programs replaced by scripts | test | done |
| `docs/voice-panel.md` — how to start, models to install, API | doc | done |
| Agentaro: `config/routes.rb`, `CerbetoController`, `app/models/cerbeto_client.rb`, view `cerbeto/show`, state labels in `chat.js` | system (other repo) | not needed for the first version (panel in Cerbeto) |
| Agentaro: `whisper-cli` + weights, Piper + an English ONNX voice, paths in `config/application.rb` | environment | to confirm on the admin's machine |
| Port 8766 on 127.0.0.1 (next to the panel's 8765; 8700 stays reserved for 0006) | system | to create |

## 5. Functional Domain Changes
**Rules:**
- The student is loaded once per server process; every reply goes through one lock (one generation at a time).
- The server never writes checkpoints, never trains and never logs the admin's text to disk (only counts and timings
  in stdout).
- The model's `thought` is displayed dimmed and never spoken; only `answer` goes to TTS.
- STT language is fixed to `en` and the TTS voice to English for this panel; the director's voice settings are untouched.
- A turn is identified by `utterance_id` from the panel; a late answer for an old `utterance_id` (after stop/reset) is
  dropped by the panel, not spoken.
- Nothing calls a network address other than `127.0.0.1`; no cloud STT/TTS fallback.

**Resilience:** the server survives a failed reply (exception → 500, lock released); the panel recovers from `offline`
without a reload; the Agentaro side works without Cerbeto running (panel in `offline`, the rest of Agentaro unaffected).

## 6. Critical Path
1. `serve.py` + `cerbeto serve` + `test_serve.py` (health, reply, reset, 400/409/500, history per conversation).
2. Manual check with `curl` against v6 on CPU: health, one reply, latency.
3. Agentaro: `CerbetoClient` + controller + route, with a stub server in tests.
4. Agentaro view: chat partial + `voice(app, chat)` wired to `/cerbeto/voice/transcribe` and `/cerbeto/reply`, the
   state labels, the `offline` banner.
5. Install / confirm Whisper weights and the English Piper voice on the admin's machine.
6. End-to-end session with the network off (epic step 4).

**Completion condition:** with the network disconnected, the admin says "What is the opposite of hot?", sees the
transcript, sees Cerbeto's answer and hears it read aloud; stopping `cerbeto serve` turns the panel `offline`, starting
it again brings the panel back to `ready` without a reload.

## 7. Control Gates
| # | Type | Condition | Verification | Blocks |
|---|---|---|---|---|
| G1 | Unit | `/v1/health`, `/v1/reply`, `/v1/reset` return the contract; 400 on empty / > 500 chars; 409 while busy; 500 on an exception and the next reply works; histories of two `conversation_id`s do not mix | `uv run pytest tests/test_serve.py` | G3 |
| G2 | Unit (Agentaro) | `CerbetoClient` maps 200/400/409/500, refused connection and timeout to the panel's errors | Agentaro test suite with a stub server | G3 |
| G3 | E2E | Completion condition: utterance → transcript → answer as text → answer read aloud, in English | manual session, recorded in Results | G5 |
| G4 | Latency | from end of speech to the first sound ≤ 5 s on the admin's machine on CPU (v6, k = 4, answers ≤ 60 tokens); the three parts (STT, model, first TTS chunk) measured separately | timings in the panel's log over 10 turns | — |
| G5 | Manual | the admin accepts the panel: states readable, stop works, the `offline` banner is clear | acceptance session | — |
| N1 | Negative | `cerbeto serve` listens only on 127.0.0.1; `--host` does not exist | `test_serve.py` | always |
| N2 | Negative | a whole turn works with the network off (no DNS, no outside connection) | E2E with Wi-Fi off / `lsof -i` showing only loopback | always |
| N3 | Negative | the director never receives a message from the Cerbeto panel | Agentaro test: no director run after `/cerbeto/voice/transcribe` | always |
| N4 | Negative | `thought` is never sent to TTS | Agentaro test on the speak call | always |

## 8. Expected Outcome
- **Direct:** a spoken conversation with the student, the fastest way for the admin to feel what it can and cannot do.
- **Measurable:** G4 timings per part; how many of 10 spoken questions are transcribed exactly and answered sensibly.
- **Side effects:** Agentaro becomes the place where Cerbeto is talked to by voice; Cerbeto gains a small loopback-only
  contract that 0006 can later reuse or replace.
- **Rollback:** remove the Agentaro route and stop `cerbeto serve`; `cerbeto chat` and `cerbeto panel` are unchanged.

## 9. Additional Information
- **Decisions:**
  - **Changed at implementation (2026-11-21): the panel is served by `cerbeto serve`, not by Agentaro.** The Agentaro
    repository was not reachable from the Cerbeto space, and the panel had to run. What is reused is Agentaro's
    *installation and logic*: the same `whisper-cli`, Whisper weights, Piper and voice files (default paths in
    `../agentaro/storage/voice`, Agentaro's environment variables honoured), the same Whisper arguments and clean-up,
    and the browser side of its voice mode. Cost: a second copy of ~150 lines of voice code. G2 and N3 (Agentaro
    tests) do not apply while there is no Agentaro view; N3 holds by construction (the panel has no route to the
    director).
  - The server replies through `Talker`'s token-by-token path (stops at `<|eos|>`): `model.generate` always runs all 60
    tokens — v6 on CPU took 3.4 s per answer instead of 0.26–0.67 s.
  - Host and Origin headers are checked (only `127.0.0.1` / `localhost` on the server's port): loopback binding alone
    does not stop another web page in the admin's browser from posting to it.
  - Panel in Agentaro, not in Cerbeto: microphone handling, silence detection, STT, TTS and chat rendering already
    exist and are local there; Cerbeto adds only the server.
  - A server, not a subprocess per turn: loading the checkpoint takes ~1.35 s and the answer itself 0.3–3.5 s, so a
    process per utterance would double the wait; a long-lived `cerbeto chat` driven over stdin was rejected as
    fragile (prompt parsing, no error codes).
  - Synchronous reply, no streaming: answers are short and the panel speaks whole sentences anyway; streaming can be
    added as `/v1/reply?stream=1` (NDJSON) when bigger students make answers slow.
  - One-line answers (`Talker.parts` unchanged): matches training and the benchmarks; multi-line output is mostly
    run-on text today.
  - Separate port 8766, standard library only: no new dependency in `brain/`, the same pattern as `panel.py`.
  - English only: the student's tokenizer and data are English; Polish needs a different student, not a panel change.
- **Default checkpoint:** `final/v6-1032M` (published, conversation-tuned, the one measured in step 1); `bil10` or a
  later v7 via `--version` once its weights are on the machine.
- **Technical debt:** no authentication (loopback only); history in memory only; one admin at a time.
- **Sources:** step 1 report of this epic; `brain/cerbeto/talk.py` (`Talker`, `load_student`); `brain/cerbeto/panel.py`;
  Agentaro `app/assets/javascripts/chat.js` (`voice`), `app/models/voice.rb`, `specs/017-voice-and-streaming.md`.
- **Open questions:** Is the English Piper voice already installed on the admin's machine? Hands-free (listen again
  after speaking) or push-to-talk as default — proposed: push-to-talk, hands-free as a toggle.

## Results
- **G1 (2026-11-21):** `uv run pytest tests/test_serve.py tests/test_voice.py` — 24 passed, 1 skipped (the real-socket
  test: the sandbox of the run forbids binding a port). Health / reply / reset contract, 400 on 8 bad bodies, 409 while
  busy without queueing, 500 then the next reply works, two conversations' histories apart, reset forgets; N1: no
  `--host`, foreign Host / Origin → 403; N4: speech is built from `answer` only.
- **Model through the server (v6-1032M, CPU, 4 threads, k = 4, T = 0), 10 questions:** load 1.35 s; reply median
  **0.46 s**, 0.26–0.67 s for 7–13 tokens, 2.1 s once (42 tokens: an unclosed thought, empty answer — shown as
  "(no answer)" with the thought dimmed). "What is the opposite of hot?" → "The opposite of hot is cold." in 0.36 s.
- **Step 4 (2026-11-21), [report](../docs/voice-panel-e2e-report.md):** `scripts/voice_e2e.py` — recorded questions
  (another Piper voice, down-sampled like the panel's `wav()`) → real handler → local Whisper → v6 → local Piper, no
  port, CPU only. **G3 server side passed:** 9 of 10 transcripts exact (the 10th: "two plus three" written "2 plus
  3"), a 17 s utterance exact, silence → "", 3 turns with memory, reset forgets, 409 at once while busy.
  **G4 passed on CPU:** end of speech → first sound median **2.38 s**, max **2.59 s** (STT 1.66 s, model 0.46 s,
  first TTS 0.24 s). **N2:** no connection off loopback from the server, no external URL in the page, no sandbox
  network violation from Whisper / Piper. Fixed: Whisper's abort when Metal has no memory (CPU fallback), a repeated
  sentence in the transcript, Esc not stopping the reading after a click on the mic, an endless "typing" bubble.
  **Still manual (G3 browser half, G5):** microphone, loudspeaker, Stop / Esc, Offline ↔ Ready — steps in the report.
