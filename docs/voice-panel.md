# Voice panel: talking to Cerbeto ([spec 0019](../specs/0019-voice-chat-panel.md))

One command starts the student and the panel; everything runs on this computer and listens on `127.0.0.1` only.

```
browser (http://127.0.0.1:8766)                      cerbeto serve (one process, model loaded once)
  microphone → silence detection → WAV  ──POST /voice/transcribe──▶  whisper-cli (English)  → text
  the admin's bubble                   ──POST /v1/reply──────────▶  Talker (one lock, history per conversation)
  Cerbeto's bubble, sentence marked    ◀─ answer + thought ─────
  playback in sentence order           ──POST /voice/speak───────▶  Piper (English voice)   → WAV
```

## Start
From `brain/`:
```bash
uv run cerbeto serve --student final --version v6-1032M --device cpu
```
then open **http://127.0.0.1:8766** (Chrome or Safari; allow the microphone once).

- The checkpoint is `checkpoints/final/v6-1032M/` (`architecture.json` + `model.safetensors`). A folder anywhere else,
  e.g. a Hugging Face download, works with `--checkpoint`:
  ```bash
  uv run cerbeto serve --checkpoint ~/.cache/huggingface/hub/models--michalasobczak--cerbeto/snapshots/<hash> --device cpu
  ```
  (`huggingface-cli download michalasobczak/cerbeto --revision v6-1032M` puts it there.)
- Other options: `--k 4` (passes of the recurrent core), `--temperature 0` (greedy, the default for the panel),
  `--port 8766`. There is no `--host`: the server is for this computer only.
- At start it prints whether speech recognition and the voice were found. Without them the panel still works by typing,
  and answers are shown without sound.

## Speech programs (reused from Agentaro)
The panel uses the **same programs and weights as Agentaro's voice mode** (its spec 017). By default they are looked
up where Agentaro installs them, `../agentaro/storage/voice/` next to this repository:

| What | Default | Override |
|---|---|---|
| Whisper program | `/opt/homebrew/bin/whisper-cli` (`brew install whisper-cpp`) | `CERBETO_WHISPER_BIN` or `AGENTARO_WHISPER_BIN` |
| Whisper weights | `ggml-large-v3-turbo-q5_0.bin` | `CERBETO_WHISPER_MODEL` or `AGENTARO_WHISPER_MODEL` |
| Piper program | `piper-venv/bin/piper` | `CERBETO_PIPER_BIN` or `AGENTARO_PIPER_BIN` |
| English voice | `en_US-lessac-medium.onnx` (+ `.onnx.json`) | `CERBETO_PIPER_VOICE` or `AGENTARO_PIPER_VOICE_EN` |
| Folder of all four | `../agentaro/storage/voice` | `CERBETO_VOICE_DIR` |
| Speech speed | `1.0` (1.0–2.0) | `CERBETO_SPEECH_SPEED` |
| Whisper on CPU only (`-ng`) | off; turned on by itself after a GPU failure | `CERBETO_WHISPER_NO_GPU=1` |
| Whisper's full 30 s window | off (the window follows the clip, ≥ 10 s) | `CERBETO_WHISPER_FULL_CONTEXT=1` |

When Metal cannot give `whisper-cli` memory it aborts (`GGML_ASSERT(buffer) failed`) instead of using the CPU; the
panel then repeats the transcription with `-ng` and stays on the CPU for the rest of the session (~1.6 s per question
on an M-series CPU). End-to-end check without a browser: `scripts/voice_e2e.py`, results in
[voice-panel-e2e-report.md](voice-panel-e2e-report.md).

The English voice is optional in Agentaro, so it may be missing. To add it (once, then offline):
```bash
cd ../agentaro/storage/voice
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

## The panel
- **Mic button** (or Space): listening; the turn is sent after ~1 s of silence, or with **Done**. Pressing the button
  again cancels.
- **States** (under the conversation): *Offline* (red banner with the start command), *Ready*, *Listening…*,
  *Thinking…* (writing down, then the student), *Speaking…* (the sentence being read is marked), *Problem* (the reason,
  and **Try again** that re-sends the last text).
- **Stop** (or Esc) interrupts reading. The microphone ignores sound while Cerbeto thinks or speaks, so its own voice
  does not come back as a question.
- **Typing** works always, also without a microphone or speech programs.
- **hands-free**: listen again right after the answer. **memory**: keep this conversation as context (off by default,
  as in `cerbeto chat`; the student does not yet tell turns apart well). **New**: forget the conversation.
- What the student "thinks" (`<|think|>`) is shown dimmed above the answer and never read aloud.
- Under each answer: the model's time and tokens, and for spoken questions the time from the end of speech to the first
  sound (G4). The server prints the same timings per turn (`transcribe`, `reply`, `speak`), never the words.

If the server stops, the panel shows *Offline* within 5 s; start it again and the panel returns to *Ready* by itself.

## API (for other clients, e.g. an Agentaro view)
| Method, path | Request | Response |
|---|---|---|
| `GET /v1/health` | — | `{"ready", "student", "version", "params", "device", "k", "temperature", "busy", "voice": {"listening", "speaking"}}` |
| `POST /v1/reply` | `{"conversation_id", "utterance_id", "text" (1–500 chars), "memory": false}` | `{"utterance_id", "answer", "thought", "seconds", "tokens"}` |
| `POST /v1/reset` | `{"conversation_id"}` | `{"ok": true}` |
| `POST /voice/transcribe` | `audio/wav`, ≤ 2 MiB | `{"text", "seconds"}` (`""` for silence) |
| `POST /voice/speak` | `{"text"}` (≤ 600 chars) | `audio/wav` |

Errors: `400` bad request, `409 {"error": "busy"}` while another reply is generated (no queue), `500` model error (the
server keeps running), `503` speech program missing or failed, `403` a Host or Origin other than `127.0.0.1`/`localhost`
on this port (another web page cannot use the server).
