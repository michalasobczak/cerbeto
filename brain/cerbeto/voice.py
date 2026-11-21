"""Speech for the voice panel (spec 0019): local Whisper turns the admin's voice into text, local Piper reads the answer.

A port of Agentaro's `Voice` (`app/models/voice.rb`, its spec 017) for the English-only panel: the same programs and
weights (by default the ones Agentaro installed in `../agentaro/storage/voice`), the same arguments and the same
clean-up of what Whisper invents on silence. Both run as separate programs — no shell, fixed arguments, a timeout, a
minimal environment — and the audio lives only in a private temporary folder deleted at once. Nothing leaves the host.
"""
import io
import os
import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from cerbeto import paths

MAX_AUDIO = 2 * 1024 * 1024     # bytes: about 60 s of 16 kHz mono WAV
MAX_TEXT = 600                  # characters per speech request
ENV_KEPT = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "TMPDIR")
# Piper runs on ONNX Runtime, whose recent builds (seen in 1.32 on macOS) start Microsoft's 1DS telemetry client
# (mobile.events.data.microsoft.com) unless this is set. Nothing may leave the admin's computer (spec 0019, N2).
ENV_SET = {"ORT_DISABLE_TELEMETRY": "1"}

NOISE = re.compile(r"^[\[(][^\])]*[\])]$")         # Whisper's notes for silence or noise, e.g. [BLANK_AUDIO]
PHANTOMS = re.compile("|".join([                    # sentences Whisper invents on silence (learnt from subtitles)
    r"thank(s| you) for watching", r"please subscribe", r"subtitles? (by|made by)[^.!?]*", r"all rights reserved",
    r"copyright", r"amara\.org", r"see you in the next (video|episode)", r"^[♪♫\s.…-]*$"]), re.I)
WHOLE_PHANTOMS = re.compile(r"^(the end|thank you|thanks|you)[.!…]*$", re.I)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿\U0001F3FB-\U0001F3FF\U0001F1E6-\U0001F1FF‍︎️⃣]")
SMILEYS = re.compile(r"(?<![\w])[:;=8xX][-'^o]?[)(\]\[DPpOo3|/\\*]+(?![\w])|(?<!\S)<3(?!\S)|(?<!\S)\^[_.]?\^(?!\S)")
MARKDOWN = re.compile(r"\*\*|__|`+|^\s{0,3}#{1,6}\s+|^\s{0,3}>\s?|^\s*[-*+]\s+|(?<=\s)[*_](?=\S)|^[*_](?=\S)"
                      r"|(?<=\S)[*_](?=\s|$|[.,!?;:])", re.M)
HINT = "A short conversation in English with Cerbeto, a small language model, about everyday facts and questions."


class VoiceError(Exception):
    """A reason the admin can read: not installed, too long, failed, timed out."""


def _voice_dir() -> Path:
    return Path(os.environ.get("CERBETO_VOICE_DIR", paths.REPO.parent / "agentaro" / "storage" / "voice"))


def _setting(name: str, agentaro: str, default) -> str:
    """CERBETO_<name> first, then Agentaro's own variable (so one install serves both), then the default."""
    return os.environ.get(f"CERBETO_{name}") or os.environ.get(agentaro) or str(default)


@dataclass
class VoiceConfig:
    whisper_bin: str
    whisper_model: str
    piper_bin: str
    piper_voice: str
    speed: float = 1.0

    @classmethod
    def from_env(cls) -> "VoiceConfig":
        d = _voice_dir()
        return cls(whisper_bin=_setting("WHISPER_BIN", "AGENTARO_WHISPER_BIN", "/opt/homebrew/bin/whisper-cli"),
                   whisper_model=_setting("WHISPER_MODEL", "AGENTARO_WHISPER_MODEL", d / "ggml-large-v3-turbo-q5_0.bin"),
                   piper_bin=_setting("PIPER_BIN", "AGENTARO_PIPER_BIN", d / "piper-venv" / "bin" / "piper"),
                   piper_voice=_setting("PIPER_VOICE", "AGENTARO_PIPER_VOICE_EN", d / "en_US-lessac-medium.onnx"),
                   speed=float(os.environ.get("CERBETO_SPEECH_SPEED", "1.0")))


def _run(program: str, args: list[str], cwd: str, timeout: float, stdin: str | None = None) -> subprocess.CompletedProcess:
    env = {**{k: os.environ[k] for k in ENV_KEPT if k in os.environ}, **ENV_SET}
    try:
        return subprocess.run([program, *args], input=stdin.encode() if stdin is not None else None, cwd=cwd, env=env,
                              capture_output=True, timeout=timeout, start_new_session=True)
    except subprocess.TimeoutExpired:
        raise VoiceError("did not finish in time") from None
    except OSError as e:
        raise VoiceError(f"could not start: {e.strerror}") from None


class Voice:
    def __init__(self, config: VoiceConfig | None = None):
        self.config = config or VoiceConfig.from_env()
        self.cpu_only = os.environ.get("CERBETO_WHISPER_NO_GPU") == "1"     # Whisper without Metal/CUDA (-ng)

    def listening(self) -> bool:
        c = self.config
        return os.access(c.whisper_bin, os.X_OK) and Path(c.whisper_model).is_file()

    def speaking(self) -> bool:
        c = self.config
        return os.access(c.piper_bin, os.X_OK) and Path(c.piper_voice).is_file()

    def status(self) -> dict:
        return {"listening": self.listening(), "speaking": self.speaking()}

    def transcribe(self, audio: bytes) -> str:
        """WAV bytes in, English text out ("" for silence or noise)."""
        if not self.listening():
            raise VoiceError("Speech recognition is not installed on this computer.")
        if len(audio) > MAX_AUDIO:
            raise VoiceError("The recording is too long.")
        if audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
            raise VoiceError("The recording is not a WAV file.")
        with tempfile.TemporaryDirectory(prefix="cerbeto-voice") as d:
            path = Path(d) / "speech.wav"
            path.write_bytes(audio)
            args = ["-m", self.config.whisper_model, "-f", str(path), "-l", "en", "-nt", "-np", "-sns",
                    "--prompt", HINT, *speed_args(audio)]
            r = _run(self.config.whisper_bin, args + (["-ng"] if self.cpu_only else []), d, timeout=60)
            if r.returncode != 0 and not self.cpu_only:
                # whisper.cpp aborts (GGML_ASSERT(buffer), SIGABRT) instead of falling back when the GPU (Metal)
                # cannot give it memory — seen in step 4 under a sandbox and possible on a busy machine. The CPU
                # gives the same text; once it was needed, it is kept for the rest of the session.
                r = _run(self.config.whisper_bin, args + ["-ng"], d, timeout=60)
                if r.returncode == 0:
                    self.cpu_only = True
            if r.returncode != 0:
                raise VoiceError("Speech recognition failed.")
        return clean_transcript(r.stdout.decode("utf-8", "replace"))

    def speak(self, text: str) -> bytes:
        """Text in, WAV bytes out, in the English Piper voice."""
        if not self.speaking():
            raise VoiceError("The voice is not installed on this computer.")
        text = readable(text)
        if not text:
            raise VoiceError("Nothing to say.")
        if len(text) > MAX_TEXT:
            raise VoiceError(f"At most {MAX_TEXT} characters at a time.")
        # Pauses shrink less than sounds, so the scale goes a bit further than 1/speed (measured in Agentaro).
        length = f"{min(2.0, max(1.0, self.config.speed)) ** -1.35:.2f}"
        with tempfile.TemporaryDirectory(prefix="cerbeto-voice") as d:
            path = Path(d) / "speech.wav"
            r = _run(self.config.piper_bin, ["-m", self.config.piper_voice, "-f", str(path), "--length-scale", length],
                     d, timeout=30, stdin=text)
            if r.returncode != 0 or not path.is_file():
                raise VoiceError("The voice failed.")
            return path.read_bytes()


def wav_seconds(audio: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(audio)) as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, EOFError, ZeroDivisionError):
        return None


def speed_args(audio: bytes) -> list[str]:
    """Whisper's encoder always works on a 30 s window (1500 frames); a question lasts a few seconds. An audio context
    sized to the clip (50 frames/s + 2 s margin, at least 10 s) cut one CPU transcription from ~7 s to ~2.5 s with the
    same text (step 4 of spec 0019). More threads: ~20% more. CERBETO_WHISPER_FULL_CONTEXT=1 turns the context off."""
    args = ["-t", str(max(1, min(8, os.cpu_count() or 4)))]
    seconds = wav_seconds(audio)
    if seconds is not None and os.environ.get("CERBETO_WHISPER_FULL_CONTEXT") != "1":
        frames = int(-(-(max(10.0, seconds + 2.0) * 50) // 64) * 64)    # rounded up to a multiple of 64
        if frames < 1500:
            args += ["-ac", str(frames)]
    return args


def _plain(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def clean_transcript(stdout: str) -> str:
    """Whisper's output without its noise notes, invented subtitle sentences, or a bare echo of the hint."""
    text = " ".join(line.strip() for line in stdout.splitlines() if line.strip() and not NOISE.match(line.strip()))
    if len(_plain(text)) > 20 and _plain(text) in _plain(HINT):
        return ""
    return without_phantoms(text)


def without_phantoms(text: str) -> str:
    if WHOLE_PHANTOMS.match(text.strip()):
        return ""
    kept = [s for s in re.split(r"(?<=[.!?…])\s+", text)
            if not (PHANTOMS.search(s.strip()) and len(re.sub(r"[\W_]", "", PHANTOMS.sub("", s))) < 12)]
    # Whisper sometimes writes a short question twice ("And the opposite of left? And the opposite of left?", step 4);
    # the student would get both. A sentence equal to the one before it is dropped.
    kept = [s for i, s in enumerate(kept) if i == 0 or _plain(s) != _plain(kept[i - 1]) or not _plain(s)]
    return " ".join(kept).strip()


def readable(text: str) -> str:
    """What Piper may read: no control characters, emoji, smileys or Markdown marks."""
    text = MARKDOWN.sub("", str(text))
    text = SMILEYS.sub("", EMOJI.sub("", text))
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return re.sub(r" {2,}", " ", text).strip()
