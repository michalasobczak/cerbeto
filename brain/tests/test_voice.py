"""Speech for the voice panel (spec 0019): local Whisper and Piper as programs, cleaned like Agentaro's `Voice`."""
import os
import stat
import struct

import pytest

from cerbeto.voice import Voice, VoiceConfig, VoiceError, clean_transcript, readable


def wav(seconds=0.1):
    n = int(16000 * seconds)
    return (b"RIFF" + struct.pack("<I", 36 + 2 * n) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
            + b"data" + struct.pack("<I", 2 * n) + b"\0" * 2 * n)


def program(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.fixture
def voice(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"x")
    (tmp_path / "en.onnx").write_bytes(b"x")
    args = tmp_path / "args"
    whisper = program(tmp_path / "whisper", f'echo "$@" > {args}\nprintf "%s" "$WHISPER_SAYS"\n')
    # piper: the text comes on stdin; writes a "WAV" to the -f path
    piper = program(tmp_path / "piper", f'echo "$@" > {args}\ncat > {tmp_path}/said\n'
                                        'while [ "$1" != "-f" ]; do shift; done; printf RIFFWAVE > "$2"\n')
    return Voice(VoiceConfig(whisper, str(tmp_path / "model.bin"), piper, str(tmp_path / "en.onnx"))), tmp_path


def test_transcribe_runs_whisper_in_english_and_cleans_the_text(voice, monkeypatch):
    v, tmp = voice
    monkeypatch.setenv("PATH", os.environ["PATH"])
    monkeypatch.setattr("cerbeto.voice.ENV_KEPT", ("PATH", "WHISPER_SAYS"))
    monkeypatch.setenv("WHISPER_SAYS", "[BLANK_AUDIO]\n What is the opposite of hot?\n")
    assert v.transcribe(wav()) == "What is the opposite of hot?"
    args = (tmp / "args").read_text().split()
    assert args[args.index("-l") + 1] == "en" and "-m" in args and "-nt" in args
    monkeypatch.setenv("WHISPER_SAYS", "Thank you for watching!")
    assert v.transcribe(wav()) == ""                       # a phantom subtitle on silence is silence


def test_transcribe_refuses_what_is_not_a_short_wav(voice):
    v, _ = voice
    with pytest.raises(VoiceError, match="not a WAV"):
        v.transcribe(b"OggS" + b"\0" * 100)
    with pytest.raises(VoiceError, match="too long"):
        v.transcribe(wav(70))


def test_speak_reads_only_readable_text_with_piper(voice):
    v, tmp = voice
    assert v.speak("**Cold.** 🙂") == b"RIFFWAVE"
    assert (tmp / "said").read_text() == "Cold."
    args = (tmp / "args").read_text().split()
    assert args[args.index("-m") + 1].endswith("en.onnx") and "--length-scale" in args
    with pytest.raises(VoiceError, match="Nothing to say"):
        v.speak(" :) ")
    with pytest.raises(VoiceError, match="At most"):
        v.speak("a" * 601)


def test_missing_programs_and_failures_are_reported(tmp_path):
    none = str(tmp_path / "none")
    v = Voice(VoiceConfig(none, none, none, none))
    assert v.status() == {"listening": False, "speaking": False}
    with pytest.raises(VoiceError, match="not installed"):
        v.speak("Hello.")
    (tmp_path / "m").write_bytes(b"x")
    broken = Voice(VoiceConfig(program(tmp_path / "w", "exit 3\n"), str(tmp_path / "m"), none, none))
    with pytest.raises(VoiceError, match="failed"):
        broken.transcribe(wav())


def test_whisper_falls_back_to_cpu_when_the_gpu_fails(tmp_path, monkeypatch):
    """Step 4: whisper-cli aborts with GGML_ASSERT(buffer) when Metal cannot allocate; the CPU (-ng) still works."""
    monkeypatch.delenv("CERBETO_WHISPER_NO_GPU", raising=False)
    (tmp_path / "m").write_bytes(b"x")
    calls = tmp_path / "calls"
    whisper = program(tmp_path / "w", f'echo "$@" >> {calls}\n'
                                      'for a in "$@"; do [ "$a" = "-ng" ] && { echo " Hello there."; exit 0; }; done\n'
                                      'exit 134\n')
    v = Voice(VoiceConfig(whisper, str(tmp_path / "m"), "none", "none"))
    assert v.transcribe(wav()) == "Hello there." and v.cpu_only
    assert v.transcribe(wav()) == "Hello there."
    lines = calls.read_text().splitlines()
    assert ["-ng" in line for line in lines] == [False, True, True]       # the GPU is tried once, not every time


def test_cleaning_rules():
    assert readable("# Yes\n- it is `hot` ;)") == "Yes it is hot"
    assert clean_transcript("(music)\n") == ""
    assert clean_transcript("Is the sky blue? Thanks for watching.") == "Is the sky blue?"
    assert clean_transcript(" And the opposite of left? And the opposite of left?") == "And the opposite of left?"
    assert clean_transcript("No. Yes. No.") == "No. Yes. No."           # only a repeat right after itself goes


def test_config_reuses_agentaro_settings(monkeypatch, tmp_path):
    for k in ("CERBETO_WHISPER_BIN", "CERBETO_PIPER_VOICE", "AGENTARO_PIPER_VOICE_EN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CERBETO_VOICE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENTARO_WHISPER_BIN", "/x/whisper-cli")
    c = VoiceConfig.from_env()
    assert c.whisper_bin == "/x/whisper-cli" and c.piper_voice == str(tmp_path / "en_US-lessac-medium.onnx")
    monkeypatch.setenv("CERBETO_WHISPER_BIN", "/y/whisper-cli")
    assert VoiceConfig.from_env().whisper_bin == "/y/whisper-cli"


def test_piper_and_whisper_run_with_onnx_runtime_telemetry_off(voice, monkeypatch):
    """N2 (step 4): ONNX Runtime under Piper starts Microsoft's 1DS telemetry unless ORT_DISABLE_TELEMETRY is set;
    the minimal environment of the programs must still carry it."""
    v, tmp = voice
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    v.config.piper_bin = program(tmp / "piper2", f'printf "%s" "$ORT_DISABLE_TELEMETRY" > {tmp}/ort\n'
                                                 'while [ "$1" != "-f" ]; do shift; done; printf RIFFWAVE > "$2"\n')
    v.speak("Cold.")
    assert (tmp / "ort").read_text() == "1"


def test_whisper_gets_an_audio_context_sized_to_the_clip(voice, monkeypatch):
    """Step 4: a 30 s encoder window for a 2 s question cost ~7 s on CPU; the context follows the clip (≥ 10 s)."""
    v, tmp = voice
    monkeypatch.delenv("CERBETO_WHISPER_FULL_CONTEXT", raising=False)

    def args(seconds):
        v.transcribe(wav(seconds))
        a = (tmp / "args").read_text().split()
        return int(a[a.index("-ac") + 1]) if "-ac" in a else None, int(a[a.index("-t") + 1])

    assert args(2) == (512, args(2)[1]) and 1 <= args(2)[1] <= 8          # 10 s · 50 frames/s, rounded up to 64
    assert args(20)[0] == 1152                                             # (20 + 2) s · 50 = 1100 → 1152
    assert args(29)[0] is None                                             # ≥ 1500 frames: Whisper's own window
    monkeypatch.setenv("CERBETO_WHISPER_FULL_CONTEXT", "1")
    assert args(2)[0] is None
