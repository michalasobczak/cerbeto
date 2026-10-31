"""Readers that turn raw library files into documents (plain text strings)."""
import re
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

_SPEAKER = re.compile(r"^\*?([A-Z0-9]{1,4}):\t")
HUMAN, STUDENT = "<|human|>", "<|student|>"
# Who is the "student" in each conversational corpus: the child in CHILDES, speaker B in Switchboard.
_STUDENT_SPEAKER = {"childes": "CHI", "switchboard": "B"}
_WIKI_TITLE = re.compile(r"^= = = .* = = =$")


def _chunks(lines: Iterator[str], size: int) -> Iterator[str]:
    buf: list[str] = []
    for line in lines:
        buf.append(line)
        if len(buf) == size:
            yield "\n".join(buf)
            buf = []
    if buf:
        yield "\n".join(buf)


def _clean_speech(path: Path, roles: str | None = None) -> Iterator[str]:
    """Utterances, one per line. With `roles`, consecutive lines of one speaker are merged into a turn
    prefixed by `<|student|>` (the child / speaker B) or `<|human|>` (everyone else)."""
    student = _STUDENT_SPEAKER.get(roles or "")
    turn_role, turn = None, []
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = _SPEAKER.match(raw.strip())
            line = _SPEAKER.sub("", raw.strip()).lstrip("- ").strip()
            # drop transcription annotations ([laughs], ][CHI turns]), file headers and metadata lines
            if not line or line.startswith(("[", "]", "%", "@", "= = =")):
                continue
            if student is None:
                yield line
                continue
            role = (STUDENT if m.group(1) == student else HUMAN) if m else turn_role or HUMAN
            if role != turn_role and turn:
                yield f"{turn_role} {' '.join(turn)}"
                turn = []
            turn_role = role
            turn.append(line)
    if turn:
        yield f"{turn_role} {' '.join(turn)}"


def _plain_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            if line := raw.strip():
                yield line


def speech(path: Path, chunk_lines: int, roles: str | None = None, **_) -> Iterator[str]:
    yield from _chunks(_clean_speech(path, roles), chunk_lines)


def lines(path: Path, chunk_lines: int, **_) -> Iterator[str]:
    yield from _chunks(_plain_lines(path), chunk_lines)


def tinystories(path: Path, **_) -> Iterator[str]:
    buf: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            if raw.strip() == "<|endoftext|>":
                if story := "\n".join(buf).strip():
                    yield story
                buf = []
            else:
                buf.append(raw.rstrip("\n"))
    if story := "\n".join(buf).strip():
        yield story


def babylm_wiki(path: Path, **_) -> Iterator[str]:
    buf: list[str] = []
    for line in _plain_lines(path):
        if _WIKI_TITLE.match(line):
            if buf:
                yield "\n".join(buf)
            buf = []
        else:
            buf.append(line)
    if buf:
        yield "\n".join(buf)


def _parquet_texts(path: Path, columns: list[str], batch_rows: int = 2048) -> Iterator[dict]:
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        yield from batch.to_pylist()


def wiki_parquet(path: Path, **_) -> Iterator[str]:
    for row in _parquet_texts(path, ["text"]):
        yield row["text"]


def fineweb_parquet(path: Path, min_edu_score: int = 0, **_) -> Iterator[str]:
    for row in _parquet_texts(path, ["text", "int_score"]):
        if (row["int_score"] or 0) >= min_edu_score:
            yield row["text"]


READERS = {
    "speech": speech,
    "lines": lines,
    "tinystories": tinystories,
    "babylm_wiki": babylm_wiki,
    "wiki_parquet": wiki_parquet,
    "fineweb_parquet": fineweb_parquet,
}
