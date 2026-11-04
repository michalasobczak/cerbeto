"""The hippocampus (spec 0005): every lesson of the day is written down at once and replayed into the weights
during sleep. Only verified experiences — whose answer agrees with the generator's truth — ever reach sleep.

SQLite, one row per experience: origin (`teacher` explanation, `code` reference solution, `human`), the prompt and
the target the student should learn, what the student actually replied, whether it was correct, and whether the
experience has already been slept on.
"""
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from cerbeto import paths

DB = paths.DATA / "brain.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
    id INTEGER PRIMARY KEY,
    created TEXT NOT NULL,
    student TEXT NOT NULL,
    day INTEGER NOT NULL,
    stage TEXT NOT NULL,
    skill TEXT NOT NULL,
    origin TEXT NOT NULL,
    task_hash TEXT NOT NULL,
    prompt TEXT NOT NULL,
    target TEXT NOT NULL,
    student_reply TEXT,
    correct INTEGER NOT NULL,
    verified INTEGER NOT NULL,
    slept INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS pending ON experiences (student, slept, verified);
"""
COLUMNS = {"asked_as": "TEXT", "teacher_reply": "TEXT"}   # added later; older databases are migrated on open


@dataclass
class Experience:
    student: str
    day: int
    stage: str
    skill: str
    origin: str
    task_hash: str
    prompt: str
    target: str
    student_reply: str
    correct: bool
    verified: bool
    id: int | None = None
    asked_as: str | None = None         # the generator's question when the homeroom teacher rephrased it
    teacher_reply: str | None = None    # the tutor's raw explanation, also when code rejected it


class Hippocampus:
    def __init__(self, path: Path = DB):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        have = {r[1] for r in self.db.execute("PRAGMA table_info(experiences)")}
        for name, kind in COLUMNS.items():
            if name not in have:
                self.db.execute(f"ALTER TABLE experiences ADD COLUMN {name} {kind}")
        self.db.commit()

    def add(self, experiences: list[Experience]) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.db.executemany(
            "INSERT INTO experiences (created, student, day, stage, skill, origin, task_hash, prompt, target, "
            "student_reply, correct, verified, asked_as, teacher_reply) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(now, e.student, e.day, e.stage, e.skill, e.origin, e.task_hash, e.prompt, e.target, e.student_reply,
              int(e.correct), int(e.verified), e.asked_as, e.teacher_reply) for e in experiences])
        self.db.commit()

    def pending(self, student: str) -> list[Experience]:
        """Verified experiences not slept on yet."""
        rows = self.db.execute(
            "SELECT student, day, stage, skill, origin, task_hash, prompt, target, student_reply, correct, verified, id "
            "FROM experiences WHERE student = ? AND slept = 0 AND verified = 1 ORDER BY id", (student,)).fetchall()
        return [Experience(*r[:9], bool(r[9]), bool(r[10]), r[11]) for r in rows]

    def memories(self, student: str, stage: str, limit: int = 20_000) -> list[tuple[str, str, bool]]:
        """(prompt, target, student was right) of experiences of `stage` already slept on — replayed in later
        sleeps, so a new night does not wash out what earlier nights taught."""
        return [(p, t, bool(c)) for p, t, c in self.db.execute(
            "SELECT prompt, target, correct FROM experiences WHERE student = ? AND stage = ? AND slept = 1 "
            "AND verified = 1 ORDER BY id DESC LIMIT ?", (student, stage, limit))]

    def mark_slept(self, ids: list[int]) -> None:
        self.db.executemany("UPDATE experiences SET slept = 1 WHERE id = ?", [(i,) for i in ids])
        self.db.commit()

    def count(self, student: str, day: int) -> int:
        return self.db.execute("SELECT COUNT(*) FROM experiences WHERE student = ? AND day = ?",
                               (student, day)).fetchone()[0]
