"""`cerbeto panel`: a local web panel for the school (spec 0005) — status, progress, exams, lessons, log, machine.

Read-only: it reads `runs/school/` (state, cycles, exams, log), the hippocampus (`data/brain.sqlite`, opened
read-only) and the machine, and never touches training. Standard library only; one page at http://127.0.0.1:8765.
"""
import json
import re
import sqlite3
import time
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from cerbeto import paths
from cerbeto.memory.hippocampus import DB
from cerbeto.top import machine

RUN = paths.REPO / "runs" / "school"
BASELINE = paths.REPO / "runs" / "exams" / "growing" / "v0032.json"
PAGE = Path(__file__).with_name("panel.html")
STUDENT = "school"


def _db() -> sqlite3.Connection | None:
    if not DB.exists():
        return None
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
    db.row_factory = sqlite3.Row
    return db


def _columns(db) -> set[str]:
    return {r[1] for r in db.execute("PRAGMA table_info(experiences)")}


def _log_tail(n: int = 80) -> list[str]:
    f = RUN / "log.txt"
    return f.read_text().splitlines()[-n:] if f.exists() else []


def _json_lines(f: Path) -> list[dict]:
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def _stage_skills(results: dict, stage: str) -> dict:
    return {s: {x: v[x]["accuracy"] for x in ("id", "ood_hard", "ood_combo")} | {"graded_by": v["id"]["graded_by"]}
            for s, v in results.get(stage, {}).items() if not s.startswith("_")}


def status() -> dict:
    st = json.loads((RUN / "state.json").read_text()) if (RUN / "state.json").exists() else None
    rhythm = yaml.safe_load((paths.CONFIG / "rhythm.yaml").read_text())
    passing = yaml.safe_load((paths.CONFIG / "curriculum.yaml").read_text())["passing"]
    log = _log_tail()
    out = {"now": time.strftime("%Y-%m-%d %H:%M:%S"), "state": st, "rhythm": rhythm, "passing": passing, "log": log,
           "cycles": _json_lines(RUN / "cycles.jsonl"), "machine": machine()}
    if out["machine"].get("ollama") is None:
        out["machine"]["ollama"] = []
    last = log[-1] if log else ""
    out["running"] = bool(log) and (time.time() - _log_time(last)) < 600 and "school stopped" not in last
    db = _db()
    if db and st:
        out["days"] = [dict(r) for r in db.execute(
            "SELECT day, stage, COUNT(*) lessons, SUM(correct) correct, SUM(origin='teacher') teacher, "
            "SUM(correct=0 AND origin='code') code, MIN(created) started, MAX(created) ended "
            "FROM experiences WHERE student=? GROUP BY day ORDER BY day", (STUDENT,))]
        day = [d for d in out["days"] if d["day"] == st["day"]]
        out["today"] = day[0] if day else {"lessons": 0}
        if st["phase"] == "day" and day and day[0]["lessons"] > 1:
            t0, t1 = (datetime.fromisoformat(day[0][k]).timestamp() for k in ("started", "ended"))
            rate = (t1 - t0) / (day[0]["lessons"] - 1)
            out["eta_day_s"] = rate * (rhythm["day"]["lessons"] - day[0]["lessons"])
    sleep = [ln for ln in log if "sleep · step" in ln]
    if st and st["phase"] == "sleep" and sleep:
        m = re.search(r"step (\d+)/(\d+)", sleep[-1])
        out["sleep"] = {"step": int(m[1]), "steps": int(m[2])}
    exams = sorted((RUN / "exams").glob("v*.json"))
    if exams and st:
        latest = json.loads(exams[-1].read_text())
        base = json.loads(BASELINE.read_text())["results"] if BASELINE.exists() else {}
        out["exam"] = {"version": exams[-1].stem, "scores": latest["scores"], "stages": {
            stage: {"skills": _stage_skills(latest["results"], stage), "baseline": _stage_skills(base, stage),
                    "passed": latest["results"][stage]["_passed"]} for stage in latest["results"]}}
    return out


def _log_time(line: str) -> float:
    try:
        t = datetime.strptime(time.strftime("%Y-%m-%d ") + line[:8], "%Y-%m-%d %H:%M:%S").timestamp()
        return t if t <= time.time() + 60 else t - 86400
    except ValueError:
        return 0


def skills(day: int | None) -> list[dict]:
    db = _db()
    if not db:
        return []
    where, args = "student=?", [STUDENT]
    if day:
        where, args = where + " AND day=?", args + [day]
    return [dict(r) for r in db.execute(
        f"SELECT skill, COUNT(*) lessons, SUM(correct) correct, SUM(origin='teacher') teacher, "
        f"SUM(correct=0 AND origin='code') code FROM experiences WHERE {where} GROUP BY skill ORDER BY skill", args)]


def lessons(q: dict) -> dict:
    db = _db()
    if not db:
        return {"total": 0, "rows": []}
    cols = _columns(db)
    where, args = ["student=?"], [STUDENT]
    for key in ("day", "skill", "origin", "stage"):
        if q.get(key):
            where.append(f"{key}=?")
            args.append(q[key])
    if q.get("correct") in ("0", "1"):
        where.append("correct=?")
        args.append(int(q["correct"]))
    if q.get("rephrased") == "1" and "asked_as" in cols:
        where.append("asked_as IS NOT NULL")
    if q.get("q"):
        where.append("(prompt LIKE ? OR student_reply LIKE ? OR target LIKE ?)")
        args += [f"%{q['q']}%"] * 3
    w = " AND ".join(where)
    extra = ", asked_as, teacher_reply" if "asked_as" in cols else ""
    total = db.execute(f"SELECT COUNT(*) FROM experiences WHERE {w}", args).fetchone()[0]
    limit, offset = min(int(q.get("limit") or 50), 200), int(q.get("offset") or 0)
    rows = db.execute(f"SELECT id, created, day, stage, skill, origin, correct, slept, prompt, student_reply, target"
                      f"{extra} FROM experiences WHERE {w} ORDER BY id DESC LIMIT ? OFFSET ?", args + [limit, offset])
    filters = {"skills": [r[0] for r in db.execute("SELECT DISTINCT skill FROM experiences WHERE student=? ORDER BY 1",
                                                   [STUDENT])],
               "days": [r[0] for r in db.execute("SELECT DISTINCT day FROM experiences WHERE student=? ORDER BY 1",
                                                 [STUDENT])]}
    return {"total": total, "rows": [dict(r) for r in rows], "filters": filters}


def exam(q: dict) -> dict:
    """One exam (a school version, or `baseline` = the nursery student): what it contained and every answer."""
    files = {f.stem: f for f in sorted((RUN / "exams").glob("v*.json"))}
    versions = list(files) + (["baseline"] if BASELINE.exists() else [])
    if not versions:
        return {"versions": []}
    v = q.get("version") or (list(files)[-1] if files else "baseline")
    data = json.loads((BASELINE if v == "baseline" else files[v]).read_text())
    items = data["items"]
    cycle = next((c for c in _json_lines(RUN / "cycles.jsonl") if c["candidate"] == v), None)
    summary = {}
    for it in items:
        s = summary.setdefault(it["stage"], {}).setdefault(it["skill"], {})
        cell = s.setdefault(it["variant"], [0, 0, 0])            # tasks, right, unanswerable
        cell[0] += 1
        cell[1] += it["correct"]
        cell[2] += it["answer"] == "I don't know"
    for key in ("stage", "skill", "variant"):
        if q.get(key):
            items = [i for i in items if i[key] == q[key]]
    if q.get("correct") in ("0", "1"):
        items = [i for i in items if i["correct"] == (q["correct"] == "1")]
    if q.get("q"):
        items = [i for i in items if q["q"].lower() in (i["prompt"] + " " + i["reply"]).lower()]
    limit, offset = min(int(q.get("limit") or 50), 500), int(q.get("offset") or 0)
    return {"versions": versions, "version": v, "cycle": cycle, "total": len(items), "tasks": len(data["items"]),
            "summary": summary, "rows": items[offset:offset + limit]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):        # quiet
        pass

    def _send(self, body: bytes, kind: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        try:
            if url.path == "/":
                return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            data = {"/api/status": status, "/api/lessons": lambda: lessons(q), "/api/exam": lambda: exam(q),
                    "/api/skills": lambda: skills(int(q["day"]) if q.get("day") else None)}.get(url.path)
            if data is None:
                return self._send(b"not found", "text/plain", 404)
            return self._send(json.dumps(data(), default=str).encode(), "application/json")
        except Exception as e:          # the panel must never die because a file is half-written
            return self._send(json.dumps({"error": repr(e)}).encode(), "application/json", 500)


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    print(f"cerbeto panel on http://{host}:{port}  (read-only; Ctrl+C to stop)")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
