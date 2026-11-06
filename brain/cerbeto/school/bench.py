"""`cerbeto school bench`: replay a school night offline in variants, one hypothesis per variant (spec 0005).

A night of the school takes ~1.5 h with its day and changes several things at once. The bench takes a copy of a
school version and lessons already in the hippocampus (no teacher, no day), runs the night with one setting changed,
and records the validation per skill at every check. Nothing in the school changes: checkpoints and the database are
only read; results go to `runs/bench/<variant>.json`. Variants: `config/bench.yaml`.
"""
import json
import sqlite3
import time
from pathlib import Path

import yaml

from cerbeto import paths
from cerbeto.memory.hippocampus import DB, Experience
from cerbeto.model import checkpoint
from cerbeto.model.consolidation import EWC
from cerbeto.model.student import pick_device
from cerbeto.school.sleep import estimate_ewc, estimate_skills_ewc, run_sleep
from cerbeto.school.validation import by_skill, validation_set

OUT = paths.REPO / "runs" / "bench"
CKPT = paths.REPO / "checkpoints" / "school"
STUDENT = "school"
COMPARES = "%The question says:%"
YES_NO = "(target LIKE '%<|answer|> Yes%' OR target LIKE '%<|answer|> No%')"
YES_NO_SKILLS = ("social.agree_disagree", "world.categories", "mix.opposite_statement", "mix.not_statement",
                 "read.facts")


def config() -> dict:
    return yaml.safe_load((paths.CONFIG / "bench.yaml").read_text())


def variant(name: str) -> dict:
    cfg = config()
    v = {**cfg["base"], **cfg["variants"][name]}
    v["sleep"] = {**cfg["base"].get("sleep", {}), **cfg["variants"][name].get("sleep", {})}
    return v


def _inconsistent(skill: str, target: str) -> bool:
    """A yes/no lesson whose thinking does not compare the claim with the fact (the retired format)."""
    return skill in YES_NO_SKILLS and target.split("<|answer|>")[-1].strip()[:3] in ("Yes", "No.", "No,") \
        and "The question says:" not in target


def lessons(days: list[int], consistent: bool, dedup: bool) -> tuple[list[Experience], list[tuple[str, str, bool]]]:
    """(the night's experiences: lessons of `days`; memories: earlier preschool lessons) from the hippocampus."""
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    marks = ",".join("?" * len(days))
    rows = db.execute(f"SELECT student, day, stage, skill, origin, task_hash, prompt, target, student_reply, correct, "
                      f"verified, id FROM experiences WHERE student = ? AND verified = 1 AND day IN ({marks}) "
                      f"ORDER BY id", (STUDENT, *days)).fetchall()
    today = [Experience(*r[:9], bool(r[9]), bool(r[10]), r[11]) for r in rows]
    mem = db.execute(f"SELECT skill, prompt, target, correct FROM experiences WHERE student = ? AND verified = 1 "
                     f"AND stage = 'preschool' AND day < ? ORDER BY id DESC", (STUDENT, min(days))).fetchall()
    if consistent:
        today = [e for e in today if not _inconsistent(e.skill, e.target)]
        mem = [m for m in mem if not _inconsistent(m[0], m[2])]
    memories = [(p, t, bool(c)) for _, p, t, c in mem]
    if dedup:                                   # each prompt once: fewer repeats of the same small-space question
        seen, kept = set(), []
        for m in memories:
            if m[0] not in seen:
                seen.add(m[0])
                kept.append(m)
        memories = kept
    return today, memories[:20_000]


def ewc_for(version: str, model, device, batches: int) -> EWC:
    f = OUT / f"ewc-{version}.safetensors"
    if not f.exists():
        estimate_ewc(model, device, batches).save(f)
    return EWC.load(f, device)


def run(name: str, device: str | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    v = variant(name)
    rhythm = yaml.safe_load((paths.CONFIG / "rhythm.yaml").read_text())
    sleep_cfg = {**rhythm["sleep"], **v["sleep"]}
    dev = pick_device() if device is None else device
    model = checkpoint.load(CKPT / v["version"], dev)
    today, memories = lessons(v["days"], v.get("consistent_yes_no", True), v.get("dedup_memories", False))
    if v.get("limit"):                          # a short run (checks of the bench itself)
        today = today[: v["limit"]]
    tasks = validation_set("preschool")
    checks: list[dict] = []
    log_file = OUT / f"{name}.log"
    log_file.unlink(missing_ok=True)

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} [{name}] {msg}"
        print(line, flush=True)
        with log_file.open("a") as f:
            f.write(line + "\n")

    def validate() -> float:
        r = by_skill(model, tasks, rhythm["day"]["k"], dev)
        checks.append(r)
        return r["_all"]

    log(f"{v['version']} · days {v['days']} · {len(today)} experiences · {len(memories)} memories · "
        f"sleep overrides {v['sleep'] or '—'}")
    ewc = ewc_for(v["version"], model, dev, sleep_cfg["ewc_batches"]) if sleep_cfg["ewc_lambda"] else None
    started = time.time()
    skills_ewc = None
    if sleep_cfg.get("ewc_skills_lambda"):
        t0 = time.time()
        skills_ewc = estimate_skills_ewc(model, dev, memories, sleep_cfg.get("ewc_skills_lessons", 256))
        log(f"skills EWC from {sleep_cfg.get('ewc_skills_lessons', 256)} earlier right lessons "
            f"({time.time() - t0:.0f}s) · lambda {sleep_cfg['ewc_skills_lambda']}")
    stats = run_sleep(model, dev, today, [], sleep_cfg, ewc, seed=v.get("seed", 1), log=log, memories=memories,
                      validate=validate, skills_ewc=skills_ewc)
    steps = [s for s, _ in stats["validation"]]
    out = {"variant": name, "settings": v, "experiences": len(today), "memories": len(memories),
           "minutes": round((time.time() - started) / 60, 1), "kept_step": stats["kept_step"],
           "lesson_loss": stats["lesson_loss"], "language_loss": stats["language_loss"],
           "checks": [{"step": s, **c} for s, c in zip(steps, checks)]}
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1))
    return out


def summary() -> None:
    """One line per finished variant: validation at the start, best (step), end; per-skill change at the end."""
    rows = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    if not rows:
        print("no bench results yet — `cerbeto school bench run <variant>`")
        return
    print(f"{'variant':16s} {'start':>6s} {'best':>6s} {'step':>5s} {'end':>6s}  lesson  change per skill at the end")
    for r in rows:
        c = r["checks"]
        best = max(c, key=lambda x: x["_all"])
        skills = " ".join(f"{k.split('.')[-1][:8]} {100 * (c[-1][k] - c[0][k]):+.0f}" for k in c[0]
                          if k not in ("_all", "step"))
        print(f"{r['variant']:16s} {100 * c[0]['_all']:6.1f} {100 * best['_all']:6.1f} {best['step']:5d} "
              f"{100 * c[-1]['_all']:6.1f}  {r['lesson_loss']:.3f}  {skills}")
