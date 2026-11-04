"""`cerbeto school run` (spec 0005): the automatic rhythm day → sleep → exam → promotion or rollback.

The school works on its own copy of a nursery student (`checkpoints/school/`); the nursery checkpoints are never
changed. State lives in `runs/school/state.json` and every phase is resumable, so Ctrl+C is safe at any moment:
- during the day: finished lessons are already in the hippocampus; the day continues where it stopped;
- during sleep: nothing is saved, the student stays on the active version; the sleep starts again;
- during the exam (a second Ctrl+C stops it at once): the candidate is examined again.

Promotion: the current stage's score (mean of id, ood_hard, ood_combo over code-graded skills) does not drop and
no earlier stage loses more than `forgetting_pp`. Otherwise rollback to the active version. A stage is passed by the
0004 thresholds; then the next stage starts. No progress for `plateau_cycles` cycles asks for growth.
"""
import json
import signal
import time
from pathlib import Path

import yaml

from cerbeto import paths
from cerbeto.exams.school import evaluate, load_pools
from cerbeto.memory.hippocampus import Hippocampus
from cerbeto.model import checkpoint
from cerbeto.model.consolidation import EWC
from cerbeto.model.student import pick_device
from cerbeto.school.day import run_day, teacher
from cerbeto.school.sleep import Interrupted, estimate_ewc, run_sleep

CKPT = paths.REPO / "checkpoints" / "school"
RUN = paths.REPO / "runs" / "school"
STATE = RUN / "state.json"
STUDENT = "school"


def config() -> dict:
    return yaml.safe_load((paths.CONFIG / "rhythm.yaml").read_text())


def passing() -> dict:
    return yaml.safe_load((paths.CONFIG / "curriculum.yaml").read_text())["passing"]


def stage_scores(results: dict) -> dict[str, float]:
    """Per stage: mean over code-graded skills of the mean of the three exam variants."""
    out = {}
    for stage, skills in results.items():
        vals = [sum(v[x]["accuracy"] for x in ("id", "ood_hard", "ood_combo")) / 3
                for name, v in skills.items() if not name.startswith("_") and v["id"]["graded_by"] == "code"]
        out[stage] = sum(vals) / len(vals)
    return out


def decide(scores: dict[str, float], stage: str, best: float | None, reference: dict[str, float],
           forgetting_pp: float) -> tuple[bool, str]:
    """Promotion gate: the current stage does not drop, no earlier stage forgets more than `forgetting_pp`."""
    for s, ref in reference.items():
        if s != stage and scores.get(s, 0) < ref - forgetting_pp / 100:
            return False, f"{s} forgot {100 * (ref - scores[s]):.1f} pp"
    if best is not None and scores[stage] < best:
        return False, f"{stage} dropped {100 * (best - scores[stage]):.1f} pp"
    return True, "ok"


class School:
    def __init__(self, teacher_name: str | None = None, device: str | None = None, log_file: Path = RUN / "log.txt"):
        self.cfg = config()
        self.device = pick_device() if device is None else device
        self.teacher_name = teacher_name
        self.ask = teacher(teacher_name)
        self.stopping = 0
        RUN.mkdir(parents=True, exist_ok=True)
        self.log_file = log_file
        self.state = json.loads(STATE.read_text()) if STATE.exists() else None

    # ── bookkeeping ──────────────────────────────────────────────────────────────────────────────────
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with self.log_file.open("a") as f:
            f.write(line + "\n")

    def save_state(self) -> None:
        STATE.write_text(json.dumps(self.state, indent=1))

    def stop(self) -> bool:
        return self.stopping > 0

    def _sigint(self, *_):
        self.stopping += 1
        if self.stopping == 1:
            self.log("stopping after the current step (Ctrl+C again to stop at once)")
        else:
            raise KeyboardInterrupt

    def start(self) -> None:
        """First run: copy the nursery student into the school as v0000."""
        src = paths.REPO / "checkpoints" / self.cfg["student"] / self.cfg["start_version"]
        model = checkpoint.load(src, "cpu")
        checkpoint.save(model, CKPT / "v0000", parent=src, meta={"school": {"from": str(src)}})
        (CKPT / "LATEST").write_text("v0000")
        self.state = {"from": str(src), "stage": self.cfg["stages"][0], "active": "v0000", "day": 1, "phase": "day",
                      "candidate": None, "best": None, "reference": {}, "plateau": 0, "passed": [], "done": False}
        self.save_state()
        self.log(f"school started from {src.relative_to(paths.REPO)}")

    def load(self, version: str):
        return checkpoint.load(CKPT / version, self.device)

    def ewc(self, model) -> EWC:
        f = RUN / "ewc.safetensors"
        if not f.exists():
            self.log("estimating synaptic consolidation (EWC) on the nursery language")
            estimate_ewc(model, self.device, self.cfg["sleep"]["ewc_batches"]).save(f)
        return EWC.load(f, self.device)

    # ── the rhythm ───────────────────────────────────────────────────────────────────────────────────
    def run(self, cycles: int | None = None) -> None:
        if self.state is None:
            self.start()
        signal.signal(signal.SIGINT, self._sigint)
        st = self.state
        model = self.load(st["active"])
        hippo = Hippocampus()
        done_cycles = 0
        while not st["done"] and not self.stop() and (cycles is None or done_cycles < cycles):
            stage = st["stage"]
            if st["phase"] == "day":
                self.log(f"── day {st['day']} · {stage} · active {st['active']}")
                stats = run_day(model, self.device, STUDENT, st["day"], stage, self.cfg["day"], self.ask,
                                self.teacher_name, self.stop, self.log)
                if self.stop():
                    break
                self.log(f"day {st['day']} done: {stats}")
                st["phase"] = "sleep"
                self.save_state()
            if st["phase"] == "sleep":
                experiences = hippo.pending(STUDENT)
                earlier = st["passed"]
                memories = hippo.memories(STUDENT, stage)
                self.log(f"── sleep · {len(experiences)} experiences · {len(memories)} memories of earlier days · "
                         f"replay of {earlier or 'nursery only'}")
                try:
                    stats = run_sleep(model, self.device, experiences, earlier, self.cfg["sleep"], self.ewc(model),
                                      seed=st["day"], stop=self.stop, log=self.log, memories=memories)
                except Interrupted:
                    self.log("sleep interrupted — no new version, the student stays on " + st["active"])
                    break
                n = len(list(CKPT.glob("v*")))
                cand = f"v{n:04d}"
                checkpoint.save(model, CKPT / cand, parent=CKPT / st["active"],
                                meta={"school": {"day": st["day"], "stage": stage, "sleep": stats}})
                hippo.mark_slept([e.id for e in experiences])
                st.update(phase="exam", candidate=cand)
                self.save_state()
            if st["phase"] == "exam":
                model = self.load(st["candidate"])
                self.examine(model)
                model = self.load(st["active"])
                done_cycles += 1
        self.save_state()
        self.log("school stopped" if not st["done"] else "school finished: all stages passed")

    def examine(self, model) -> None:
        st, cfg = self.state, self.cfg
        stages = cfg["stages"][: cfg["stages"].index(st["stage"]) + 1]
        pools = {k: v for k, v in load_pools().items() if k[0] in stages}
        self.log(f"── exam of {st['candidate']} · stages {stages}")
        started = time.time()
        out = evaluate(model, cfg["day"]["k"], self.device, pools=pools)
        scores = stage_scores(out["results"])
        (RUN / "exams").mkdir(exist_ok=True)
        (RUN / "exams" / f"{st['candidate']}.json").write_text(json.dumps(
            {"scores": scores, "results": out["results"], "items": out["items"]}, indent=1))
        stage = st["stage"]
        ok, why = decide(scores, stage, st["best"], st["reference"], passing()["forgetting_pp"])
        gain = scores[stage] - (st["best"] or 0)
        record = {"day": st["day"], "candidate": st["candidate"], "stage": stage, "scores": scores,
                  "promoted": ok, "why": why, "minutes": round((time.time() - started) / 60, 1),
                  "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with (RUN / "cycles.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        summary = " · ".join(f"{s} {100 * v:.1f}%" for s, v in scores.items())
        if ok:
            st["active"], st["best"] = st["candidate"], scores[stage]
            (CKPT / "LATEST").write_text(st["active"])
            self.log(f"PROMOTED {st['active']} · {summary} · gain {100 * gain:+.1f} pp")
        else:
            self.log(f"ROLLBACK to {st['active']} · {summary} · {why}")
        st["plateau"] = 0 if ok and gain * 100 >= cfg["exam"]["min_gain_pp"] else st["plateau"] + 1
        if st["plateau"] >= cfg["exam"]["plateau_cycles"]:
            self.log(f"no progress for {st['plateau']} cycles — growth wanted, but the student is at the parameter "
                     f"limit (open question in spec 0005)")
            st["plateau"] = 0
        if ok and out["results"][stage]["_passed"]:
            st["reference"][stage] = scores[stage]
            st["passed"].append(stage)
            i = cfg["stages"].index(stage) + 1
            self.log(f"*** {stage} PASSED with {st['active']} ***")
            if i == len(cfg["stages"]):
                st["done"] = True
            else:
                st["stage"], st["best"], st["plateau"] = cfg["stages"][i], None, 0
                (RUN / "ewc.safetensors").unlink(missing_ok=True)   # consolidate what was learned so far
        for s in st["passed"]:                      # keep references of passed stages current with the active version
            if ok and s != stage:
                st["reference"][s] = max(st["reference"][s], scores[s])
        st.update(phase="day", candidate=None, day=st["day"] + 1)
        self.save_state()


def status() -> None:
    if not STATE.exists():
        print("school not started — `cerbeto school run`")
        return
    st = json.loads(STATE.read_text())
    print(json.dumps({k: v for k, v in st.items()}, indent=1))
    f = RUN / "cycles.jsonl"
    if f.exists():
        for line in f.read_text().splitlines()[-10:]:
            r = json.loads(line)
            print(f"day {r['day']:3d} {r['candidate']} {r['stage']:9s} "
                  + " ".join(f"{s} {100 * v:5.1f}%" for s, v in r["scores"].items())
                  + f"  {'promoted' if r['promoted'] else 'rollback: ' + r['why']}")
