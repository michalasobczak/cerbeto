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
from cerbeto.school.day import lesson_weights, run_day, teacher
from cerbeto.school.sleep import Interrupted, estimate_ewc, run_sleep
from cerbeto.school.validation import score, validation_set

CKPT = paths.REPO / "checkpoints" / "school"
RUN = paths.REPO / "runs" / "school"
BASELINE = paths.REPO / "runs" / "exams" / "growing"     # the nursery student's exam: weights of day 1
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
                                self.teacher_name, self.stop, self.log, weights=self.weights(stage))
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
                                      seed=st["day"], stop=self.stop, log=self.log, memories=memories,
                                      validate=lambda: score(model, validation_set(stage), self.cfg["day"]["k"],
                                                             self.device))
                except Interrupted:
                    self.log("sleep interrupted — no new version, the student stays on " + st["active"])
                    break
                hippo.mark_slept([e.id for e in experiences])
                if stats["kept_step"] == 0:      # the best moment of the night was before it: nothing to examine
                    self.unchanged(stats)
                    model = self.load(st["active"])
                    done_cycles += 1
                    continue
                n = len(list(CKPT.glob("v*")))
                cand = f"v{n:04d}"
                checkpoint.save(model, CKPT / cand, parent=CKPT / st["active"],
                                meta={"school": {"day": st["day"], "stage": stage, "sleep": stats}})
                st.update(phase="exam", candidate=cand)
                self.save_state()
            if st["phase"] == "exam":
                model = self.load(st["candidate"])
                self.examine(model)
                model = self.load(st["active"])
                done_cycles += 1
        self.save_state()
        self.log("school stopped" if not st["done"] else "school finished: all stages passed")

    def grow(self, extra_mlp: int = 0, extra_heads: int = 0, reason: str = "") -> str:
        """Grow the active version into a new one with the same behaviour (|Δ loss| guard, parameter limit), and
        make it active. Only while no candidate waits for its exam (phase `day`, or `sleep` before the night ran);
        the next night trains the new neurons."""
        from cerbeto.model import growth
        from cerbeto.train.loop import student_yaml
        from cerbeto.train.data import ShelfStream
        import numpy as np
        st = self.state
        if st is None or st["phase"] == "exam" or st.get("candidate"):
            raise SystemExit("grow only when no candidate waits for its exam; stop the school first")
        sy = student_yaml()
        model = self.load(st["active"])
        probe = ShelfStream("s0_speech", "valid", model.cfg.max_seq_len).batch(8, np.random.default_rng(1), self.device)
        new = growth.apply(model, "grow_width", probe, k=sy["recurrence"]["k_eval"], max_params=sy["max_params"],
                           max_delta_loss=sy["growth"]["max_delta_loss"], log_path=RUN / "growth_log.jsonl",
                           reason=reason, extra_mlp=extra_mlp, extra_heads=extra_heads, std=sy["growth"]["new_weight_std"])
        cand = f"v{len(list(CKPT.glob('v*'))):04d}"
        checkpoint.save(new, CKPT / cand, parent=CKPT / st["active"],
                        meta={"school": {"day": st["day"], "stage": st["stage"], "grown": {
                            "extra_mlp": extra_mlp, "extra_heads": extra_heads, "reason": reason}}})
        (RUN / "ewc.safetensors").unlink(missing_ok=True)      # importance is re-estimated for the new shape
        self.log(f"GROWN {st['active']} → {cand} · {model.num_params():,} → {new.num_params():,} parameters · "
                 f"+{extra_mlp} MLP neurons, +{extra_heads} heads per block · {reason}")
        st["active"] = cand
        (CKPT / "LATEST").write_text(cand)
        self.save_state()
        return cand

    def weights(self, stage: str) -> dict[str, float]:
        """Lesson weights of the day from the active version's last exam (the nursery exam on day 1)."""
        f = RUN / "exams" / f"{self.state['active']}.json"
        if not f.exists():
            f = BASELINE / f"{self.cfg['start_version']}.json"
        results = json.loads(f.read_text())["results"].get(stage) if f.exists() else None
        w = lesson_weights(results, self.cfg["day"])
        self.log("lesson weights · " + " · ".join(f"{k} {v:.2f}" for k, v in sorted(w.items()) if k.split(".")[0]
                                                  in {s.split(".")[0] for s in (results or {})} | {"mix"}))
        return w

    def progress(self, gained: bool) -> None:
        st = self.state
        st["plateau"] = 0 if gained else st["plateau"] + 1
        if st["plateau"] >= self.cfg["exam"]["plateau_cycles"]:
            self.log(f"no progress for {st['plateau']} cycles — growth wanted (`cerbeto school grow`, spec 0005)")
            st["plateau"] = 0

    def unchanged(self, stats: dict) -> None:
        """A night whose best moment was its start: the weights are those of the active version — no new version,
        no exam, no promotion; it counts as a cycle without progress."""
        st = self.state
        self.log(f"night changed nothing (validation best before sleep, {100 * max(v for _, v in stats['validation']):.1f}%) — "
                 f"no new version, the student stays on {st['active']}")
        with (RUN / "cycles.jsonl").open("a") as f:
            f.write(json.dumps({"day": st["day"], "candidate": None, "stage": st["stage"], "scores": {},
                                "promoted": False, "why": "night changed nothing", "minutes": 0,
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        self.progress(False)
        st.update(phase="day", candidate=None, day=st["day"] + 1)
        self.save_state()

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
        self.progress(ok and gain * 100 >= cfg["exam"]["min_gain_pp"])
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
