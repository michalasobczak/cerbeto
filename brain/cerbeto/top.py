"""`cerbeto top`: a live terminal dashboard of the students and the machine, in the spirit of `top`.

Read-only: it only reads files the training already writes (metrics.jsonl, checkpoints, exam results) and asks
the OS and the local Ollama what is running, so it never slows training down.
"""
import json
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import psutil
import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from cerbeto import paths

STUDENTS = ("growing", "control")
SPARK = "▁▂▃▄▅▆▇█"


# ── data ─────────────────────────────────────────────────────────────────────────────────────────────

def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:  # a line being written right now
            pass
    return events


def sparkline(values: list[float], width: int = 24) -> str:
    values = values[-width:]  # recent trend; the whole run would be flattened by the first, random-weight eval
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(SPARK[min(len(SPARK) - 1, int((v - lo) / span * (len(SPARK) - 1)))] for v in values)


def _ts(event: dict) -> float:
    return datetime.fromisoformat(event["time"]).timestamp()


def training_process(student: str) -> psutil.Process | None:
    for proc in psutil.process_iter(["cmdline"]):
        cmd = proc.info.get("cmdline") or []
        if "nursery" in cmd and "run" in cmd and student in cmd and any("cerbeto" in c for c in cmd):
            if not any(c.endswith("uv") for c in cmd[:1]):  # the python child, not the `uv run` wrapper
                return proc
    return None


def summarize(student: str, now: float | None = None, run_root: Path | None = None,
              cfg: dict | None = None, proc: psutil.Process | None | bool = False) -> dict:
    """Everything the dashboard shows about one student's nursery run."""
    now = now or time.time()
    run_root = run_root or paths.REPO / "runs" / "nursery"
    cfg = cfg or yaml.safe_load((paths.CONFIG / "nursery.yaml").read_text())
    events = read_events(run_root / student / "metrics.jsonl")
    if proc is False:
        proc = training_process(student)
    total = cfg["total_tokens"]
    s = {"student": student, "total": total, "events": len(events), "pid": proc.pid if proc else None}
    if not events:
        s["status"] = "running" if proc else "not started"
        return s
    evals = [e for e in events if e["event"] == "eval"]
    last = events[-1]
    finished = any(e.get("finished") for e in events if e["event"] == "checkpoint")
    s["status"] = "running" if proc else ("finished" if finished else "stopped")
    s["tokens"] = last["tokens"]
    s["last_event_age"] = now - _ts(last)
    if len(evals) >= 2:
        a, b = evals[-2], evals[-1]
        dt = _ts(b) - _ts(a)
        s["rate"] = (b["tokens"] - a["tokens"]) / dt if dt > 0 else None
    else:
        s["rate"] = evals[-1].get("tokens_per_s") if evals else None
    if s["status"] == "running" and s.get("rate"):
        # between evals the log is silent: estimate where the run is now (shown with "~")
        s["tokens_est"] = min(total, int(last["tokens"] + s["rate"] * s["last_event_age"]))
        s["eta"] = (total - s["tokens_est"]) / s["rate"]
    if evals:
        e = evals[-1]
        s.update(params=e["params"], phase=e.get("phase"), lr=e.get("lr"), train_loss=e.get("train_loss"),
                 valid=e["valid"], eval_age=now - _ts(e))
        s["history"] = {k: [x["valid"][k] for x in evals] for k in e["valid"]}
    s["growths"] = [g for g in events if g["event"] == "growth"]
    s["rejected"] = [g for g in events if g["event"] == "growth_rejected"]
    grows = cfg["students"][student].get("grow", False)
    deadlines = [d for d in cfg["growth"]["deadlines_tokens"] if d > s["tokens"]]
    s["next_deadline"] = deadlines[0] if grows and len(s["growths"]) < len(cfg["growth"]["schedule"]) and deadlines \
        else None
    s["growth_steps"] = (len(s["growths"]), len(cfg["growth"]["schedule"]) if grows else 0)
    ckpt = paths.REPO / "checkpoints" / student / "LATEST"
    s["version"] = ckpt.read_text().strip() if ckpt.exists() else None
    exams = sorted((run_root / student).glob("exam_*.json"))
    s["exam"] = json.loads(exams[-1].read_text()) if exams else None
    return s


def machine(teacher_url: str = "http://localhost:11434", pids: tuple[int, ...] = ()) -> dict:
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    m = {"ram_used": vm.total - vm.available, "ram_total": vm.total, "swap_used": sw.used, "swap_total": sw.total,
         "cpu": psutil.cpu_percent(interval=None)}
    try:  # Apple GPU: utilisation and memory from the IOAccelerator registry (no sudo needed)
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True,
                             timeout=2).stdout
        stats = {}
        for key in ("Device Utilization %", "In use system memory"):
            i = out.find(f'"{key}"=')
            if i >= 0:
                stats[key] = int(out[i + len(key) + 3:].split(",")[0].split("}")[0])
        m["gpu_util"] = stats.get("Device Utilization %")
        m["gpu_mem"] = stats.get("In use system memory")
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=2).stdout
        m["power"] = "AC" if "AC Power" in batt else "battery"
        pct = [w for w in batt.split() if w.endswith("%;")]
        if pct and m["power"] == "battery":
            m["power"] += f" {pct[0][:-1]}"
        therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=2).stdout
        m["thermal"] = "throttled" if "CPU_Speed_Limit" in therm and "= 100" not in therm else "ok"
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        with urllib.request.urlopen(f"{teacher_url}/api/ps", timeout=1) as resp:
            m["ollama"] = [{"name": x["name"], "size": x.get("size_vram") or x.get("size", 0),
                            "until": x.get("expires_at", "")} for x in json.load(resp).get("models", [])]
    except (OSError, ValueError):
        m["ollama"] = None
    m["procs"] = []
    for pid in pids:
        try:
            p = psutil.Process(pid)
            m["procs"].append({"pid": pid, "rss": p.memory_info().rss, "cpu": p.cpu_percent(interval=None)})
        except psutil.Error:
            pass
    return m


# ── rendering ────────────────────────────────────────────────────────────────────────────────────────

def _mtok(n: float) -> str:
    return f"{n / 1e6:,.1f}M"


def _dur(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    return f"{h}h {rem // 60:02d}m" if h else f"{rem // 60}m {rem % 60:02d}s"


def _gb(n: float) -> str:
    return f"{n / 2**30:.1f} GB"


STATUS_STYLE = {"running": "bold green", "finished": "bold cyan", "stopped": "bold red", "not started": "dim"}


def render_student(s: dict) -> Panel:
    head = Text()
    head.append(f"{s['status'].upper()}", STATUS_STYLE[s["status"]])
    if s.get("pid"):
        head.append(f"  pid {s['pid']}", "dim")
    if s.get("version"):
        head.append(f"  · {s['version']}")
    if s.get("params"):
        head.append(f"  · {s['params']:,} params")
    if s.get("phase"):
        head.append(f"  · phase {s['phase']}")
    rows = [head]
    if "tokens" in s:
        tokens = s.get("tokens_est", s["tokens"])
        bar = Table.grid(padding=(0, 1))
        bar.add_row(ProgressBar(total=s["total"], completed=tokens, width=40),
                    Text(f"{'~' if 'tokens_est' in s else ''}{_mtok(tokens)} / {_mtok(s['total'])}"
                         f"  ({tokens / s['total']:.0%})"))
        rows.append(bar)
        info = []
        if s.get("rate"):
            info.append(f"{s['rate']:,.0f} tok/s")
        if s.get("eta") is not None:
            info.append(f"ETA {_dur(s['eta'])} (~{time.strftime('%H:%M', time.localtime(time.time() + s['eta']))})")
        if s.get("lr") is not None:
            info.append(f"lr {s['lr']:.2e}")
        if s.get("eval_age") is not None:
            info.append(f"last eval {_dur(s['eval_age'])} ago")
        rows.append(Text("   ".join(info), "dim"))
        if s["status"] == "running" and s.get("last_event_age", 0) > 45 * 60:
            rows.append(Text(f"⚠ no log line for {_dur(s['last_event_age'])} — is it stuck?", "bold yellow"))
    if s.get("valid"):
        loss = Table(box=None, padding=(0, 2), show_edge=False)
        loss.add_column("validation loss", style="bold")
        loss.add_column("now", justify="right")
        loss.add_column("Δ last", justify="right")
        loss.add_column("start", justify="right", style="dim")
        loss.add_column("trend (last 24 evals)")
        for shelf, hist in s["history"].items():
            delta = hist[-1] - hist[-2] if len(hist) > 1 else 0.0
            style = "green" if delta < 0 else ("red" if delta > 0 else "")
            loss.add_row(shelf, f"{hist[-1]:.3f}", Text(f"{delta:+.3f}", style), f"{hist[0]:.2f}",
                         Text(sparkline(hist), "cyan"))
        rows.append(loss)
    if s.get("growths") or s.get("next_deadline") or s.get("rejected"):
        done, planned = s["growth_steps"]
        g = Text(f"growth {done}/{planned}: ", "bold")
        g.append("  ".join(f"{_mtok(e['tokens'])} {e['trigger']} → {e['params'] / 1e6:.1f}M" for e in s["growths"]))
        if s.get("next_deadline"):
            g.append(f"   next at the latest: {_mtok(s['next_deadline'])}", "yellow")
        if s.get("rejected"):
            g.append(f"   {len(s['rejected'])} rejected", "red")
        rows.append(g)
    if s.get("exam"):
        e = s["exam"]
        x = Text(f"exam {e['version']}: ", "bold")
        if "blimp" in e:
            acc = e["blimp"]["accuracy"]
            x.append(f"BLiMP {acc:.1%} ", "green" if acc >= 0.65 else "red")
        if "conversation" in e:
            x.append(f"· conversation probe {e['conversation']['fitting']:.0%}")
        rows.append(x)
    return Panel(Group(*rows), title=f"[bold]{s['student']}[/] student", title_align="left")


def render_machine(m: dict) -> Panel:
    t = Table.grid(padding=(0, 3))
    cells = [f"RAM {_gb(m['ram_used'])} / {_gb(m['ram_total'])}",
             f"swap {_gb(m['swap_used'])}" + (" ⚠" if m["swap_used"] > 2 * 2**30 else ""),
             f"CPU {m['cpu']:.0f}%"]
    if m.get("gpu_util") is not None:
        cells.append(f"GPU {m['gpu_util']}%" + (f" · {_gb(m['gpu_mem'])}" if m.get("gpu_mem") else ""))
    if m.get("power"):
        cells.append(f"power {m['power']}")
    if m.get("thermal"):
        cells.append(f"thermal {m['thermal']}")
    t.add_row(*cells)
    rows: list = [t]
    for p in m["procs"]:
        rows.append(Text(f"training pid {p['pid']}: RSS {_gb(p['rss'])}, CPU {p['cpu']:.0f}%", "dim"))
    if m["ollama"] is None:
        rows.append(Text("Ollama: not reachable", "dim"))
    elif not m["ollama"]:
        rows.append(Text("Ollama: no model loaded (GPU free for training)", "green"))
    else:
        names = ", ".join(f"{x['name']} ({_gb(x['size'])})" for x in m["ollama"])
        rows.append(Text(f"Ollama: {names} — shares the GPU with training", "yellow"))
    return Panel(Group(*rows), title="[bold]machine[/]", title_align="left")


def render(students: list[str]) -> Group:
    sums = [summarize(s) for s in students]
    pids = tuple(s["pid"] for s in sums if s.get("pid"))
    header = Text(f"cerbeto top — {time.strftime('%Y-%m-%d %H:%M:%S')}   ctrl-c to quit", "bold")
    return Group(header, *(render_student(s) for s in sums), render_machine(machine(pids=pids)))


def run(students: list[str] | None, interval: float, once: bool) -> None:
    students = students or list(STUDENTS)
    console = Console()
    if once:
        console.print(render(students))
        return
    try:
        with Live(render(students), console=console, screen=True, auto_refresh=False) as live:
            while True:
                time.sleep(interval)
                live.update(render(students), refresh=True)
    except KeyboardInterrupt:
        pass
