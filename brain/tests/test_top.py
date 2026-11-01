"""`cerbeto top`: run summary from metrics.jsonl (status, rate, ETA, growth) and rendering without a live terminal."""
import json
from datetime import datetime, timedelta

from rich.console import Console

from cerbeto import top

CFG = {"total_tokens": 100_000_000,
       "students": {"growing": {"grow": True}, "control": {"grow": False}},
       "growth": {"deadlines_tokens": [30_000_000, 60_000_000], "schedule": [[{}], [{}]]}}
T0 = datetime(2026, 11, 1, 12, 0, 0)


def _write(tmp_path, student, events):
    d = tmp_path / student
    d.mkdir(parents=True)
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n{\"partial")
    return tmp_path


def _eval(minutes, tokens, mean, params=7_000_000):
    return {"time": (T0 + timedelta(minutes=minutes)).isoformat(timespec="seconds"), "event": "eval",
            "tokens": tokens, "valid": {"s0_speech": mean + 1, "mean": mean}, "params": params,
            "phase": "A-speech", "lr": 1e-3, "train_loss": mean}


def test_running_student_rate_eta_and_next_deadline(tmp_path):
    root = _write(tmp_path, "growing", [
        _eval(0, 0, 9.0),
        _eval(10, 10_000_000, 4.0),
        {"time": (T0 + timedelta(minutes=10)).isoformat(), "event": "growth", "tokens": 10_000_000,
         "trigger": "plateau", "params": 14_000_000},
        _eval(20, 20_000_000, 3.5, 14_000_000),
    ])
    now = (T0 + timedelta(minutes=25)).timestamp()
    s = top.summarize("growing", now=now, run_root=root, cfg=CFG, proc=type("P", (), {"pid": 42})())
    assert s["status"] == "running" and s["pid"] == 42
    assert round(s["rate"]) == 16_667                      # 10M tokens in 10 minutes
    assert s["tokens_est"] == 25_000_000                   # 5 more minutes at that rate
    assert round(s["eta"]) == 75_000_000 * 600 // 10_000_000
    assert s["next_deadline"] == 30_000_000 and s["growth_steps"] == (1, 2)
    assert s["history"]["mean"] == [9.0, 4.0, 3.5]


def test_status_without_a_process(tmp_path):
    root = _write(tmp_path, "control", [_eval(0, 0, 9.0), _eval(10, 10_000_000, 4.0)])
    s = top.summarize("control", run_root=root, cfg=CFG, proc=None)
    assert s["status"] == "stopped" and "eta" not in s and s["next_deadline"] is None
    assert top.summarize("growing", run_root=tmp_path / "none", cfg=CFG, proc=None)["status"] == "not started"


def test_finished_run(tmp_path):
    root = _write(tmp_path, "growing", [_eval(0, 0, 9.0), {"time": T0.isoformat(), "event": "checkpoint",
                                                            "tokens": 100_000_000, "finished": True}])
    assert top.summarize("growing", run_root=root, cfg=CFG, proc=None)["status"] == "finished"


def test_sparkline_shows_the_recent_trend():
    line = top.sparkline([100.0] + list(range(20, 0, -1)), width=10)
    assert len(line) == 10 and line[0] == "█" and line[-1] == "▁"   # scaled to the last 10 values, not to 100


def test_render_does_not_crash(tmp_path):
    root = _write(tmp_path, "growing", [_eval(0, 0, 9.0), _eval(10, 10_000_000, 4.0)])
    s = top.summarize("growing", run_root=root, cfg=CFG, proc=None)
    m = {"ram_used": 2**34, "ram_total": 2**35, "swap_used": 0, "swap_total": 0, "cpu": 10.0, "gpu_util": 99,
         "gpu_mem": 2**30, "power": "AC", "thermal": "ok", "ollama": [], "procs": []}
    console = Console(width=120, record=True)
    console.print(top.render_student(s), top.render_machine(m))
    text = console.export_text()
    assert "STOPPED" in text and "10.0M / 100.0M" in text and "GPU free" in text
