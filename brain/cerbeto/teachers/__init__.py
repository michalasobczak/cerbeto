"""Teacher clients — large LLMs that teach and grade the student (spec 0005)."""
import json
import urllib.request

import yaml

from cerbeto import paths


def teacher_config(name: str | None = None) -> dict:
    cfg = yaml.safe_load((paths.CONFIG / "teachers.yaml").read_text())
    return cfg["teachers"][name or cfg["default"]]


def chat(messages: list[dict], teacher: str | None = None, temperature: float = 0.2, timeout: int = 300,
         json_output: bool = False, think: bool = False) -> str:
    t = teacher_config(teacher)
    if t["provider"] != "ollama":
        raise NotImplementedError(f"provider {t['provider']} arrives with spec 0005")
    body = {"model": t["model"], "messages": messages, "stream": False, "think": think,
            "options": {"temperature": temperature}}
    if json_output:
        body["format"] = "json"
    req = urllib.request.Request(f"{t['base_url']}/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["message"]["content"]
