"""Shared: the Hugging Face client from the token in the repo's .env (hf=…), never printed."""
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[2]
MODEL_REPO, SPACE = "michalasobczak/cerbeto", "michalasobczak/cerbeto-chat"


def api() -> HfApi:
    env = {k.strip().lower(): v.strip().strip("'\"")
           for k, v in (line.split("=", 1) for line in (REPO / ".env").read_text().splitlines() if "=" in line)}
    return HfApi(token=env["hf"])
