"""Refresh the model card (card.md) on the hub without new weights: step, tokens and version come from the newest v<N>-<tokens>M tag.

    uv run --with huggingface_hub python ../scripts/hf/update_card.py
"""
import json, re
from pathlib import Path

from huggingface_hub import hf_hub_download

from _hub import MODEL_REPO, api

hub = api()
tag = max((t.name for t in hub.list_repo_refs(MODEL_REPO).tags if re.fullmatch(r"v\d+-\d+M", t.name)), key=lambda t: int(t[1:].split("-")[0]))
ft = json.loads(Path(hf_hub_download(MODEL_REPO, "architecture.json", revision=tag)).read_text())["final_test"]
card = (Path(__file__).parent / "card.md").read_text()
card = card.replace("{STEP}", f"{ft['step']:,}").replace("{TOKENS}", f"{ft['tokens'] / 1e6:.0f}M").replace("{TAG}", tag)
hub.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=MODEL_REPO, commit_message=f"model card for {tag}")
print("UPDATED card", tag)
