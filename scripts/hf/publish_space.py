"""Step 4: add a model to the static chat Space (the page, models.json and the ONNX int8 model from export_web.py). The Space keeps
every model in models.json, so visitors can compare them; --default puts this one first (what a new visitor gets).

    uv run --with huggingface_hub python ../scripts/hf/publish_space.py v6-1032M --short v6 --tokens 1423e6 \
        --label "v6 — published version" --note "v6: 1.42B tokens read, tuned to answer questions" --default
    uv run --with huggingface_hub python ../scripts/hf/publish_space.py bil2-avg --short preview --tokens 2485e6 \
        --label "preview — 2.5B tokens" --note "..."
--keep: the model folder is already in the Space (only models.json and the page change). --remove: drop the tag from the list.
"""
import argparse, json, shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from _hub import SPACE, api

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--short", help="badge next to the name, e.g. v6 or preview")
ap.add_argument("--label", help="the line in the model picker")
ap.add_argument("--tokens", type=float, help="tokens read in the model's whole life")
ap.add_argument("--note", default="", help="one sentence under the chat")
ap.add_argument("--default", action="store_true")
ap.add_argument("--keep", action="store_true")
ap.add_argument("--remove", action="store_true")
a = ap.parse_args()

here, model = Path(__file__).parent, Path("/tmp/web/model")
hub = api()
hub.create_repo(SPACE, repo_type="space", space_sdk="static", private=False, exist_ok=True)
try:
    models = json.loads(Path(hf_hub_download(SPACE, "models.json", repo_type="space", token=hub.token, force_download=True)).read_text())
except EntryNotFoundError:
    models = []
old = next((m for m in models if m["tag"] == a.tag), None)
models = [m for m in models if m["tag"] != a.tag]
if not a.remove:
    entry = {"tag": a.tag, "short": a.short or old["short"], "label": a.label or old["label"],
             "tokens": a.tokens or old["tokens"], "note": a.note or (old or {}).get("note", "")}
    models.insert(0 if a.default or not models else len(models), entry)

# each checkpoint in its own folder: browsers cache the model by its address, so a new tag is always downloaded
site = Path("/tmp/web/site"); shutil.rmtree(site, ignore_errors=True); site.mkdir(parents=True)
if not a.remove and not a.keep:
    dst = site / "models" / a.tag; (dst / "onnx").mkdir(parents=True)
    for f in model.glob("*.json"):
        shutil.copy(f, dst / f.name)
    shutil.copy(model / "onnx/model_quantized.onnx", dst / "onnx/model_quantized.onnx")   # int8, ~110 MB; fp32 stays local
for f in ["index.html", "README.md"]:
    shutil.copy(here / "web" / f, site / f)
(site / "models.json").write_text(json.dumps(models, indent=1))
hub.upload_folder(folder_path=str(site), repo_id=SPACE, repo_type="space",
                  commit_message=f"{'remove' if a.remove else 'add'} {a.tag} ({len(models)} models)",
                  delete_patterns=[f"models/{a.tag}/**"] if a.remove else None)
print("PUBLISHED", f"https://huggingface.co/spaces/{SPACE}", [m["tag"] for m in models])
