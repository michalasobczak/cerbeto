"""Step 4: publish the static chat Space (the page + the ONNX int8 model from export_web.py), labelled with the checkpoint tag.

    uv run --with huggingface_hub python ../scripts/hf/publish_space.py bil-1000M
"""
import json, shutil, sys
from pathlib import Path

from _hub import SPACE, api

tag, here, model = sys.argv[1], Path(__file__).parent, Path("/tmp/web/model")
tokens = tag.rsplit("-", 1)[-1]
site = Path("/tmp/web/site"); shutil.rmtree(site, ignore_errors=True); (site / "model/onnx").mkdir(parents=True)
for f in model.glob("*.json"):
    shutil.copy(f, site / "model" / f.name)
shutil.copy(model / "onnx/model_quantized.onnx", site / "model/onnx/model_quantized.onnx")   # int8, ~110 MB; fp32 stays local
for f in ["index.html", "README.md"]:
    (site / f).write_text((here / "web" / f).read_text().replace("{TAG}", tag).replace("{TOKENS}", tokens))

hub = api()
hub.create_repo(SPACE, repo_type="space", space_sdk="static", private=False, exist_ok=True)
hub.upload_folder(folder_path=str(site), repo_id=SPACE, repo_type="space", commit_message=f"Cerbeto chat in the browser ({tag}, ONNX int8)")
print("PUBLISHED", f"https://huggingface.co/spaces/{SPACE}")
