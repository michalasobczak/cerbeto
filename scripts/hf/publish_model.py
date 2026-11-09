"""Step 1: publish a Cerbeto checkpoint (native format) to the model repo, tagged <name>-<tokens>M, with the model card.

    uv run --with huggingface_hub python ../scripts/hf/publish_model.py checkpoints/final/bil/latest bil
"""
import json, shutil, sys
from pathlib import Path

from _hub import MODEL_REPO, REPO, api

src, name = REPO / sys.argv[1], sys.argv[2]
stage = Path("/tmp/hf_cerbeto"); shutil.rmtree(stage, ignore_errors=True); stage.mkdir()
for _ in range(5):                    # `latest` is replaced every 30 minutes: weights and metadata must come from the same save
    a1 = (src / "architecture.json").read_text()
    shutil.copy(src / "model.safetensors", stage)
    a2 = (src / "architecture.json").read_text()
    if a1 == a2:
        break
else:
    sys.exit("the checkpoint kept changing")
(stage / "architecture.json").write_text(a2)
shutil.copy(REPO / "brain/cerbeto/model/student.py", stage)
shutil.copy(REPO / "brain/cerbeto/tokenizer/artifacts/tokenizer.json", stage)
shutil.copy(REPO / "LICENSE.md", stage)
ft = json.loads(a2)["final_test"]
mtok = f"{ft['tokens'] / 1e6:.0f}M"
card = (Path(__file__).parent / "card.md").read_text()
(stage / "README.md").write_text(card.replace("{STEP}", f"{ft['step']:,}").replace("{TOKENS}", mtok))

hub = api()
hub.create_repo(MODEL_REPO, private=False, exist_ok=True)
hub.upload_folder(folder_path=str(stage), repo_id=MODEL_REPO, commit_message=f"{name} at {mtok} tokens (step {ft['step']})")
hub.create_tag(MODEL_REPO, tag=f"{name}-{mtok}", exist_ok=True)
shutil.rmtree(stage)
print("PUBLISHED", f"https://huggingface.co/{MODEL_REPO}", "tag", f"{name}-{mtok}")
