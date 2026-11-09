"""Step 1: publish a Cerbeto checkpoint (native format) to the model repo, tagged v<N>-<tokens>M (N counts the
published versions: v1, v2, ...), with the model card. Prints the tag for step 4.

    uv run --with huggingface_hub python ../scripts/hf/publish_model.py checkpoints/final/bil/latest
"""
import json, re, shutil, sys
from pathlib import Path

from _hub import MODEL_REPO, REPO, api

src = REPO / sys.argv[1]
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

hub = api()
hub.create_repo(MODEL_REPO, private=False, exist_ok=True)
tags = {t.name: t.target_commit for t in hub.list_repo_refs(MODEL_REPO).tags}
if not any(re.fullmatch(r"v\d+-\d+M", t) for t in tags):   # the first versions were tagged bil-<tokens>M: number them v1, v2, ...
    old = sorted((t for t in tags if re.fullmatch(r"bil-\d+M", t)), key=lambda t: int(t[4:-1]))
    for i, t in enumerate(old, 1):
        hub.create_tag(MODEL_REPO, tag=f"v{i}-{t[4:]}", revision=tags[t])
        tags[f"v{i}-{t[4:]}"] = tags[t]
version = 1 + max((int(m[1]) for t in tags if (m := re.fullmatch(r"v(\d+)-\d+M", t))), default=0)
tag = f"v{version}-{mtok}"
card = (Path(__file__).parent / "card.md").read_text()
(stage / "README.md").write_text(card.replace("{STEP}", f"{ft['step']:,}").replace("{TOKENS}", mtok).replace("{TAG}", tag))
hub.upload_folder(folder_path=str(stage), repo_id=MODEL_REPO, commit_message=f"{tag}: {mtok} tokens (step {ft['step']})")
hub.create_tag(MODEL_REPO, tag=tag)
shutil.rmtree(stage)
print("PUBLISHED", f"https://huggingface.co/{MODEL_REPO}", "tag", tag)
