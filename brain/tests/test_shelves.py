"""Gates G3 and N1 (spec 0001): every shelf has a complete manifest; no library data is tracked by git."""
import json
import subprocess

import pytest

from cerbeto import paths
from cerbeto.library.shelves import digest, is_valid_doc, load_config

REQUIRED = {"shelf", "stage", "sources", "license", "fk_max", "docs", "words"}


def test_every_configured_shelf_has_a_manifest():
    if not paths.SHELVES.exists():
        pytest.skip("shelves not built yet")
    for name in load_config()["shelves"]:
        manifest = json.loads((paths.SHELVES / name / "manifest.json").read_text())
        assert REQUIRED <= manifest.keys()
        assert manifest["license"]
        assert all(src["license"] for src in manifest["sources"])
        assert manifest["docs"]["train"] > 0 and manifest["docs"]["valid"] > 0


def test_tokens_recorded_after_tokenization():
    if not any(paths.SHELVES.glob("*/train.bin")):
        pytest.skip("shelves not tokenized yet")
    for name in load_config()["shelves"]:
        manifest = json.loads((paths.SHELVES / name / "manifest.json").read_text())
        assert manifest["tokens"]["train"] > 0


def test_dedup_digest_ignores_case_and_whitespace():
    assert digest("Tom  has a\nball.") == digest("tom has a ball.")


def test_valid_split_is_deterministic_and_small():
    docs = [f"document number {i}" for i in range(20_000)]
    picked = [d for d in docs if is_valid_doc(d, 5)]
    assert picked == [d for d in docs if is_valid_doc(d, 5)]
    assert 40 < len(picked) < 180


def test_no_library_data_tracked_by_git():
    out = subprocess.run(["git", "ls-files", "data/"], cwd=paths.REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
