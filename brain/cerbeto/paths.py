import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"
DATA = Path(os.environ.get("CERBETO_DATA", REPO / "data"))
LIBRARY = DATA / "library"
SHELVES = DATA / "shelves"
TOKENIZER_FILE = Path(__file__).resolve().parent / "tokenizer" / "artifacts" / "tokenizer.json"
