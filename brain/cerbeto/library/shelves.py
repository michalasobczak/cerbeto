"""Builds text shelves from the raw library: cleaning, librarian filter, dedup, train/valid split."""
import hashlib
import json
import os
import re
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path

import yaml

from cerbeto import paths
from cerbeto.library import sources
from cerbeto.library.readability import fk_grade, words


def load_config() -> dict:
    return yaml.safe_load((paths.CONFIG / "library.yaml").read_text())


def digest(text: str) -> bytes:
    return hashlib.blake2b(" ".join(text.lower().split()).encode(), digest_size=8).digest()


def is_valid_doc(text: str, per_mille: int) -> bool:
    return int.from_bytes(digest(text)[:4], "little") % 1000 < per_mille


_JUNK = re.compile(r"<[a-zA-Z/]|https?://|www\.|@|\(Ed\.\)|\bpp\.|\bet al\.|Find this resource")


def is_clean_prose(paragraph: str) -> bool:
    """Plain prose only: ends like a sentence, mostly letters, no markup, links or citations."""
    if not paragraph.endswith((".", "!", "?", '"', "”", "’")) or _JUNK.search(paragraph):
        return False
    letters = sum(c.isalpha() or c.isspace() for c in paragraph)
    digits = sum(c.isdigit() for c in paragraph)
    return letters / len(paragraph) >= 0.9 and digits / len(paragraph) < 0.03


def filter_paragraphs(job: tuple[str, float, int]) -> tuple[list[str], int]:
    """Librarian: keep clean prose readable at or below `fk_max`. Returns (kept, total)."""
    text, fk_max, min_words = job
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    kept = [p for p in paragraphs
            if len(words(p)) >= min_words and is_clean_prose(p) and fk_grade(p) <= fk_max]
    return kept, len(paragraphs)


def source_files(pattern: str) -> list[Path]:
    """A source path may be a glob (many shards of one dataset): its files in sorted order."""
    if any(c in pattern for c in "*?["):
        files = sorted(paths.LIBRARY.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no library files match {pattern}")
        return files
    return [paths.LIBRARY / pattern]


_NORM = re.compile(r"[^a-z0-9]+")
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def _norm(text: str) -> str:
    return _NORM.sub(" ", text.lower()).strip()


def exam_prompts() -> set[str]:
    """Every exam and validation prompt (with its context), normalised; empty if the exams are not built."""
    from cerbeto.exams.school import EXAMS
    out = set()
    for f in EXAMS.glob("*/*.jsonl"):
        for line in f.read_text().splitlines():
            task = json.loads(line)
            for text in (task["question"], f"{task.get('context', '')} {task['question']}"):
                if len(n := _norm(text)) >= 12:
                    out.add(n)
    return out


def leaks(doc: str, prompts: set[str], longest: int = 4) -> bool:
    """True if a run of up to `longest` consecutive sentences of the document is an exam prompt."""
    sents = [_norm(s) for p in doc.split("\n") for s in _SENTENCES.split(p) if s.strip()]
    return any(" ".join(sents[i:i + n]) in prompts for n in range(1, longest + 1) for i in range(len(sents)))


def _fingerprint(name: str, shelf: dict) -> str:
    h = hashlib.sha256(json.dumps(shelf, sort_keys=True).encode())
    for src in shelf["sources"]:
        for key in ("path", "valid_path"):
            if key in src:
                for f in source_files(src[key]):
                    st = f.stat()
                    h.update(f"{name}:{f.relative_to(paths.LIBRARY)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()


def _docs(src: dict, shelf: dict, stats: dict, seen: set[bytes], workers: int) -> Iterator[str]:
    reader = sources.READERS[src["kind"]]
    options = {k: v for k, v in src.items() if k not in ("kind", "path", "valid_path", "license")}
    options["min_edu_score"] = shelf.get("min_edu_score", 0)
    raw = (doc for f in source_files(src["path"]) for doc in reader(f, **options))

    if shelf.get("fk_max") is None:
        for doc in raw:
            key = digest(doc)
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
            yield doc
        return

    jobs = ((doc, shelf["fk_max"], shelf.get("min_words", 0)) for doc in raw)
    with Pool(workers) as pool:
        for kept, total in pool.imap(filter_paragraphs, jobs, chunksize=64):
            stats["paragraphs_seen"] += total
            fresh = []
            for p in kept:
                key = digest(p)
                if key in seen:
                    stats["duplicates"] += 1
                else:
                    seen.add(key)
                    fresh.append(p)
            stats["paragraphs_rejected"] += total - len(fresh)
            if fresh:
                yield "\n".join(fresh)


def build_shelf(name: str, shelf: dict, per_mille: int, force: bool = False) -> dict:
    out = paths.SHELVES / name
    manifest_path = out / "manifest.json"
    fingerprint = _fingerprint(name, shelf)
    if not force and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("build_fingerprint") == fingerprint and (out / "train.jsonl").exists():
            print(f"[{name}] up to date — skipping")
            return manifest

    out.mkdir(parents=True, exist_ok=True)
    stats = {"docs": {"train": 0, "valid": 0}, "words": {"train": 0, "valid": 0},
             "duplicates": 0, "paragraphs_seen": 0, "paragraphs_rejected": 0}
    prompts = exam_prompts() if shelf.get("exclude_exam_prompts") else set()
    if shelf.get("exclude_exam_prompts"):
        if not prompts:
            raise RuntimeError(f"[{name}] exclude_exam_prompts needs the exams (cerbeto exam build)")
        stats["exam_leaks_dropped"] = 0
    seen: set[bytes] = set()
    workers = max(1, (os.cpu_count() or 2) - 1)

    def write(f, split: str, doc: str) -> None:
        if prompts and leaks(doc, prompts):
            stats["exam_leaks_dropped"] += 1
            return
        f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
        stats["docs"][split] += 1
        stats["words"][split] += len(doc.split())

    with (out / "train.jsonl").open("w") as ftrain, (out / "valid.jsonl").open("w") as fvalid:
        for src in shelf["sources"]:
            print(f"[{name}] reading {src['path']}")
            for doc in _docs(src, shelf, stats, seen, workers):
                if "valid_path" in src:
                    write(ftrain, "train", doc)
                elif is_valid_doc(doc, per_mille):
                    write(fvalid, "valid", doc)
                else:
                    write(ftrain, "train", doc)
            if "valid_path" in src:
                valid_src = {**src, "path": src["valid_path"]}
                for doc in _docs(valid_src, shelf, stats, seen, workers):
                    write(fvalid, "valid", doc)

    manifest = {
        "shelf": name,
        "stage": shelf["stage"],
        "description": shelf["description"],
        "fk_max": shelf.get("fk_max"),
        "sources": [{"path": s["path"], "license": s["license"]} for s in shelf["sources"]],
        "license": "; ".join(sorted({s["license"] for s in shelf["sources"]})),
        "build_fingerprint": fingerprint,
        **stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[{name}] docs train/valid: {stats['docs']['train']}/{stats['docs']['valid']}, "
          f"words: {stats['words']['train']:,}")
    return manifest


def build_all(only: list[str] | None = None, force: bool = False) -> None:
    cfg = load_config()
    for name, shelf in cfg["shelves"].items():
        if not only or name in only:
            build_shelf(name, shelf, cfg["valid_per_mille"], force)
