"""Spec 0014: on 2,000 documents of the first FineWeb-Edu shard, how much text the grade 7-8 band adds against <= 7."""
import itertools

from cerbeto.library import sources
from cerbeto.library.shelves import filter_paragraphs, load_config, source_files

sh = load_config()["shelves"]["s3_edu_8"]
files = source_files(sh["sources"][0]["path"])
docs = list(itertools.islice(sources.READERS["fineweb_parquet"](files[0], min_edu_score=3), 2000))
band = [filter_paragraphs((d, 8.0, 20, 7.0, 7.0)) for d in docs]
easy = [filter_paragraphs((d, 7.0, 20)) for d in docs]
cb = sum(len(" ".join(k)) for k, _, _ in band)
ce = sum(len(" ".join(k)) for k, _, _ in easy)
print(f"{len(files)} shards; paragraphs 7-8: {sum(len(k) for k, _, _ in band)}, <= 7: {sum(len(k) for k, _, _ in easy)}; "
      f"chars 7-8 / <= 7 = {cb / max(1, ce):.2f} (so s3_edu_8 ~ {cb / max(1, ce) * 534:.0f}M tokens at 8k)")
