"""Spec 0010: readers of the new sources, glob sources, the exam-leak filter of shelves."""
import pyarrow as pa
import pyarrow.parquet as pq

from cerbeto.library import shelves, sources


def test_soda_maps_two_speakers_to_roles(tmp_path):
    f = tmp_path / "soda.parquet"
    pq.write_table(pa.table({"dialogue": [["Hi Ann!", "Hello!", "How are you?"], ["Only one turn."]],
                             "speakers": [["Bob", "Ann", "Bob"], ["Bob"]]}), f)
    docs = list(sources.soda_parquet(f))
    assert docs == ["<|human|> Hi Ann!\n<|student|> Hello!\n<|human|> How are you?"]


def test_cosmopedia_keeps_only_given_audiences(tmp_path):
    f = tmp_path / "c.parquet"
    pq.write_table(pa.table({"text": ["a", "b", "c"], "audience": ["young_children", "general", "young_children"]}), f)
    assert list(sources.cosmopedia_parquet(f, audiences=["young_children"])) == ["a", "c"]


def test_glob_source_lists_shards_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(shelves.paths, "LIBRARY", tmp_path)
    for name in ("x_002.parquet", "x_000.parquet", "x_001.parquet"):
        (tmp_path / name).write_text("")
    assert [f.name for f in shelves.source_files("x_*.parquet")] == ["x_000.parquet", "x_001.parquet", "x_002.parquet"]
    assert shelves.source_files("x_000.parquet") == [tmp_path / "x_000.parquet"]


def test_leak_finds_an_exam_prompt_inside_a_document():
    prompts = {shelves._norm("Anna has a dog. What is the name of Anna's dog?")}
    assert shelves.leaks("Story time.\nAnna has a dog. What is the name of Anna's dog? Nobody knew.", prompts)
    assert not shelves.leaks("Anna has a dog named Rex. She loves him.", prompts)


def test_curly_quotes_end_clean_prose():
    assert shelves.is_clean_prose("Mia looked at her little brother and said that they would all go to the park today.”")
