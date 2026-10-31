"""Gate G2 (spec 0001): the librarian's Flesch-Kincaid grade matches hand-computed references."""
import pytest

from cerbeto.library.readability import count_syllables, fk_grade
from cerbeto.library.shelves import filter_paragraphs


@pytest.mark.parametrize("word,syllables", [
    ("cat", 1), ("make", 1), ("played", 1), ("table", 2), ("water", 2), ("running", 2),
    ("garden", 2), ("children", 2), ("happily", 3), ("beautiful", 3),
])
def test_syllables(word, syllables):
    assert count_syllables(word) == syllables


@pytest.mark.parametrize("text,expected", [
    # 6 words, 1 sentence, 6 syllables: 0.39*6 + 11.8*1 - 15.59
    ("The cat sat on the mat.", -1.45),
    # 8 words, 1 sentence, 14 syllables: 0.39*8 + 11.8*14/8 - 15.59
    ("The children played happily in the beautiful garden.", 8.18),
    # 2 sentences of 4 words, 8 syllables: 0.39*4 + 11.8*1 - 15.59
    ("Tom has a dog. The dog is big.", -2.23),
])
def test_fk_reference_texts(text, expected):
    assert fk_grade(text) == pytest.approx(expected, abs=0.5)


def test_filter_keeps_easy_and_drops_hard_paragraphs():
    easy = "Tom has a red ball. He likes to play with it in the park every day with his friend Sam."
    hard = ("The electrophysiological characterisation of transmembrane ion channels necessitates "
            "sophisticated instrumentation and considerable methodological expertise.")
    kept, total = filter_paragraphs((f"{easy}\n{hard}", 4.0, 10))
    assert total == 2
    assert kept == [easy]


@pytest.mark.parametrize("junk", [
    "<p> tag is specially used to write paragraph. This tag will take automatic line breaks.",
    "Daniels, A. K. (1973). How free should professions be? In E. Freidson (Ed.), The professions.",
    "Visit https://example.com to read more about the big red dog and his friends today.",
    "The frequencies from old maps are listed below",
])
def test_librarian_rejects_markup_citations_and_fragments(junk):
    from cerbeto.library.shelves import is_clean_prose
    assert not is_clean_prose(junk)


def test_speech_reader_drops_annotations(tmp_path):
    from cerbeto.library.sources import speech
    raw = "*MOT:\tdo you like this room?\n*CHI:\t[laughs].\n][CHI turns around]\n= = = childes/x.cha = = =\n*CHI:\tyeah.\n"
    (tmp_path / "c.txt").write_text(raw)
    assert list(speech(tmp_path / "c.txt", chunk_lines=10)) == ["do you like this room?\nyeah."]
