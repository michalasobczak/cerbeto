"""Flesch-Kincaid Grade Level — the librarian's measure of text difficulty."""
import re

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")
_VOWEL_GROUP = re.compile(r"[aeiouy]+")


def count_syllables(word: str) -> int:
    w = word.lower()
    if len(w) <= 3:
        return 1
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")):
        w = w[:-1]
    if w.endswith(("es", "ed")) and not w.endswith(("ted", "ded", "ses", "zes", "ces", "ges")):
        w = w[:-2]
    return max(1, len(_VOWEL_GROUP.findall(w)))


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def fk_grade(text: str) -> float:
    ws = words(text)
    if not ws:
        return 0.0
    sentences = max(1, len(_SENTENCE_END.findall(text)))
    syllables = sum(count_syllables(w) for w in ws)
    return 0.39 * len(ws) / sentences + 11.8 * syllables / len(ws) - 15.59
