"""Answer normalization and grading by code (spec 0004, gates G3, G6, N2).

Rules:
- the graded text is what follows `<|answer|>` when the student wrote it, otherwise the whole reply;
- "I don't know" is correct only on unanswerable tasks; on answerable ones it is an error (no hiding behind it);
- on an unanswerable task anything other than "I don't know" is a made-up answer — always an error;
- yes/no: the first yes- or no-word decides; numbers: the first number after the last "=", otherwise the last
  number in the reply (digits or words) — "3 + 4 = 7" and "Tom has 3, then 4 more, so 7" both give 7;
- text: an accepted answer must appear as whole words (simple inflections count: roars, chirping, closed) and no
  competitor (`reject`) may appear — listing every option does not pass; a reason after "because" or "than" may
  name the other options ("Emma is the fastest because she is faster than Sam");
  phrase: an accepted phrase must appear; form (soft skills): a stance and a reason;
- number / text: echoing three or more words of the question back is not an answer.
"""
import re

from cerbeto.curriculum import Task
from cerbeto.curriculum.numbers import numbers_in

SPECIAL = re.compile(r"<\|[^|]*\|>")
IDK = ["i don't know", "i do not know", "i dont know", "don't know", "dont know", "dunno", "i'm not sure",
       "i am not sure", "not sure", "no idea", "i can't know", "i cannot know", "i can not know", "you didn't tell me",
       "you did not tell me", "nobody told me", "it doesn't say", "it does not say", "the story doesn't say",
       "the story does not say"]
YES = {"yes", "yeah", "yep", "yup", "yea", "sure", "right", "correct", "true", "uhhuh", "mhm"}
NO = {"no", "nope", "nah", "not", "wrong", "false", "never"}


def extract(reply: str) -> str:
    """The answer part of a reply: after the last `<|answer|>` if present; special tokens removed."""
    if "<|answer|>" in reply:
        reply = reply.rsplit("<|answer|>", 1)[1]
    return SPECIAL.sub(" ", reply)


def normalize(text: str) -> str:
    text = extract(text).lower().replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"[^a-z0-9' -]+", " ", text)
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)      # quotes, but keep don't / it's
    return re.sub(r"\s+", " ", text).strip()


def contains(text: str, phrase: str) -> bool:
    phrase = normalize(phrase)
    return bool(phrase) and re.search(rf"(?<![a-z0-9']){re.escape(phrase)}(?![a-z0-9'])", text) is not None


def contains_form(text: str, word: str) -> bool:
    """`contains`, also for simple inflections of the last word: roar → roars, chirp → chirping, close → closed."""
    phrase = normalize(word)
    stem = re.escape(phrase[:-1]) + "e?" if phrase.endswith("e") else re.escape(phrase)
    return bool(phrase) and re.search(rf"(?<![a-z0-9']){stem}(?:s|es|ing|ed|d)?(?![a-z0-9'])", text) is not None


def is_idk(reply: str) -> bool:
    text = normalize(reply)
    return any(contains(text, p) for p in IDK)


def yes_no(text: str) -> str | None:
    for word in text.replace("-", " ").split():
        if word in YES:
            return "yes"
        if word in NO or word.endswith("n't"):
            return "no"
    return None


def answer_number(reply: str) -> int | None:
    raw = extract(reply)
    if "=" in raw:
        after = numbers_in(normalize(raw.rsplit("=", 1)[1]))
        return after[0] if after else None
    found = numbers_in(normalize(raw))
    return found[-1] if found else None


def grade(task: Task, reply: str) -> bool:
    text = normalize(reply)
    if is_idk(reply):
        return task.answer_type == "unknown"
    if task.answer_type == "unknown" or not text:
        return False
    if task.answer_type in ("number", "text") and len(text.split()) >= 3 and contains(normalize(task.question), text):
        return False                   # echoing a piece of the question
    if task.answer_type == "yesno":
        return yes_no(text) == task.answer
    if task.answer_type == "number":
        return answer_number(reply) == int(task.answer)
    if task.answer_type == "text":
        accepted = task.accept or [task.answer]
        claim = re.split(r"\b(?:because|than)\b", text, maxsplit=1)[0]   # a reason may name the other options
        return any(contains_form(claim, x) for x in accepted) and not any(contains(claim, x) for x in task.reject)
    if task.answer_type == "phrase":
        return any(contains(text, x) for x in (task.accept or [task.answer]))
    if task.answer_type == "form":     # an opinion with a reason: a stance and "because …" with some content
        m = re.search(r"\bbecause\b(.*)", text)
        return m is not None and len(m.group(1).split()) >= 2
    raise ValueError(f"unknown answer type {task.answer_type}")
