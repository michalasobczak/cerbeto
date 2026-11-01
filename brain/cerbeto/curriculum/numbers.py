"""Numbers in words, both ways (0–9999): "twenty-one" ↔ 21."""
import re

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
VALUES = {w: i for i, w in enumerate(ONES)} | {w: 10 * i for i, w in enumerate(TENS) if w}


def to_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + to_words(rest))
    rest = n % 1000
    return to_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + to_words(rest))


_TOKEN = re.compile(r"\d[\d,]*|[a-z]+")


def numbers_in(text: str) -> list[int]:
    """All numbers in a text, in order, written in digits ("21", "1,000") or words ("twenty-one", "a hundred")."""
    toks = _TOKEN.findall(text.lower().replace("-", " "))
    found, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t[0].isdigit():
            found.append(int(t.replace(",", "")))
            i += 1
            continue
        if t not in VALUES and not (t == "a" and i + 1 < len(toks) and toks[i + 1] in ("hundred", "thousand")):
            i += 1
            continue
        total, current, j = 0, 0, i
        while j < len(toks):
            w = toks[j]
            if w in VALUES and not (current % 100 and current % 10 == 0 and VALUES[w] >= 10) \
                    and not (current % 10 and VALUES[w] < 100 and current % 100 >= 1 and w in ONES and current >= 1
                             and current % 100 < 20 and j > i and toks[j - 1] not in ("hundred", "and")):
                current += VALUES[w]
            elif w == "a" and j == i:
                current += 1
            elif w == "hundred":
                current = max(current, 1) * 100
            elif w == "thousand":
                total += max(current, 1) * 1000
                current = 0
            elif w == "and" and j + 1 < len(toks) and toks[j + 1] in VALUES and j > i \
                    and toks[j - 1] in ("hundred", "thousand"):
                pass
            else:
                break
            j += 1
        found.append(total + current)
        i = max(j, i + 1)
    return found
