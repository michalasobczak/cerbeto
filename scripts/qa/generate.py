"""Spec 0019, step 2: fact questions from shelf passages, on one GPU, with a local teacher (vLLM, Qwen3-8B-AWQ, no thinking).
Four passes per chunk of passages (--stage all): the teacher rates each passage for textbook knowledge (0-5; below
--min-rate no questions), writes question-answer pairs grounded in the rated passages, scores each pair without the
passage (0 unclear or false ... 5 a basic fact; below --min-score dropped), and finally answers each question alone,
closed book: the pair stays only if that answer agrees with its own. A fact the teacher knows without the passage is
general knowledge, and a question that leans on its passage cannot be answered, so stories, trivia and detail fall out. Only pairs that pass both, and the rule checks below, are kept.
Resumable: passages already in the output are skipped. Standalone (no cerbeto import), in its own environment. The full run uses the stages on all cards in turn:

    CUDA_VISIBLE_DEVICES=0 ~/qa_env/bin/python scripts/qa/generate.py --stage rate --model Qwen/Qwen3-14B-AWQ \
        --passages passages.jsonl --shard 0/4 --out rated_0.jsonl
    CUDA_VISIBLE_DEVICES=0 ~/qa_env/bin/python scripts/qa/generate.py --stage gen --rated rated_*.jsonl \
        --passages passages.jsonl --shard 0/4 --out cand_0.jsonl
    (then judge.py on the candidates; ~/qa_env: uv-managed Python 3.12 with vllm==0.34.0, no system packages)
"""
import argparse, json, re, sys, time
from collections import Counter
from pathlib import Path

RATE = """How much does this passage teach well-known, lasting general knowledge of the kind found in a school textbook or a
children's encyclopedia (science, nature, animals, the body, space, the Earth, countries, history, famous people,
inventions, language, how things work)?
0 = none: a story or plot, an advertisement, advice, a recipe, local news, a minor person, place, band or company.
3 = some textbook facts among other things.
5 = rich in textbook facts.
Answer with one digit only.

Passage:
\"\"\"
{text}
\"\"\""""

GEN = """Read the passage below. Write {n} question-answer pairs that teach the general facts in it, the kind a school
textbook or a children's encyclopedia teaches.

Rules:
- Closed book: every question must make complete sense on its own to someone who has never seen the passage. Name the
  thing you ask about. Never mention "the passage", "the text", "the author", "this lesson", "the story" or "the article".
- Ask only about well-known, lasting facts: science, nature, animals, the body, space, the Earth, countries, history,
  famous people, inventions, language, numbers, how things work.
- Never ask about: plots or characters of movies, books, games, operas or TV shows; minor people, bands, companies, teams,
  towns or events; dates of releases or awards; exercises, word problems or puzzles; advice, opinions or marketing.
- The answer must be stated in the passage or follow directly from it, and it must be correct.
- Answer in one short, complete, simple sentence of at most 20 words that repeats the subject, for example
  Q: What gas do plants take in from the air? A: Plants take in carbon dioxide from the air.
- Use plain English a ten-year-old understands.
- Mix the kinds of questions: what, who, where, when, why, how, how many, which of two (name both options in the question),
  and yes/no (the answer starts with "Yes" or "No").
- Fewer good pairs are better than many weak ones. If the passage has no such facts, return an empty list.

Return only JSON: {{"qa": [{{"q": "...", "a": "...", "type": "what|who|where|when|why|how|how_many|which|yes_no"}}]}}

Passage:
\"\"\"
{text}
\"\"\""""

CHECK = """You review quiz questions for school students. You see only the questions, not where they came from. Score every
numbered pair from 0 to 5:
0 = unclear on its own (it says "he", "the incident", "the group", "the hotel" without naming it), or the answer is wrong
    or does not answer the question, or you are not sure it is true.
1 = trivia nobody needs: the plot or a character of a movie, book, game or show; a minor person, place, band, company or
    event; advice, opinion, marketing or a puzzle.
2-3 = true and clear, but a narrow detail.
4 = a solid fact a school textbook would teach.
5 = a basic fact every educated person should know.

Return only JSON: {{"scores": [4, 0, ...]}} with exactly {m} integers, one per pair in order.

Pairs:
{pairs}"""

BAD = re.compile(r"\b(passage|text|author|article|chapter|section|lesson|story|paragraph|above|below|mentioned|according to|"
                 r"this (book|course|unit|module|essay|page)|the writer|the reader|in this)\b", re.I)

ASK = "Answer in one short, complete sentence. If you do not know, say \"I don't know\".\n\nQuestion: {q}"
STOP = set("a an the is are was were be been of in on at to for and or by with from as that this it its they their there "
           "what which who whom where when why how do does did can could has have had not no yes".split())
# words that only restate the question ("X is called Y", "X was led by Y"): they say nothing about the answer itself
GENERIC = set("called named known led made used found located means mean consists part type kind example also usually often "
              "people person many most some one main important called".split())


def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w


def content(s: str, drop: set) -> set:
    return {stem(w) for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and w not in GENERIC and len(w) > 1} - drop


SAME = """Do these two answers to the question state the same fact? Small differences in wording or detail do not matter;
a different name, place, number or claim does. Reply with yes or no only.

Question: {q}
Answer 1: {a}
Answer 2: {b}"""


def numbers(s: str) -> set:
    return set(re.findall(r"\d+(?:[.,]\d+)?", s))


def agrees(q: str, ref: str, ans: str) -> bool | None:
    """Does the closed-book answer say what the reference says? True / False, or None when the words do not decide it
    (then the teacher is asked with SAME). Numbers in the reference must all come back; for yes/no the same yes or no;
    otherwise at least half of the reference's own words (not in the question, not mere restating) must come back."""
    a = ans.lower().lstrip()
    if re.search(r"don't know|do not know|not sure|unknown|cannot", a):
        return False
    if not numbers(ref) <= numbers(ans):
        return False
    if re.match(r"(yes|no)\b", ref.lower()):
        return a.startswith("no") if ref.lower().startswith("no") else a.startswith("yes")
    qw = content(q, set())
    key = {w for w in content(ref, qw) if not any(w[:5] == x[:5] for x in qw)}
    if not key:
        return None
    got = key & content(ans, set())
    return True if got and len(got) / len(key) >= 0.5 else None


TYPES = {"what", "who", "where", "when", "why", "how", "how_many", "which", "yes_no"}


def parse(s: str) -> dict | None:
    s = s.strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None


def rule_ok(p: dict) -> bool:
    q, a = str(p.get("q", "")).strip(), str(p.get("a", "")).strip()
    if not q.endswith("?") or not 3 <= len(q.split()) <= 30 or not 2 <= len(a.split()) <= 30:
        return False
    if BAD.search(q) or BAD.search(a) or sum(ord(c) < 128 for c in q + a) < len(q + a) * 0.99:
        return False
    if re.search(r"\b(i don't know|not sure|cannot be determined|not stated|unknown)\b", a, re.I):
        return False
    if p.get("type") == "yes_no" and not re.match(r"(yes|no)\b", a, re.I):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rate", "gen", "all"], default="all",
                    help="rate: passage ratings only; gen: pairs from passages already rated (--rated), no check; all: all three")
    ap.add_argument("--passages", type=Path, required=True)
    ap.add_argument("--rated", type=Path, nargs="*", default=[], help="gen: files written by --stage rate")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: this GPU takes every n-th passage from the i-th")
    ap.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--min-rate", type=int, default=3, help="passages rated below this get no questions")
    ap.add_argument("--min-score", type=int, default=4, help="all: pairs scored below this by the check are dropped")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many passages (tests)")
    ap.add_argument("--max-len", type=int, default=3072, help="1536 for --stage rate with the 14B")
    ap.add_argument("--util", type=float, default=0.93)
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--eager", action="store_true", help="no CUDA graphs (frees ~0.6 GB)")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    rows = [json.loads(x) for k, x in enumerate(a.passages.open()) if k % n == i]
    rates = {}
    for f in a.rated:
        for x in f.open():
            r = json.loads(x); rates[r["pid"]] = r["rate"]
    if a.stage == "gen":
        rows = [r for r in rows if rates.get(r["pid"], -1) >= a.min_rate]
    done_file = a.out.with_suffix(".done")
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    rows = [r for r in rows if r["pid"] not in done]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{a.stage} shard {a.shard}: {len(rows)} passages to do, {len(done)} done", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(a.model, max_model_len=a.max_len, gpu_memory_utilization=a.util, max_num_seqs=a.max_seqs, enforce_eager=a.eager, seed=19)
    gen_sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=900, seed=19)
    chk_sp = SamplingParams(temperature=0.0, max_tokens=160)
    rate_sp = SamplingParams(temperature=0.0, max_tokens=3)
    ask_sp = SamplingParams(temperature=0.0, max_tokens=60)
    kw = {"chat_template_kwargs": {"enable_thinking": False}, "use_tqdm": False}
    hist, scores = Counter(), Counter()
    stats = Counter()
    for c in range(0, len(rows), a.chunk):
        t0, part = time.time(), rows[c:c + a.chunk]
        if a.stage == "gen":
            good, rmap = part, rates
        else:
            outs = llm.chat([[{"role": "user", "content": RATE.format(text=r["text"])}] for r in part], rate_sp, **kw)
            rated = [(r, int(m.group())) for r, o in zip(part, outs) if (m := re.search(r"\d", o.outputs[0].text))]
            hist.update(v for _, v in rated)
            if a.stage == "rate":
                with a.out.open("a") as f, done_file.open("a") as df:
                    f.write("".join(json.dumps({"pid": r["pid"], "rate": v}) + "\n" for r, v in rated))
                    df.write("\n".join(r["pid"] for r in part) + "\n")
                stats["passages"] += len(part)
                dt = time.time() - t0
                print(f"{time.strftime('%H:%M:%S')} {stats['passages']} rated · {len(part) / dt:.1f} passages/s · rates {dict(sorted(hist.items()))}", flush=True)
                continue
            good = [r for r, v in rated if v >= a.min_rate]
            rmap = {r["pid"]: v for r, v in rated}
        stats["rated ok"] += len(good)
        prompts = [[{"role": "user", "content": GEN.format(n=max(3, min(8, len(r["text"].split()) // 40)), text=r["text"])}] for r in good]
        outs = llm.chat(prompts, gen_sp, **kw) if good else []
        cand = []
        for r, o in zip(good, outs):
            d = parse(o.outputs[0].text)
            qa = [p for p in (d or {}).get("qa", []) if isinstance(p, dict)]
            stats["written"] += len(qa)
            ok = [p for p in qa if rule_ok(p)]
            stats["rules"] += len(ok)
            cand.append(ok)
        record = lambda p, r, v: json.dumps({"q": p["q"].strip(), "a": p["a"].strip(), "type": p.get("type") if p.get("type") in TYPES else "other",
                                             "shelf": r["shelf"], "pid": r["pid"], "split": r["split"], "rate": rmap.get(r["pid"]), "score": v}) + "\n"
        with a.out.open("a") as f, done_file.open("a") as df:
            if a.stage == "gen":
                for r, ok in zip(good, cand):
                    f.write("".join(record(p, r, None) for p in ok)); stats["kept"] += len(ok)
            else:
                idx = [k for k, ok in enumerate(cand) if ok]
                checks = [[{"role": "user", "content": CHECK.format(m=len(cand[k]), pairs="\n".join(
                    f"{j + 1}. Q: {p['q']}\n   A: {p['a']}" for j, p in enumerate(cand[k])))}] for k in idx]
                passed = []
                for k, o in zip(idx, llm.chat(checks, chk_sp, **kw) if checks else []):
                    sc = (parse(o.outputs[0].text) or {}).get("scores")
                    if not isinstance(sc, list) or len(sc) != len(cand[k]):
                        continue
                    scores.update(v for v in sc if isinstance(v, int))
                    passed += [(p, good[k], v) for p, v in zip(cand[k], sc) if isinstance(v, int) and v >= a.min_score]
                stats["scored ok"] += len(passed)
                outs = llm.chat([[{"role": "user", "content": ASK.format(q=p["q"])}] for p, _, _ in passed], ask_sp, **kw) if passed else []
                closed = [o.outputs[0].text.strip() for o in outs]
                verdict = [agrees(p["q"], p["a"], b) for (p, _, _), b in zip(passed, closed)]
                ask = [k for k, v in enumerate(verdict) if v is None]
                if ask:
                    same = llm.chat([[{"role": "user", "content": SAME.format(q=passed[k][0]["q"], a=passed[k][0]["a"], b=closed[k])}]
                                     for k in ask], rate_sp, **kw)
                    for k, o in zip(ask, same):
                        verdict[k] = o.outputs[0].text.strip().lower().startswith("yes")
                for (p, r, v), ok in zip(passed, verdict):
                    if ok:
                        f.write(record(p, r, v)); stats["kept"] += 1
            df.write("\n".join(r["pid"] for r in part) + "\n")
        stats["passages"] += len(part)
        dt = time.time() - t0
        print(f"{time.strftime('%H:%M:%S')} {dict(stats)} · {len(part) / dt:.1f} passages/s · rates {dict(sorted(hist.items()))} · scores {dict(sorted(scores.items()))}", flush=True)
    print("finished", dict(stats), flush=True)


if __name__ == "__main__":
    sys.exit(main())
