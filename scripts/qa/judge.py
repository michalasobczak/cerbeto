"""Spec 0020: the closed-book checks of question-answer pairs by the teacher (Qwen3-8B-AWQ; with --think it reasons before
it scores; the 14B does not fit a 12 GB card with room for a batch), on one GPU. It sees only the pairs, ten per prompt, never the passage, and scores each 0-5 (see CHECK in generate.py); the
score is added to every pair as "judge". --mode known (default): the teacher answers each question alone, with no passage,
and the pair is marked "known" when that answer agrees with the pair's answer (most of its own words come back): a fact a
general model knows is general knowledge, and a question that leans on its passage cannot be answered. Resumable.

    CUDA_VISIBLE_DEVICES=0 ~/qa_env/bin/python scripts/qa/judge.py --pairs cand_0.jsonl --out judged_0.jsonl
"""
import argparse, json, re, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate import ASK, CHECK, SAME, agrees, parse  # noqa: E402

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    ap.add_argument("--per-prompt", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=4000, help="pairs per vLLM call")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--util", type=float, default=0.93)
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--mode", choices=["known", "score"], default="known",
                    help="known: the teacher answers closed book and the pair stays if it agrees; score: 0-5 scores (CHECK)")
    ap.add_argument("--think", action="store_true", help="let the teacher think before it scores")
    ap.add_argument("--eager", action="store_true", help="no CUDA graphs (frees ~0.6 GB)")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    rows = [json.loads(x) for k, x in enumerate(a.pairs.open()) if k % n == i]
    done = sum(1 for _ in a.out.open()) if a.out.exists() else 0
    rows = rows[done:]
    print(f"shard {a.shard}: {len(rows)} pairs to judge, {done} done", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, max_model_len=a.max_len, gpu_memory_utilization=a.util, max_num_seqs=a.max_seqs, enforce_eager=a.eager, seed=19)
    sp = SamplingParams(temperature=0.6 if a.think else 0.0, top_p=0.95, max_tokens=3000 if a.think else 80, seed=19)
    kw = {"chat_template_kwargs": {"enable_thinking": a.think}, "use_tqdm": False}
    hist = Counter()
    for c in range(0, len(rows), a.chunk):
        t0, part = time.time(), rows[c:c + a.chunk]
        if a.mode == "known":
            outs = llm.chat([[{"role": "user", "content": ASK.format(q=p["q"])}] for p in part],
                            SamplingParams(temperature=0.0, max_tokens=60), **kw)
            closed = [o.outputs[0].text.split("</think>")[-1].strip() for o in outs]
            verdict = [agrees(p["q"], p["a"], b) for p, b in zip(part, closed)]
            ask = [k for k, v in enumerate(verdict) if v is None]
            if ask:
                same = llm.chat([[{"role": "user", "content": SAME.format(q=part[k]["q"], a=part[k]["a"], b=closed[k])}] for k in ask],
                                SamplingParams(temperature=0.0, max_tokens=3), **kw)
                for k, o in zip(ask, same):
                    verdict[k] = o.outputs[0].text.strip().lower().startswith("yes")
            with a.out.open("a") as f:
                for p, b, ok in zip(part, closed, verdict):
                    hist[ok] += 1
                    f.write(json.dumps({**p, "known": ok, "closed_book": b}) + "\n")
            dt = time.time() - t0
            print(f"{time.strftime('%H:%M:%S')} {c + len(part)} pairs · {len(part) / dt:.0f} pairs/s · agrees {dict(hist)}", flush=True)
            continue
        groups = [part[j:j + a.per_prompt] for j in range(0, len(part), a.per_prompt)]
        prompts = [[{"role": "user", "content": CHECK.format(m=len(g), pairs="\n".join(
            f"{j + 1}. Q: {p['q']}\n   A: {p['a']}" for j, p in enumerate(g)))}] for g in groups]
        outs = llm.chat(prompts, sp, **kw)
        with a.out.open("a") as f:
            for g, o in zip(groups, outs):
                sc = (parse(o.outputs[0].text.split("</think>")[-1]) or {}).get("scores")
                ok = isinstance(sc, list) and len(sc) == len(g)
                for j, p in enumerate(g):
                    v = sc[j] if ok and isinstance(sc[j], int) else -1
                    hist[v] += 1
                    f.write(json.dumps({**p, "judge": v}) + "\n")
        dt = time.time() - t0
        print(f"{time.strftime('%H:%M:%S')} {c + len(part)} pairs · {len(part) / dt:.0f} pairs/s · scores {dict(sorted(hist.items()))}", flush=True)
    print("finished", dict(sorted(hist.items())), flush=True)


if __name__ == "__main__":
    main()
