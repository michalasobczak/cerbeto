"""School exams (spec 0004): frozen, hashed pools per skill × variant, a leak check, and a batched exam run.

    cerbeto exam build                 → data/exams/<stage>/<skill>.<variant>.jsonl + hashes.txt + manifest.json
    cerbeto exam check-leak            → exam prompts vs 1M training tasks (gate G4)
    cerbeto exam run --student growing → runs/exams/<student>/<version>.json + a results table
"""
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from cerbeto import paths
from cerbeto.curriculum import EXAM_SEED_BASE, EXAM_VARIANTS, Task, generate, skills, training_stream
from cerbeto.exams.grader import grade, is_idk

EXAMS = paths.DATA / "exams"
RESULTS = paths.REPO / "runs" / "exams"


def config() -> dict:
    return yaml.safe_load((paths.CONFIG / "curriculum.yaml").read_text())


def _space(skill: str, variant: str, probe: int) -> int:
    """Number of distinct prompts among `probe` sampled tasks — an estimate of the skill's task space."""
    return len({generate(skill, variant, EXAM_SEED_BASE + 10**8 + i, unanswerable=False).hash for i in range(probe)})


def build(root: Path = EXAMS) -> dict:
    cfg = config()
    ex = cfg["exams"]
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "pools": {}}
    all_hashes: set[str] = set()
    digest = hashlib.blake2b(digest_size=12)
    for stage, names in cfg["stages"].items():
        for name in names:
            for variant in EXAM_VARIANTS:
                # small task spaces: `id` keeps most of the space for lessons; ood variants never appear in lessons
                space = _space(name, variant, ex["space_probe"])
                share = ex["small_space_share"] if variant == "id" else 1.0
                n_answerable = round(ex["pool_size"] * (1 - ex["unanswerable_share"]))
                if space < ex["space_probe"] * 0.5:
                    n_answerable = min(n_answerable, max(1, int(space * share)))
                n_unknown = round(n_answerable * ex["unanswerable_share"] / (1 - ex["unanswerable_share"]))
                size = n_answerable + n_unknown
                tasks, seen, seed = [], set(), EXAM_SEED_BASE
                for i in range(size):
                    unknown = (i + 1) * n_unknown // size > i * n_unknown // size   # spread evenly, exact count
                    for _ in range(10_000):
                        t = generate(name, variant, seed, unanswerable=unknown)
                        seed += 1
                        if t.hash not in seen:
                            break
                    else:
                        raise RuntimeError(f"cannot fill {name}.{variant}: task space too small")
                    seen.add(t.hash)
                    tasks.append(t)
                tasks = balance_yes_no(tasks, ex["unanswerable_share"])
                out = root / stage / f"{name}.{variant}.jsonl"
                out.parent.mkdir(parents=True, exist_ok=True)
                text = "".join(json.dumps(t.to_dict()) + "\n" for t in tasks)
                out.write_text(text)
                digest.update(text.encode())
                all_hashes |= seen
                manifest["pools"][f"{stage}/{name}.{variant}"] = {
                    "tasks": len(tasks), "unanswerable": sum(t.answer_type == "unknown" for t in tasks),
                    "space_estimate": space}
    (root / "hashes.txt").write_text("".join(h + "\n" for h in sorted(all_hashes)))
    manifest["exam_set_version"] = digest.hexdigest()
    manifest["tasks"] = sum(p["tasks"] for p in manifest["pools"].values())
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def balance_yes_no(tasks: list[Task], unanswerable_share: float) -> list[Task]:
    """Half "yes", half "no" (±1) among yes/no tasks, so always giving the same word scores ~50%; the larger side
    is cut and the unanswerable tasks are trimmed to keep their share."""
    yes = [t for t in tasks if t.answer_type == "yesno" and t.answer == "yes"]
    no = [t for t in tasks if t.answer_type == "yesno" and t.answer == "no"]
    if not yes and not no or abs(len(yes) - len(no)) <= 1:
        return tasks
    keep = min(len(yes), len(no)) + 1
    drop = {t.hash for t in (yes if len(yes) > len(no) else no)[keep:]}
    kept = [t for t in tasks if t.hash not in drop]
    answerable = sum(t.answer_type != "unknown" for t in kept)
    n_unknown = round(answerable * unanswerable_share / (1 - unanswerable_share))
    unknown = [t.hash for t in kept if t.answer_type == "unknown"]
    drop = set(unknown[n_unknown:])
    return [t for t in kept if t.hash not in drop]


def exam_hashes(root: Path = EXAMS) -> set[str]:
    f = root / "hashes.txt"
    return set(f.read_text().split()) if f.exists() else set()


def load_pools(root: Path = EXAMS) -> dict[tuple[str, str, str], list[Task]]:
    pools = {}
    for stage, names in config()["stages"].items():
        for name in names:
            for variant in EXAM_VARIANTS:
                f = root / stage / f"{name}.{variant}.jsonl"
                pools[(stage, name, variant)] = [Task.from_dict(json.loads(x)) for x in f.read_text().splitlines()]
    return pools


def check_leak(n: int = 1_000_000, root: Path = EXAMS) -> dict:
    """Gate G4: the training stream never produces an exam prompt. Also reports how many would collide without
    the stream's filter (how much the filter has to do)."""
    exam = exam_hashes(root)
    raw = filtered = 0
    stream = training_stream(seed=12345, exclude=exam)
    rng_names = [s.name for s in skills()]
    rng = random.Random(54321)
    for i in range(n):
        t = generate(rng.choice(rng_names), "train", rng.randrange(EXAM_SEED_BASE))
        raw += t.hash in exam
        filtered += next(stream).hash in exam
    return {"training_tasks": n, "exam_prompts": len(exam), "collisions_without_filter": raw,
            "collisions_in_training_stream": filtered}


# ── running an exam ──────────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def answer(model, prompts: list[str], k: int, device, max_new_tokens: int = 48, batch_size: int = 64) -> list[str]:
    """Greedy replies to many prompts. Prompts of equal token length are batched together, so no padding or
    attention mask is needed; generation of a batch stops when every row hit a stop token."""
    from cerbeto.tokenizer import load as load_tokenizer
    tok = load_tokenizer()
    eos = tok.token_to_id("<|eos|>")
    stops = torch.tensor([tok.token_to_id(t) for t in ("<|eos|>", "<|human|>", "<|student|>")], device=device)
    encoded = [[eos] + tok.encode(f"<|human|> {p}\n<|student|>").ids for p in prompts]
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, ids in enumerate(encoded):
        by_len[len(ids)].append(i)
    replies = [""] * len(prompts)
    model.eval()
    for length, idxs in sorted(by_len.items()):
        for b in range(0, len(idxs), batch_size):
            rows = idxs[b:b + batch_size]
            x = torch.tensor([encoded[i] for i in rows], device=device)
            done = torch.zeros(len(rows), dtype=torch.bool, device=device)
            for _ in range(max_new_tokens):
                logits = model(x[:, -model.cfg.max_seq_len:], k)[:, -1, :]
                nxt = logits.argmax(-1)
                done |= torch.isin(nxt, stops)
                x = torch.cat([x, nxt[:, None]], dim=1)
                if done.all():
                    break
            for r, i in enumerate(rows):
                out = x[r, length:].tolist()
                cut = next((j for j, t in enumerate(out) if t in stops.tolist()), len(out))
                replies[i] = tok.decode(out[:cut], skip_special_tokens=False).strip()
    return replies


def evaluate(model, k: int, device, pools=None, limit: int | None = None) -> dict:
    cfg = config()
    pools = pools or load_pools()
    flat = [(key, t) for key, tasks in pools.items() for t in (tasks[:limit] if limit else tasks)]
    replies = answer(model, [t.prompt for _, t in flat], k, device, cfg["exams"]["max_new_tokens"],
                     cfg["exams"]["batch_size"])
    cells: dict[tuple, list] = defaultdict(list)
    items = []
    for (key, t), reply in zip(flat, replies):
        ok = grade(t, reply)
        cells[key].append((t, reply, ok))
        items.append({"stage": key[0], "skill": key[1], "variant": key[2], "hash": t.hash, "prompt": t.prompt,
                      "answer": t.answer, "reply": reply, "correct": ok})
    results = {}
    for (stage, skill, variant), rows in cells.items():
        answerable = [r for r in rows if r[0].answer_type != "unknown"]
        unknown = [r for r in rows if r[0].answer_type == "unknown"]
        results.setdefault(stage, {}).setdefault(skill, {})[variant] = {
            "accuracy": sum(r[2] for r in rows) / len(rows), "n": len(rows),
            "honesty": sum(r[2] for r in unknown) / len(unknown) if unknown else None,
            "idk_on_answerable": sum(is_idk(r[1]) for r in answerable) / len(answerable) if answerable else None,
            "graded_by": rows[0][0].graded_by if answerable else "code",
        }
    for stage, sk in results.items():
        sk["_passed"] = passed(sk, cfg["passing"])
    return {"results": results, "items": items}


def passed(stage_results: dict, passing: dict) -> bool:
    """Moving to the next grade (spec 0004): every code-graded skill masters what was taught (`id`), and the stage
    transfers on average (mean `ood_hard`) with no skill below the floor (`ood_hard_floor`)."""
    code = [v for s, v in stage_results.items() if not s.startswith("_") and v["id"]["graded_by"] == "code"]
    if not code:
        return False
    hard = [v["ood_hard"]["accuracy"] for v in code]
    return (all(v["id"]["accuracy"] >= passing["id"] for v in code) and sum(hard) / len(hard) >= passing["ood_hard"]
            and min(hard) >= passing.get("ood_hard_floor", passing["ood_hard"]))


def run(student: str, version: str | None, device: str | None, limit: int | None = None) -> dict:
    from cerbeto.talk import load_student
    model, version, dev = load_student(student, version, device)
    k = yaml.safe_load((paths.CONFIG / "student.yaml").read_text())["recurrence"]["k_eval"]
    manifest = json.loads((EXAMS / "manifest.json").read_text())
    started = time.time()
    out = evaluate(model, k, dev, limit=limit)
    out.update(student=student, version=version, params=model.num_params(), k=k, limit=limit,
               exam_set_version=manifest["exam_set_version"], seconds=round(time.time() - started),
               time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    path = RESULTS / student / f"{version}{'' if limit is None else f'.limit{limit}'}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print_table(out)
    print(f"saved {path}")
    return out


def print_table(out: dict) -> None:
    print(f"{out['student']}/{out['version']} · {out['params']:,} params · K={out['k']} · exam set "
          f"{out['exam_set_version']} · {out['seconds']} s")
    print(f"{'skill':24s} {'id':>6s} {'ood_hard':>9s} {'ood_combo':>10s} {'honesty':>8s} {'idk?':>6s}")
    for stage, sk in out["results"].items():
        print(f"── {stage} {'(passed)' if sk['_passed'] else '(not passed)'}")
        for skill, v in sk.items():
            if skill.startswith("_"):
                continue
            hon = [x["honesty"] for x in v.values() if x["honesty"] is not None]
            idk = [x["idk_on_answerable"] for x in v.values() if x["idk_on_answerable"] is not None]
            soft = " (soft)" if v["id"]["graded_by"] != "code" else ""
            print(f"{skill + soft:24s} {v['id']['accuracy']:6.0%} {v['ood_hard']['accuracy']:9.0%} "
                  f"{v['ood_combo']['accuracy']:10.0%} {sum(hon) / len(hon):8.0%} {sum(idk) / len(idk):6.0%}")
