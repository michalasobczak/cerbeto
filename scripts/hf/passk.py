"""Is the knowledge in the model but not reached? bench1000's questions answered once greedily (pass@1, as bench1000)
and K times sampled: pass@K counts a question right if any sample is right. A big gap between them means the facts are
there and RL on the checkable answers could pull them out; a small gap means they are not there (a bigger model).

    CUDA_VISIBLE_DEVICES=0 BENCH_DEVICE=cuda uv run --with huggingface_hub python ../scripts/hf/passk.py checkpoints/final/facts2/final [K] [T]
"""
import importlib.util, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.argv, args = sys.argv[:1], sys.argv[1:]          # bench1000 runs its loop on import: give it no models
spec = importlib.util.spec_from_file_location("bench1000", HERE / "bench1000.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

name, K, T = args[0], int(args[1]) if len(args) > 1 else 8, float(args[2]) if len(args) > 2 else 0.7
dev = os.environ.get("BENCH_DEVICE", "cpu")
torch.manual_seed(19)
t = b.Talker(b.checkpoint.load(b.fetch(name), dev), torch.device(dev), k=1, temperature=0, max_new_tokens=40)
t0, rows = time.time(), []
for item in b.QS:
    t.temperature = 0
    greedy = b.right(item, b.ask(t, item))
    t.temperature = T
    samples = [b.right(item, b.ask(t, item)) for _ in range(K)]
    rows.append({"part": item["part"], "greedy": greedy, "samples": sum(samples), "any": any(samples)})
(b.OUT / ("passk_" + name.replace("/", "_") + ".json")).write_text(json.dumps(rows))
print(f"{name}: K={K}, T={T}, {time.time() - t0:.0f} s")
print(f"{'':11}{'pass@1':>10}{'sample avg':>12}{f'pass@{K}':>10}{'of':>6}")
by = defaultdict(list)
for r in rows:
    by[r["part"]].append(r); by["ALL"].append(r)
for part in [*b.PARTS, "ALL"]:
    rs = by[part]
    print(f"{part:11}{sum(r['greedy'] for r in rs):>10}{sum(r['samples'] for r in rs) / K:>12.1f}{sum(r['any'] for r in rs):>10}{len(rs):>6}")
