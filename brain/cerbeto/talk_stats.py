"""Inference statistics for `cerbeto chat` (/stats): what the student did while it wrote its last reply.

One extra forward pass over the whole exchange (context + reply) keeps every attention matrix for the rows that
produced the reply. From it: how sure the student was of each token (probability, entropy, alternatives) and, per
block, where its attention went — the first token (the "sink"), the earlier conversation, the human's last message
or its own reply so far — and how spread out it was.
"""
import math

import torch

from cerbeto.model.student import Attention

BAR = "▏▎▍▌▋▊▉█"


def bar(x: float, width: int = 10) -> str:
    """A horizontal bar for a share 0…1."""
    full = max(0.0, min(1.0, x)) * width
    whole, part = int(full), int((full - int(full)) * len(BAR))
    return (BAR[-1] * whole + (BAR[part - 1] if part else "")).ljust(width)


def layer_names(cfg, k: int) -> list[str]:
    """One name per attention call, in order: prelude, the core K times, coda."""
    names = [f"prelude {i + 1}" for i in range(cfg.n_prelude)]
    for p in range(k):
        names += [f"core {i + 1}" + (f" · pass {p + 1}" if k > 1 else "") for i in range(cfg.n_core)]
    return names + [f"coda {i + 1}" for i in range(cfg.n_coda)]


@torch.no_grad()
def analyse(model, ids: list[int], n_hist: int, n_ctx: int, k: int, device) -> dict:
    """`ids` = context + reply; the context is `ids[:n_ctx]`, of which `ids[:n_hist]` is the earlier conversation."""
    cut = max(0, len(ids) - model.cfg.max_seq_len)               # the model sees only the last max_seq_len tokens
    ids, n_hist, n_ctx = ids[cut:], max(1, n_hist - cut), n_ctx - cut
    rows = slice(n_ctx - 1, len(ids) - 1)            # the positions whose output was a reply token
    Attention.capture, Attention.capture_rows = [], rows
    try:
        logits = model(torch.tensor([ids], device=device), k)[0, rows].float()
        att = Attention.capture
    finally:
        Attention.capture, Attention.capture_rows = None, slice(None)
    probs = logits.softmax(-1)
    reply = torch.tensor(ids[n_ctx:])
    p_chosen = probs.cpu()[torch.arange(len(reply)), reply]
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1).cpu() / math.log(2)
    top = probs.topk(3, -1)

    pos = torch.arange(n_ctx - 1, len(ids) - 1).float()          # query positions; a row can see pos + 1 tokens
    regions = {"sink": slice(0, 1), "earlier": slice(1, n_hist), "question": slice(n_hist, n_ctx),
               "own reply": slice(n_ctx, len(ids))}
    layers = []
    for a in att:                                                   # (1, heads, rows, T)
        a = a[0]
        h = -(a * a.clamp_min(1e-12).log()).sum(-1)                 # (heads, rows)
        spread = (h / torch.log(pos + 1).clamp_min(1e-6)).mean().item()
        shares = {name: a[..., r].sum(-1).mean().item() for name, r in regions.items()}
        focused = int((a.max(-1).values.mean(-1) > 0.5).sum())      # heads that mostly look at one token
        layers.append({"spread": spread, "focused_heads": focused, "heads": a.shape[0], **shares})

    mean = torch.stack([a[0].mean(0) for a in att]).mean(0)         # (rows, T): over blocks and heads
    looked = mean.clone()
    looked[:, 0] = 0                                                # the sink says nothing about content
    overall = looked.mean(0)
    most = overall.topk(min(8, len(ids) - 1))
    return {"ids": ids, "n_hist": n_hist, "n_ctx": n_ctx, "reply": reply.tolist(), "p": p_chosen.tolist(),
            "entropy": entropy.tolist(), "top": list(zip(top.indices.cpu().tolist(), top.values.cpu().tolist())),
            "looks_at": looked.argmax(-1).tolist(), "most": list(zip(most.indices.tolist(), most.values.tolist())),
            "layers": layers}


def render(stats: dict, tok, names: list[str], timing: dict, model, k: int) -> str:
    ids, n_ctx, n_hist = stats["ids"], stats["n_ctx"], stats["n_hist"]
    piece = lambda i: repr(tok.decode([i], skip_special_tokens=False))[1:-1] or "∅"    # noqa: E731
    region = lambda j: "sink" if j == 0 else "earlier" if j < n_hist else "question" if j < n_ctx else "reply"  # noqa: E731
    n = len(stats["reply"])
    out = []

    s = timing["seconds"]
    nll = -sum(math.log(max(p, 1e-12)) for p in stats["p"]) / max(1, n)
    out.append("── inference ──")
    out.append(f"  {model.num_params():,} parameters · K = {k} · {len(names)} attention calls per token · {timing['device']}")
    out.append(f"  context {n_ctx} tokens ({n_ctx / model.cfg.max_seq_len:.0%} of {model.cfg.max_seq_len}; earlier "
               f"conversation {max(0, n_hist - 1)}, question {n_ctx - n_hist}) · reply {n} tokens")
    out.append(f"  {s:.2f} s · {n / s if s else 0:.1f} tokens/s · {1000 * s / max(1, n):.0f} ms per token "
               f"(no KV cache: each token re-reads the whole context) · temperature {timing['temperature']}")
    out.append(f"  reply perplexity {math.exp(nll):.2f} · mean probability of the chosen token "
               f"{sum(stats['p']) / max(1, n):.0%}")

    out.append("── tokens: how sure, what else, where it looked ──")
    out.append(f"  {'token':<14}{'p':>5}  {'':10} {'bits':>5}  {'alternatives':<34} looks at")
    for i, t in enumerate(stats["reply"]):
        alt = ", ".join(f"{piece(a)} {q:.0%}" for a, q in zip(*stats["top"][i]) if a != t)[:34]
        j = stats["looks_at"][i]
        out.append(f"  {piece(t)[:13]:<14}{stats['p'][i]:>5.0%}  {bar(stats['p'][i])} {stats['entropy'][i]:>5.2f}  "
                   f"{alt:<34} {piece(ids[j])[:12]} ({region(j)})")

    out.append("── attention per block (mean over heads and reply tokens) ──")
    out.append(f"  {'block':<18}{'spread':>7}  {'sink':>5} {'earlier':>8} {'question':>9} {'own':>5}  focused heads")
    for name, l in zip(names, stats["layers"]):
        out.append(f"  {name:<18}{l['spread']:>7.2f}  {l['sink']:>5.0%} {l['earlier']:>8.0%} {l['question']:>9.0%} "
                   f"{l['own reply']:>5.0%}  {l['focused_heads']}/{l['heads']}")
    out.append("  spread: attention entropy / its maximum (0 = one token, 1 = evenly over all it can see)")

    out.append("── most attended context tokens (all blocks, without the sink) ──")
    for j, w in stats["most"]:
        out.append(f"  {bar(w / stats['most'][0][1])} {w:.1%}  #{j:<4} {piece(ids[j])[:20]:<20} ({region(j)})")
    return "\n".join(out)
