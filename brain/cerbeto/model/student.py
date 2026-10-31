"""The student: a small transformer with a recurrent core ("thinking in a loop").

    tokens → embedding → prelude blocks → core blocks × K → coda blocks → norm → tied head

The core is applied K times with shared weights; K is sampled during training and can be raised at
inference so the student "thinks longer". All shapes live in `StudentConfig`, which growth operators
rewrite (see growth.py).
"""
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class StudentConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_heads: int = 8
    head_dim: int = 32
    mlp_hidden: int = 704
    n_prelude: int = 2
    n_core: int = 2
    n_coda: int = 2
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StudentConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def with_(self, **changes) -> "StudentConfig":
        return replace(self, **changes)

    @property
    def n_blocks(self) -> int:
        return self.n_prelude + self.n_core + self.n_coda


def count_params(cfg: StudentConfig) -> int:
    """Closed-form parameter count (gate G3): tied embedding, per-block norms/attention/SwiGLU, final norm."""
    d, a, f = cfg.d_model, cfg.n_heads * cfg.head_dim, cfg.mlp_hidden
    block = 2 * d + 4 * d * a + 3 * d * f
    return cfg.vocab_size * d + cfg.n_blocks * block + d


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(cfg: StudentConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=device).float() / cfg.head_dim))
    angles = torch.outer(torch.arange(cfg.max_seq_len, device=device).float(), inv)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    t = x.shape[-2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[:t], sin[:t]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        inner = cfg.n_heads * cfg.head_dim
        self.q = nn.Linear(cfg.d_model, inner, bias=False)
        self.k = nn.Linear(cfg.d_model, inner, bias=False)
        self.v = nn.Linear(cfg.d_model, inner, bias=False)
        self.o = nn.Linear(inner, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = (proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
                   for proj in (self.q, self.k, self.v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class Student(nn.Module):
    def __init__(self, cfg: StudentConfig, seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        if seed is not None:
            torch.manual_seed(seed)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.prelude = nn.ModuleList(Block(cfg) for _ in range(cfg.n_prelude))
        self.core = nn.ModuleList(Block(cfg) for _ in range(cfg.n_core))
        self.coda = nn.ModuleList(Block(cfg) for _ in range(cfg.n_coda))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        # Activation checkpointing: keep only block inputs during training and recompute the rest in the
        # backward pass. Costs ~30% speed, removes the K-fold activation memory of the recurrent core.
        self.checkpointing = False
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def blocks(self):
        yield from self.prelude
        yield from self.core
        yield from self.coda

    def forward(self, idx: torch.Tensor, k: int = 4) -> torch.Tensor:
        cos, sin = rope_tables(self.cfg, idx.device)
        recompute = self.checkpointing and self.training and torch.is_grad_enabled()

        def run(block: Block, x: torch.Tensor) -> torch.Tensor:
            return checkpoint(block, x, cos, sin, use_reentrant=False) if recompute else block(x, cos, sin)

        x = self.embed(idx)
        for block in self.prelude:
            x = run(block, x)
        for _ in range(k):
            for block in self.core:
                x = run(block, x)
        for block in self.coda:
            x = run(block, x)
        return self.norm(x) @ self.embed.weight.T

    def loss(self, idx: torch.Tensor, targets: torch.Tensor, k: int = 4) -> torch.Tensor:
        logits = self(idx, k)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, k: int = 4, temperature: float = 0.8,
                 top_k: int | None = 40, stop_token: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.max_seq_len:], k)[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat((idx, nxt), dim=1)
            if stop_token is not None and (nxt == stop_token).all():
                break
        return idx


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
