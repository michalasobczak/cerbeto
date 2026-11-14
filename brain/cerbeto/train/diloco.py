"""Spec 0015 — DiLoCo: one model trained by several GPUs that sync only every H steps.

Every worker runs the `final_test.train` loop of an arm on its own data order. After H inner steps it sends the
coordinator its pseudo-gradient, base − local weights (bf16), plus how many tokens and shelf windows it read. The
coordinator averages the pseudo-gradients that arrived, takes one outer step (SGD with Nesterov momentum on the
shared weights θ) and answers every worker with that update (bf16): each applies it to its copy of θ, so all copies
stay identical, and goes on from θ with its own AdamW. A worker that is late gets the whole θ instead and its round
is dropped; a worker that joins (or restarts) gets the whole θ too.

The coordinator needs no GPU. It keeps the run as `final_test` would: runs/final/<name>/{state.json,log.jsonl} for
the panel and checkpoints/final/<name>/{latest,final}.

    python -m cerbeto.train.diloco coordinator bil2 --name d1 --tokens 1e8 --workers 4
    python -m cerbeto.train.diloco worker bil2 --coordinator 172.31.115.13 --rank 0
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import socketserver
import struct
import threading
import time

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from cerbeto.model import checkpoint
from cerbeto.train import final_test as ft

PORT = 5960
OUTER = {"lr": 0.7, "momentum": 0.9, "every": 500, "late_after": 120.0, "first_wait": 900.0}


# --- wire format: 8-byte header length, JSON header, 8-byte payload length, payload --------------------------
def send(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    h = json.dumps(header).encode()
    sock.sendall(struct.pack("!Q", len(h)) + h + struct.pack("!Q", len(payload)))
    if payload:
        sock.sendall(payload)


def _exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError("peer closed")
        got += k
    return bytes(buf)


def receive(sock: socket.socket) -> tuple[dict, bytes]:
    header = json.loads(_exact(sock, struct.unpack("!Q", _exact(sock, 8))[0]))
    n = struct.unpack("!Q", _exact(sock, 8))[0]
    return header, (_exact(sock, n) if n else b"")


def to_bytes(v: torch.Tensor) -> bytes:
    """A flat tensor as raw bytes; bf16 travels as its int16 bit pattern (numpy has no bf16)."""
    v = v.detach().contiguous().cpu()
    return (v.view(torch.int16) if v.dtype == torch.bfloat16 else v).numpy().tobytes()


def from_bytes(b: bytes, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.bfloat16:
        return torch.frombuffer(bytearray(b), dtype=torch.int16).view(torch.bfloat16)
    return torch.frombuffer(bytearray(b), dtype=dtype)


# --- the outer step -------------------------------------------------------------------------------------------
class Outer:
    """SGD with Nesterov momentum on θ, the pseudo-gradient as the gradient (as torch.optim.SGD(nesterov=True)).
    `step` returns the update rounded to bf16 and applies exactly that, so the workers' copies stay identical."""
    def __init__(self, theta: torch.Tensor, lr: float, momentum: float, buf: torch.Tensor | None = None):
        self.theta, self.lr, self.momentum = theta, lr, momentum
        self.buf = buf if buf is not None else torch.zeros_like(theta)

    def step(self, grads: list[torch.Tensor], weights: list[float] | None = None) -> torch.Tensor:
        w = torch.tensor(weights if weights else [1.0] * len(grads), dtype=torch.float32)
        w = (w / w.sum()).view(-1, 1)
        g = (torch.stack([x.float() for x in grads]) * w).sum(0)        # a shortened round counts less
        self.buf.mul_(self.momentum).add_(g)
        update = (self.lr * (g + self.momentum * self.buf)).to(torch.bfloat16)
        self.theta.sub_(update.float())
        return update


# --- coordinator ----------------------------------------------------------------------------------------------
class Coordinator:
    def __init__(self, arm: str, name: str, tokens: float, workers: int, lr: float, momentum: float,
                 late_after: float):
        self.arm, self.name, self.workers, self.late_after = arm, name, workers, late_after
        self.run, self.ckpt = ft.RUN / name, ft.CKPT / name
        self.run.mkdir(parents=True, exist_ok=True)
        latest = self.ckpt / "latest"
        state_file = self.run / "state.json"
        if latest.exists() and state_file.exists():
            self.model = checkpoint.load(latest, "cpu")
            self.state = json.loads(state_file.read_text())
            buf = torch.load(latest / "outer.pt") if (latest / "outer.pt").exists() else None
        else:
            self.model = ft.start_model(arm, "cpu")
            self.state = {"step": 0, "tokens": 0, "target_tokens": int(tokens), "params": self.model.num_params(),
                          "round": 0, "shelf_tokens": {}, "diloco": {"workers": workers, "lr": lr,
                                                                     "momentum": momentum}}
            buf = None
            state_file.write_text(json.dumps(self.state))
        self.theta = parameters_to_vector(self.model.parameters()).detach().float().clone()
        self.outer = Outer(self.theta, lr, momentum, buf)
        self.lock = threading.Condition()
        self.pending: dict[int, tuple[torch.Tensor, dict]] = {}
        self.first_arrival = None
        self.result: dict[int, bytes] = {}           # round -> update bytes, kept for the round's stragglers
        self.round_started, self.round_seconds = time.time(), None
        self.saved, self.t_last, self.tokens_last = time.time(), time.time(), self.state["tokens"]
        self.log = (self.run / "log.jsonl").open("a")

    def done(self) -> bool:
        return self.state["tokens"] >= self.state["target_tokens"]

    def public(self) -> dict:
        return {k: self.state[k] for k in ("step", "tokens", "target_tokens", "round", "shelf_tokens")}

    def hello(self) -> tuple[dict, bytes]:
        """The whole θ, and how much of the current round is left: a worker that joins or missed a round does only
        that share of the steps, so it arrives with the others (2026-11-14: a late worker stayed late for good)."""
        with self.lock:
            left = 1.0
            if self.round_seconds:
                left = min(1.0, max(0.1, 1 - (time.time() - self.round_started) / self.round_seconds))
            return ({"kind": "theta", "state": self.public(), "every": OUTER["every"], "left": round(left, 3)},
                    to_bytes(self.theta))

    def contribute(self, rank: int, rnd: int, delta: torch.Tensor, info: dict) -> tuple[dict, bytes]:
        """Blocks until round `rnd` closes; returns the update, or the whole θ when the worker missed the round."""
        with self.lock:
            if rnd != self.state["round"] or self.done():
                return self.hello()
            self.pending[rank] = (delta, info)
            self.first_arrival = self.first_arrival or time.time()
            while rnd == self.state["round"]:
                wait = self.late_after if self.state["round"] else OUTER["first_wait"]   # startup: wait for all
                late = time.time() - self.first_arrival > wait
                if len(self.pending) >= self.workers or (late and self.pending):
                    self.close_round()
                    break
                self.lock.wait(timeout=1.0)
            if rnd not in self.result or rank not in self.result[rnd][1]:
                return self.hello()
            return {"kind": "update", "state": self.public()}, self.result[rnd][0]

    def close_round(self) -> None:
        ranks = sorted(self.pending)
        infos = [self.pending[r][1] for r in ranks]
        update = self.outer.step([self.pending[r][0] for r in ranks], [max(1, i["tokens"]) for i in infos])
        st = self.state
        st["round"] += 1
        st["step"] = st["round"] * OUTER["every"]
        st["tokens"] += sum(i["tokens"] for i in infos)
        for i in infos:
            for s, n in i["shelf_tokens"].items():
                st["shelf_tokens"][s] = st["shelf_tokens"].get(s, 0) + n
        self.result = {st["round"] - 1: (to_bytes(update), set(ranks))}
        self.pending, self.first_arrival = {}, None
        now = time.time()
        self.round_seconds, self.round_started = now - self.round_started, now
        self.write_log(ranks, infos)
        if self.done():
            self.save(final=True)
        elif time.time() - self.saved > ft.CFG["save_minutes"] * 60:
            self.save()
        self.lock.notify_all()

    def write_log(self, ranks: list[int], infos: list[dict]) -> None:
        now = time.time()
        losses = [i["loss"] for i in infos if i.get("loss")]
        row = {"time": time.strftime("%H:%M:%S"), "step": self.state["step"], "tokens": self.state["tokens"],
               "round": self.state["round"], "workers": ranks,
               "tok_s": round((self.state["tokens"] - self.tokens_last) / max(1e-6, now - self.t_last))}
        for key in ("shelf_loss", "pair_loss"):
            if losses:
                row[key] = round(sum(x[key] for x in losses) / len(losses), 4)
        valid = [i["valid"] for i in infos if i.get("valid")]
        if valid:
            row["valid"] = valid[0]
        self.t_last, self.tokens_last = now, self.state["tokens"]
        self.state["tok_s_last"] = row["tok_s"]
        self.log.write(json.dumps(row) + "\n")
        self.log.flush()
        print(row, flush=True)

    def record_valid(self, h: dict) -> None:
        """A validator's result on θ of a round, as a log row of its own (the panel reads `valid` rows)."""
        with self.lock:
            row = {"time": time.strftime("%H:%M:%S"), "step": h["step"], "tokens": h["tokens"], "round": h["round"],
                   "valid": h["valid"]}
            self.state["last_valid"] = h["valid"]
            self.log.write(json.dumps(row) + "\n")
            self.log.flush()
            print(row, flush=True)

    def save(self, final: bool = False) -> None:
        vector_to_parameters(self.theta, self.model.parameters())
        tmp, latest = self.ckpt / "latest.tmp", self.ckpt / "latest"
        if tmp.exists():
            shutil.rmtree(tmp)
        parent = ft.ARM_START.get(self.arm, ft.START)
        checkpoint.save(self.model, tmp, parent=parent, meta={"final_test": self.state, "diloco": True})
        torch.save(self.outer.buf, tmp / "outer.pt")
        if latest.exists():
            shutil.rmtree(latest)
        tmp.rename(latest)
        (self.run / "state.json").write_text(json.dumps(self.state))
        self.saved = time.time()
        if final:
            out = self.ckpt / "final"
            if out.exists():
                shutil.rmtree(out)
            checkpoint.save(self.model, out, parent=parent,
                            meta={"final_test": {**self.state, "cfg": ft.CFG, "shelves": ft.ARM_SHELF_MIX.get(self.arm),
                                                 "pairs": ft.ARM_PAIR_MIX.get(self.arm)}, "diloco": True})
            print(f"{self.name}: finished · {out}", flush=True)


def serve(coord: Coordinator, host: str = "0.0.0.0", port: int = PORT) -> socketserver.ThreadingTCPServer:
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            sock = self.request
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                while True:
                    h, payload = receive(sock)
                    if h["kind"] == "hello":
                        if h["rank"] >= 0:
                            print(f"worker {h['rank']} joined from {self.client_address[0]}", flush=True)
                        send(sock, *coord.hello())
                    elif h["kind"] == "status":
                        with coord.lock:
                            send(sock, {"kind": "status", "state": coord.public()})
                    elif h["kind"] == "valid":
                        coord.record_valid(h)
                        send(sock, {"kind": "ok"})
                    elif h["kind"] == "delta":
                        send(sock, *coord.contribute(h["rank"], h["round"], from_bytes(payload, torch.bfloat16),
                                                     h["info"]))
            except ConnectionError:
                pass

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def coordinate(arm: str, name: str, tokens: float, workers: int, port: int = PORT) -> None:
    coord = Coordinator(arm, name, tokens, workers, OUTER["lr"], OUTER["momentum"], OUTER["late_after"])
    server = serve(coord, port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"{name}: coordinator for {workers} workers on :{port} · round {coord.state['round']} · "
          f"{coord.state['tokens']:,}/{coord.state['target_tokens']:,} tokens", flush=True)
    try:
        while not coord.done():
            time.sleep(5)
        time.sleep(30)                                  # the workers fetch the last answer and stop
    finally:
        server.shutdown()


# --- worker ---------------------------------------------------------------------------------------------------
class Finished(Exception):
    pass


class Link:
    """The worker's side: θ (fp32, on the GPU), the round, and the counts read since the last sync."""
    def __init__(self, host: str, port: int, rank: int, connect=socket.create_connection):
        self.rank = rank
        self.sock = connect((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.theta = None
        self.state: dict = {}

    def hello(self) -> None:
        send(self.sock, {"kind": "hello", "rank": self.rank})
        self.take(*receive(self.sock))

    def take(self, header: dict, payload: bytes, device=None) -> str:
        device = device or (self.theta.device if self.theta is not None else "cpu")
        self.state = header["state"]
        if "every" in header:
            OUTER["every"] = header["every"]
        if header["kind"] == "theta":
            self.theta = from_bytes(payload, torch.float32).to(device)
            self.next_sync = self.state["step"] + max(1, round(OUTER["every"] * header.get("left", 1.0)))
        else:
            self.theta.sub_(from_bytes(payload, torch.bfloat16).to(device).float())
            self.next_sync = self.state["step"] + OUTER["every"]
        return header["kind"]

    def __call__(self, model, opt, state: dict, shelves) -> None:
        """The `sync` hook of final_test.train: every H steps."""
        if state["step"] < self.next_sync:
            return
        local = parameters_to_vector(model.parameters()).detach().float()
        delta = (self.theta - local).to(torch.bfloat16)
        info = {"tokens": state["tokens"] - self.tokens_at_sync,
                "shelf_tokens": {s: shelves.used[s] - self.shelves_at_sync.get(s, 0) for s in shelves.used},
                "loss": state.pop("last_loss", None), "valid": state.pop("last_valid", None)}
        send(self.sock, {"kind": "delta", "rank": self.rank, "round": self.state["round"], "info": info},
             to_bytes(delta))
        self.take(*receive(self.sock))
        self.adopt(model, state, shelves)
        if self.state["tokens"] >= self.state["target_tokens"]:
            raise Finished

    def adopt(self, model, state: dict, shelves) -> None:
        """θ into the model, the global counts into the loop's state."""
        with torch.no_grad():
            vector_to_parameters(self.theta.to(next(model.parameters()).dtype), model.parameters())
        state["step"], state["tokens"] = self.state["step"], self.state["tokens"]
        for s in shelves.used:
            shelves.used[s] = int(self.state["shelf_tokens"].get(s, 0))
        self.tokens_at_sync = state["tokens"]
        self.shelves_at_sync = dict(shelves.used)


def validate(arm: str, host: str, port: int = PORT, every_rounds: int = 2, poll: float = 30.0) -> None:
    """The validator (a GPU too weak to train: 2026-11-14, a 3060 with a dead fan): every `every_rounds` rounds it
    fetches θ and computes the arm's validation, so no worker stops for it; the coordinator logs the result."""
    device = ft.pick_device()
    link = Link(host, port, rank=-1)
    model = ft.start_model(arm, "cpu").to(device)
    vocab = model.cfg.vocab_size
    valid = {s: ft.ShelfStream(s, "valid", ft.CFG["seq_len"], vocab=vocab).fixed_batches(8, 8, device, seed=1)
             for s in ft.ARM_VALID.get(arm, ("s0_speech", "s0_stories"))}
    k, last = ft.ks(arm)["k_eval"], None
    print(f"validator: {device} · {len(valid)} shelves every {every_rounds} rounds", flush=True)
    try:
        while True:
            send(link.sock, {"kind": "status"})
            st = receive(link.sock)[0]["state"]
            done = st["tokens"] >= st["target_tokens"]
            if st["round"] and (last is None or st["round"] - last >= every_rounds or (done and last != st["round"])):
                link.hello()
                with torch.no_grad():
                    vector_to_parameters(link.theta.to(device), model.parameters())
                v = ft.validation(model, valid, k)
                send(link.sock, {"kind": "valid", "round": link.state["round"], "step": link.state["step"],
                                 "tokens": link.state["tokens"], "valid": v})
                receive(link.sock)
                last = link.state["round"]
            if done and last == st["round"]:
                break
            time.sleep(poll)
    except ConnectionError:
        pass
    print("validator: the run is finished", flush=True)


def work(arm: str, host: str, rank: int, port: int = PORT, recompute: bool = True, bf16: bool = True,
         max_temp: int | None = None, own_validation: bool = True) -> None:
    device = ft.pick_device()
    link = Link(host, port, rank)
    link.hello()
    link.theta = link.theta.to(device)
    model = ft.start_model(arm, "cpu")
    with torch.no_grad():
        vector_to_parameters(link.theta.cpu(), model.parameters())
    state = {"step": link.state["step"], "tokens": link.state["tokens"], "target_tokens": link.state["target_tokens"],
             "params": model.num_params(), "shelf_tokens": link.state["shelf_tokens"]}
    link.tokens_at_sync = state["tokens"]
    link.shelves_at_sync = dict(link.state["shelf_tokens"])
    print(f"worker {rank}: round {link.state['round']} · {device}", flush=True)
    try:
        ft.train(arm, state["target_tokens"], seed=1_000_003 * (rank + 1), bf16=bf16, name=f"{arm}.w{rank}",
                 device=device, model=model, state=state, sync=link, recompute=recompute, validate=own_validation and rank == 0,
                 max_temp=max_temp)
    except Finished:
        print(f"worker {rank}: the run is finished", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["coordinator", "worker", "validator"])
    p.add_argument("arm", help="the recipe from final_test (e.g. bil2)")
    p.add_argument("--name", help="coordinator: run folder (default <arm>-diloco)")
    p.add_argument("--tokens", type=float, default=1e8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--every", type=int, default=OUTER["every"], help="inner steps between syncs (H)")
    p.add_argument("--coordinator", default="172.31.115.13")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--no-recompute", action="store_true", help="worker: keep all activations (12 GB cards)")
    p.add_argument("--max-temp", type=int, help="worker: pause between steps to keep the card under this (C)")
    p.add_argument("--no-validate", action="store_true", help="worker: rank 0 skips validation (a validator runs)")
    p.add_argument("--every-rounds", type=int, default=2, help="validator: validate θ every n rounds")
    a = p.parse_args()
    OUTER["every"] = a.every
    if a.what == "coordinator":
        coordinate(a.arm, a.name or f"{a.arm}-diloco", a.tokens, a.workers, a.port)
    elif a.what == "validator":
        validate(a.arm, a.coordinator, a.port, a.every_rounds)
    else:
        work(a.arm, a.coordinator, a.rank, a.port, recompute=not a.no_recompute, max_temp=a.max_temp,
             own_validation=not a.no_validate)


if __name__ == "__main__":
    main()
