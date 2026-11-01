"""Command line entry point: `cerbeto <group> <command>`."""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="cerbeto")
    sub = parser.add_subparsers(dest="group", required=True)

    lib = sub.add_parser("library", help="school library (spec 0001)")
    lib_sub = lib.add_subparsers(dest="cmd", required=True)
    build = lib_sub.add_parser("build", help="build text shelves from the raw library")
    build.add_argument("--shelf", action="append", help="only this shelf (repeatable)")
    build.add_argument("--force", action="store_true")
    tok = lib_sub.add_parser("tokenize", help="tokenize shelves into .bin streams")
    tok.add_argument("--force", action="store_true")
    lib_sub.add_parser("report", help="shelves, licences, tokens and gate results")

    tk = sub.add_parser("tokenizer", help="own BPE tokenizer (spec 0001)")
    tk.add_subparsers(dest="cmd", required=True).add_parser("train")

    tr = sub.add_parser("train", help="train a student")
    tr.add_argument("--smoke", action="store_true", help="fresh student, fixed time budget (spec 0002)")
    tr.add_argument("--minutes", type=float, default=30)
    tr.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    tr.add_argument("--shelf", default="s0_stories")
    tr.add_argument("--batch", type=int, default=16)

    nu = sub.add_parser("nursery", help="stage 0 language learning (spec 0003)")
    nu_sub = nu.add_subparsers(dest="cmd", required=True)
    nr = nu_sub.add_parser("run", help="train (or resume) a student in the nursery")
    nr.add_argument("--student", choices=["growing", "control"], required=True)
    nr.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    nr.add_argument("--max-tokens", type=int)

    ex = sub.add_parser("exam", help="exams")
    ex_sub = ex.add_subparsers(dest="cmd", required=True)
    en = ex_sub.add_parser("nursery", help="BLiMP + conversation probe (spec 0003)")
    en.add_argument("--student", default="growing")
    en.add_argument("--version")
    en.add_argument("--blimp", action="store_true")
    en.add_argument("--conversation", action="store_true")
    en.add_argument("--device", default="cpu", help="cpu keeps the GPU free for training")

    ch = sub.add_parser("chat", help="talk to a student in the terminal")
    ch.add_argument("--student", default="growing")
    ch.add_argument("--version", help="checkpoint version, e.g. v0032 (default: latest)")
    ch.add_argument("--k", type=int, default=4, help="passes of the recurrent core")
    ch.add_argument("--temperature", type=float, default=0.7)
    ch.add_argument("--device", help="mps / cpu (default: best available)")
    ch.add_argument("--say", action="append", help="one-shot: say this and print the reply (repeatable)")

    args = parser.parse_args()
    if args.group == "library" and args.cmd == "build":
        from cerbeto.library.shelves import build_all
        build_all(args.shelf, args.force)
    elif args.group == "library" and args.cmd == "tokenize":
        from cerbeto.library.tokenize import tokenize_all
        tokenize_all(args.force)
    elif args.group == "library" and args.cmd == "report":
        from cerbeto.library.report import report
        sys.exit(0 if report() else 1)
    elif args.group == "train":
        if not args.smoke:
            parser.error("only --smoke training exists so far (full training: spec 0003)")
        from cerbeto.train.smoke import run
        summary = run(args.minutes, args.precision, args.shelf, args.batch)
        sys.exit(0 if summary["drop"] >= 0.5 else 1)
    elif args.group == "nursery" and args.cmd == "run":
        from cerbeto.train.loop import Trainer, load_config
        trainer = Trainer(args.student, load_config())
        if not args.fresh and trainer.resume():
            print(f"resumed {args.student} at {trainer.state['tokens']:,} tokens")
        trainer.run(args.max_tokens)
    elif args.group == "exam" and args.cmd == "nursery":
        from cerbeto.exams.nursery import run as run_exam
        both = not (args.blimp or args.conversation)
        run_exam(args.student, args.version, args.blimp or both, args.conversation or both, args.device)
    elif args.group == "chat":
        from cerbeto.talk import run as run_chat
        run_chat(args.student, args.version, args.device, args.k, args.temperature, args.say)
    elif args.group == "tokenizer" and args.cmd == "train":
        from cerbeto.library.shelves import load_config
        from cerbeto.tokenizer import train
        cfg = load_config()["tokenizer"]
        train(cfg["vocab_size"], cfg["sample_mb_per_shelf"])
        print("tokenizer saved")
