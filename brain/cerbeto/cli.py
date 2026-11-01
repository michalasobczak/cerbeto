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
    ex_sub.add_parser("build", help="generate and freeze the school exam pools (spec 0004)")
    lk = ex_sub.add_parser("check-leak", help="exam prompts vs training tasks (spec 0004 G4)")
    lk.add_argument("--tasks", type=int, default=1_000_000)
    er = ex_sub.add_parser("run", help="school exams for a student version (spec 0004)")
    er.add_argument("--student", default="growing")
    er.add_argument("--version")
    er.add_argument("--device")
    er.add_argument("--limit", type=int, help="only the first N tasks of every pool (quick check)")

    sc = sub.add_parser("school", help="school (spec 0004 smoke check; full school: spec 0005)")
    sc_sub = sc.add_subparsers(dest="cmd", required=True)
    sm = sc_sub.add_parser("smoke", help="short direct lessons from the generators, then exams")
    sm.add_argument("--student", default="growing")
    sm.add_argument("--version")
    sm.add_argument("--minutes", type=float, default=30)
    sm.add_argument("--stage", default="preschool", help="preschool | grade1 | grade2 | all")
    sm.add_argument("--lr", type=float, default=3e-4)
    sm.add_argument("--device")
    sm.add_argument("--no-exam", action="store_true", help="skip the exams afterwards")

    ch = sub.add_parser("chat", help="talk to a student in the terminal")
    ch.add_argument("--student", default="growing")
    ch.add_argument("--version", help="checkpoint version, e.g. v0032 (default: latest)")
    ch.add_argument("--k", type=int, default=4, help="passes of the recurrent core")
    ch.add_argument("--temperature", type=float, default=0.7)
    ch.add_argument("--device", help="mps / cpu (default: best available)")
    ch.add_argument("--say", action="append", help="one-shot: say this and print the reply (repeatable)")

    tp = sub.add_parser("top", help="live dashboard of the students and the machine")
    tp.add_argument("--student", action="append", choices=["growing", "control"], help="only this student")
    tp.add_argument("--interval", type=float, default=5, help="seconds between refreshes")
    tp.add_argument("--once", action="store_true", help="print one snapshot and exit")

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
    elif args.group == "exam" and args.cmd == "build":
        from cerbeto.exams.school import build
        m = build()
        print(f"{m['tasks']:,} exam tasks in {len(m['pools'])} pools · exam set {m['exam_set_version']}")
        small = {k: v["tasks"] for k, v in m["pools"].items() if v["tasks"] < 300}
        if small:
            print("smaller pools (small task space):", small)
    elif args.group == "exam" and args.cmd == "check-leak":
        from cerbeto.exams.school import check_leak
        r = check_leak(args.tasks)
        print(r)
        sys.exit(0 if r["collisions_in_training_stream"] == 0 else 1)
    elif args.group == "exam" and args.cmd == "run":
        from cerbeto.exams.school import run as run_school_exam
        run_school_exam(args.student, args.version, args.device, args.limit)
    elif args.group == "school" and args.cmd == "smoke":
        from cerbeto.school.smoke import run as run_smoke
        r = run_smoke(args.student, args.version, args.minutes, None if args.stage == "all" else args.stage,
                      lr=args.lr, device=args.device)
        if not args.no_exam:
            from cerbeto.exams.school import run as run_school_exam
            run_school_exam("smoke", r["path"].rsplit("/", 1)[1], args.device)
    elif args.group == "chat":
        from cerbeto.talk import run as run_chat
        run_chat(args.student, args.version, args.device, args.k, args.temperature, args.say)
    elif args.group == "top":
        from cerbeto.top import run as run_top
        run_top(args.student, args.interval, args.once)
    elif args.group == "tokenizer" and args.cmd == "train":
        from cerbeto.library.shelves import load_config
        from cerbeto.tokenizer import train
        cfg = load_config()["tokenizer"]
        train(cfg["vocab_size"], cfg["sample_mb_per_shelf"])
        print("tokenizer saved")
