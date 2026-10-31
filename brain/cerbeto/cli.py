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
    elif args.group == "tokenizer" and args.cmd == "train":
        from cerbeto.library.shelves import load_config
        from cerbeto.tokenizer import train
        cfg = load_config()["tokenizer"]
        train(cfg["vocab_size"], cfg["sample_mb_per_shelf"])
        print("tokenizer saved")
