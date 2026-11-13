"""Validation of one arm over its run (run on the M6): tokens and bits per byte per shelf, every n-th row."""
import json
import sys
import urllib.request

name, every = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
d = json.load(urllib.request.urlopen("http://localhost:8765/api/final"))
a = [x for x in d["arms"] if x["name"] == name][0]
for i, v in enumerate(a["valid"]):
    if i % every == 0 or i == len(a["valid"]) - 1:
        print(f"{v['tokens'] / 1e6:7.1f}M", "  ".join(f"{k.split('_', 1)[1]} {x:.3f}" for k, x in v["valid_bpb"].items()))
