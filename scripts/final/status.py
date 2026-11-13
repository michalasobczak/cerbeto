"""Short status of the arms on the panel (run on the M6): progress, last validation in bits per byte, v6's."""
import json
import urllib.request

d = json.load(urllib.request.urlopen("http://localhost:8765/api/final"))
arms = {a["name"]: a for a in d["arms"]}
for name in ("vext", "bil2"):
    a = arms.get(name)
    if not a:
        continue
    v = a["valid"][-1] if a["valid"] else {}
    last = a.get("last") or {}
    print(name, f"progress {a['progress']:.3f}", "tokens", last.get("tokens"), "shelf_loss", last.get("shelf_loss"),
          "pair_loss", last.get("pair_loss"), "lr", last.get("lr"), "tok_s", last.get("tok_s"))
    print("   valid at", v.get("tokens"), v.get("valid_bpb"))
print("v6", arms["vext"].get("ref_bpb"))
