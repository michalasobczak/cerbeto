---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: LICENSE.md
language:
- en
library_name: pytorch
pipeline_tag: text-generation
tags:
- from-scratch
- small-language-model
- curriculum-learning
- conversational
datasets:
- roneneldan/TinyStories
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/cosmopedia
- allenai/soda
---

# Cerbeto — a small language model raised like a child

Cerbeto is a 106.8M-parameter language model trained from scratch, grown step by step like a child:
nursery → preschool → school. This checkpoint is **{TAG}**.

- **Try it:** [Cerbeto chat Space](https://huggingface.co/spaces/michalasobczak/cerbeto-chat) (runs in your browser, ~110 MB downloaded once).
- **Everything else is on GitHub: [github.com/michalasobczak/cerbeto](https://github.com/michalasobczak/cerbeto)** — the checkpoint table (tags here are `v<row>-<tokens>M`,
  the row of that table and the tokens read in all), architecture, training data and their licences, benchmark results and specs.

It is a research toy: easily confused, it makes things up. Do not use it for anything that matters.

## Use

The model is plain PyTorch (`student.py`, only `torch` needed) with a `tokenizers` BPE.

```python
import importlib.util, json, re, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

repo = "michalasobczak/cerbeto"
spec = importlib.util.spec_from_file_location("student", hf_hub_download(repo, "student.py"))
student = importlib.util.module_from_spec(spec); spec.loader.exec_module(student)
arch = json.load(open(hf_hub_download(repo, "architecture.json")))
model = student.Student(student.StudentConfig.from_dict(arch["config"])).eval()
model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")))
tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))

eos = tok.token_to_id("<|eos|>")
prompt = "<|human|> Tell me a story about a cat.\n<|student|>"
ids = torch.tensor([[eos] + tok.encode(prompt).ids])
out = model.generate(ids, 120, k=1, temperature=0.7)[0, ids.shape[1]:].tolist()
text = tok.decode(out, skip_special_tokens=False)
print(re.sub(r"<\|[^|]*\|>", "", text.split("<|eos|>")[0].split("<|human|>")[0].split("<|answer|>")[-1]).strip())
```

Conversation format: `<|human|> …\n<|student|><|answer|> …<|eos|>` per turn; the reply may start with `<|think|> … <|/think|>`.

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md): free for any noncommercial use (personal, research, education, hobby).
Commercial use requires the author's written permission — contact via [GitHub](https://github.com/michalasobczak).
