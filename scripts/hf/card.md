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

Cerbeto is a 106.8M-parameter language model trained from scratch on a single Mac mini, grown step by step like a child:
nursery → preschool → school. Code, specs and the whole story: **[github.com/michalasobczak/cerbeto](https://github.com/michalasobczak/cerbeto)**.

**Try it: [Cerbeto chat Space](https://huggingface.co/spaces/michalasobczak/cerbeto-chat)** (runs in your browser, ~110 MB downloaded once).

This checkpoint is **student D during the one-billion-token run (spec 0010)**: step {STEP}, **{TOKENS} of 1,000M tokens**.
It is a snapshot from the middle of training; later snapshots are tagged `bil-<tokens>M`.

## Architecture

| | |
|---|---|
| parameters | 106,849,920 |
| blocks | 20 (2 prelude + 16 core + 2 coda), K = 1 |
| width | d = 640, 20 heads × 32 |
| MLP | SwiGLU, 1792 |
| positions | RoPE (θ = 10,000) |
| norm | RMSNorm |
| vocabulary | 8,192 BPE, tied embedding |
| context | 512 tokens |

Grown function-preservingly from a 34.6M recurrent-depth student (Huginn-style prelude / core × K / coda): wider, the core unrolled into 16 blocks.
Close in shape to SmolLM-135M and MobileLLM-125M.

## Training data

Two of three steps are windows of text shelves: educational text (FineWeb-Edu, up to grade 7) 32%, Cosmopedia stories for young children 22%,
SODA conversations 15%, TinyStories 15%, spoken English (BabyLM: CHILDES, Switchboard, BNC spoken, OpenSubtitles) 8%, children's books (Project Gutenberg via BabyLM) 5%, Simple English Wikipedia 3%.
Every third step: request → reply pairs generated and checked by code — conversations of 2–8 turns (facts the human tells and asks back later),
preschool lessons, stories on request, who the student is, two-step relations. Exam prompts never appear in training.

Corpora keep their own licences: TinyStories CDLA-Sharing-1.0, FineWeb-Edu ODC-By 1.0, Cosmopedia Apache-2.0, SODA CC BY 4.0,
Wikipedia CC BY-SA 4.0, BabyLM MIT, Gutenberg public domain.

## What it can and cannot do (spot check at ~420M tokens)

- **Can:** short polite conversation, remember a name or a pet told earlier in the talk, preschool skills (opposites, sizes, categories,
  "who is the tallest"), fluent simple stories.
- **Cannot yet:** answer world-knowledge questions (it often knows the fact when completing a sentence, but answers questions with preschool
  templates such as "The moon says oink"), react to sad news, say "I don't know" about made-up words, keep a story to the requested topic.

The 80M-token predecessor passed the preschool exam (code-graded: id 89.7–100% per skill, mean ood_hard 78.5%).
Do not use it for anything that matters: it is a research toy.

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
