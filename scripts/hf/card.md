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

Cerbeto is a 106.8M-parameter language model trained from scratch, first on a single Mac mini and later on other commodity hardware, grown step by step like a
child: nursery → preschool → school. Code, specs and the whole story: **[github.com/michalasobczak/cerbeto](https://github.com/michalasobczak/cerbeto)**.

**Try it: [Cerbeto chat Space](https://huggingface.co/spaces/michalasobczak/cerbeto-chat)** (runs in your browser, ~110 MB downloaded once).

This checkpoint is **version {TAG}**: **v6 moved to a 16,384-word vocabulary, a second billion tokens (spec 0014/0015) and
ten billion tokens of a new school-level data set (spec 0016)**, 12.5B tokens read in all, trained by DiLoCo on several GPUs.
Versions are tagged `v<N>-<tokens>M` (v1, v2, … then the tokens read in all), the newest is on `main`; v1–v5 are snapshots
from the middle of the 1B run, v6 is the conversation-tuned 1B model.

## Architecture

| | |
|---|---|
| parameters | 106,849,920 |
| blocks | 20 (2 prelude + 16 core + 2 coda), K = 1 |
| width | d = 640, 20 heads × 32 |
| MLP | SwiGLU, 1792 |
| positions | RoPE (θ = 10,000) |
| norm | RMSNorm |
| vocabulary | 16,384 BPE, tied embedding (8,192 up to v6) |
| context | 512 tokens |

Grown function-preservingly from a 34.6M recurrent-depth student (Huginn-style prelude / core × K / coda): wider, the core unrolled into 16 blocks.
Close in shape to SmolLM-135M and MobileLLM-125M.

## Training data

The ten-billion-token run (three of four steps): FineWeb-Edu paragraphs of grade 8–12 40%, Cosmopedia stories for older
readers 19%, Cosmopedia textbooks and lessons 15%, and 26% the earlier shelves (FineWeb-Edu up to grade 8, textbooks,
SODA conversations, TinyStories, spoken English from BabyLM, children's books, Simple English Wikipedia).
Every fourth step: request → reply pairs — conversations of 2–8 turns (facts the human tells and asks back later), questions
about the world with short answers written by a teacher model, preschool lessons, stories on request, "I don't know" about
made-up words. Exam prompts never appear in training.

Corpora keep their own licences: TinyStories CDLA-Sharing-1.0, FineWeb-Edu ODC-By 1.0, Cosmopedia Apache-2.0, SODA CC BY 4.0,
Wikipedia CC BY-SA 4.0, BabyLM MIT, Gutenberg public domain.

## What it can and cannot do (bench1000)

1,000 questions asked once, greedy, an answer right when it matches a pattern; none of them in the tuning data.
A difference of ~25 points is beyond chance.

| Version | Tokens read | bench1000 |
|---|---|---|
| v1 | 0.44B | 312 |
| v5 | 0.70B | 316 |
| v6 | 1.03B | 488 |
| **{TAG}** | 12.5B | **503** |

Ten times more text lowered validation on every new shelf (school text, textbooks) but moved the answers little: knowledge
asked as a question stayed at ~69 of 280 (capitals, animals, nature). Arithmetic is near zero.

- **Can:** short polite conversation, remember a name or a pet told earlier in the talk, preschool skills (opposites, sizes,
  categories, "who is the youngest"), simple stories, sentences about school topics; it can **think out loud** before a
  true/false answer.
- **Cannot yet:** most facts asked directly, arithmetic; it is a 107M model with a 512-token context, so it is easily confused
  and makes things up.

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
