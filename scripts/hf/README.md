# Publishing Cerbeto on Hugging Face

- Model (native format, tags per checkpoint): <https://huggingface.co/michalasobczak/cerbeto>
- Chat in the browser (static Space, free): <https://huggingface.co/spaces/michalasobczak/cerbeto-chat>

Run on the Mac mini. The token is `hf=…` in the repo's `.env` (git-ignored); the scripts never print it.
Gradio Spaces need HF PRO on the free CPU, so the chat is a static page: the model runs in the visitor's browser
(transformers.js on WASM, ONNX int8 with a KV cache, ~110 MB downloaded once).

| step | script | what it does |
|---|---|---|
| 1 | `publish_model.py <checkpoint>` | uploads weights, `architecture.json`, `student.py`, tokenizer, licence and the card (`card.md`), tags the next version `v<N>-<tokens>M` (v1, v2, … then the tokens trained) and prints it |
| 2 | `to_llama.py /tmp/web/cerbeto_hf` | Cerbeto at K = 1 is a Llama: same blocks, only RoPE pairs differ (ours `(2i, 2i+1)`, Llama `(i, i + d/2)`), so q and k rows are permuted per head; checks logits (≤ 1e-3) and 40 greedy tokens against the original |
| 3 | `export_web.py` | ONNX export with a KV cache (`optimum-cli`), int8 dynamic quantization (ONNX Runtime, `QUInt8`), greedy output of fp32 / int8 compared with PyTorch |
| 4 | `publish_space.py <tag>` | the page (`web/index.html`) and the int8 model to the Space |
| before 1 | `spot_check.py <last tag> <checkpoint>` | the same chat questions asked 10 times each to both, counted right by a pattern (CPU, training not disturbed): publish when chat is better |
| any time | `update_card.py` | uploads `card.md` again, filled for the newest tag, without new weights |

```bash
cd ~/Work/indatify/platformy/model/brain
uv run --with huggingface_hub python ../scripts/hf/publish_model.py checkpoints/final/bil/final
cd ..
uv run --no-project --python 3.12 --with torch --with transformers --with safetensors --with tokenizers --with huggingface_hub python scripts/hf/to_llama.py /tmp/web/cerbeto_hf
uv run --no-project --python 3.12 --with optimum-onnx --with onnxruntime --with torch python scripts/hf/export_web.py
cd brain
uv run --with huggingface_hub python ../scripts/hf/publish_space.py v<N>-1000M      # the tag step 1 printed
```

Before step 1, compare the checkpoint with the last version (`spot_check.py`) and update the "What it can and cannot do" section of `card.md` with the result.
`to_llama.py` reads the model from the hub (the main branch), so step 1 comes first.
