---
title: Cerbeto chat
emoji: 🐣
colorFrom: yellow
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://github.com/michalasobczak/cerbeto/blob/main/LICENSE.md
models:
- michalasobczak/cerbeto
short_description: Chat with a 107M model trained from scratch like a child
---

Chat with [Cerbeto](https://huggingface.co/michalasobczak/cerbeto), a small language model trained from scratch on one Mac mini and raised
like a child (nursery → preschool → school). The model runs entirely in your browser (transformers.js, ONNX int8, ~110 MB, downloaded once).
`models/{TAG}/` holds the checkpoint `{TAG}` converted to the Llama layout (identical logits) and exported to ONNX.
Code and story: [github.com/michalasobczak/cerbeto](https://github.com/michalasobczak/cerbeto).
