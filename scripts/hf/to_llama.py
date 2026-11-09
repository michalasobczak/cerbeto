"""Step 2: Cerbeto (K = 1) → LlamaForCausalLM: same math; our RoPE pairs (2i, 2i+1), Llama pairs (i, i + d/2), so q and k rows are permuted per head.
Checks that the logits match, then saves a transformers model + tokenizer to OUT.

    uv run --no-project --python 3.12 --with torch --with transformers --with safetensors --with tokenizers --with huggingface_hub \
        python scripts/hf/to_llama.py /tmp/web/cerbeto_hf
"""
import importlib.util, json, shutil, sys
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM
from tokenizers import Tokenizer

REPO, OUT = "michalasobczak/cerbeto", sys.argv[1]
spec = importlib.util.spec_from_file_location("student", hf_hub_download(REPO, "student.py"))
student = importlib.util.module_from_spec(spec); spec.loader.exec_module(student)
arch = json.load(open(hf_hub_download(REPO, "architecture.json")))
cfg = student.StudentConfig.from_dict(arch["config"])
src = student.Student(cfg).eval(); src.load_state_dict(load_file(hf_hub_download(REPO, "model.safetensors")))
blocks = list(src.blocks())
print("blocks", len(blocks), cfg)

lc = LlamaConfig(vocab_size=cfg.vocab_size, hidden_size=cfg.d_model, intermediate_size=cfg.mlp_hidden, num_hidden_layers=len(blocks),
                 num_attention_heads=cfg.n_heads, num_key_value_heads=cfg.n_heads, head_dim=cfg.head_dim, max_position_embeddings=cfg.max_seq_len,
                 rms_norm_eps=cfg.norm_eps, rope_theta=cfg.rope_theta, tie_word_embeddings=True, attention_bias=False, mlp_bias=False,
                 hidden_act="silu", torch_dtype="float32")
tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
EOS = tok.token_to_id("<|eos|>")
lc.bos_token_id = lc.eos_token_id = lc.pad_token_id = EOS
dst = LlamaForCausalLM(lc).eval()

perm = torch.cat([torch.arange(0, cfg.head_dim, 2), torch.arange(1, cfg.head_dim, 2)])
def rope_rows(w):   # [heads*hd, d]: (2i, 2i+1) pairs → (i, i + hd/2)
    return w.view(cfg.n_heads, cfg.head_dim, -1)[:, perm].reshape(w.shape)
sd = {"model.embed_tokens.weight": src.embed.weight, "model.norm.weight": src.norm.weight, "lm_head.weight": src.embed.weight}
for i, b in enumerate(blocks):
    p = f"model.layers.{i}."
    sd |= {p + "input_layernorm.weight": b.attn_norm.weight, p + "post_attention_layernorm.weight": b.mlp_norm.weight,
           p + "self_attn.q_proj.weight": rope_rows(b.attn.q.weight), p + "self_attn.k_proj.weight": rope_rows(b.attn.k.weight),
           p + "self_attn.v_proj.weight": b.attn.v.weight, p + "self_attn.o_proj.weight": b.attn.o.weight,
           p + "mlp.gate_proj.weight": b.mlp.gate.weight, p + "mlp.up_proj.weight": b.mlp.up.weight, p + "mlp.down_proj.weight": b.mlp.down.weight}
missing, unexpected = dst.load_state_dict({k: v.detach().clone() for k, v in sd.items()}, strict=False)
assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
dst.tie_weights()

ids = torch.tensor([[EOS] + tok.encode("<|human|> Tell me a story about a cat.\n<|student|><|answer|> Once upon a time, there was a cat").ids])
with torch.no_grad():
    a, b = src(ids, 1), dst(ids).logits
diff = (a - b).abs().max().item()
same_argmax = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
print(f"max |logit diff| {diff:.2e} · same argmax {same_argmax:.3f}")
assert diff < 1e-3 and same_argmax == 1.0
with torch.no_grad():
    g1 = src.generate(ids, 40, k=1, temperature=0)[0, ids.shape[1]:].tolist()
    g2 = dst.generate(ids, max_new_tokens=40, do_sample=False)[0, ids.shape[1]:].tolist()
print("greedy 40 tokens identical:", g1 == g2)

dst.generation_config.eos_token_id = [EOS, tok.token_to_id("<|human|>"), tok.token_to_id("<|student|>")]
dst.save_pretrained(OUT, safe_serialization=True)
specials = [t for t in tok.get_vocab() if t.startswith("<|") and t.endswith("|>") and t != "<|eos|>"]
shutil.copy(hf_hub_download(REPO, "tokenizer.json"), Path(OUT) / "tokenizer.json")
tc = {"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<|eos|>", "eos_token": "<|eos|>", "pad_token": "<|eos|>",
      "additional_special_tokens": sorted(specials), "model_max_length": cfg.max_seq_len, "clean_up_tokenization_spaces": False}
(Path(OUT) / "tokenizer_config.json").write_text(json.dumps(tc, indent=2))
(Path(OUT) / "special_tokens_map.json").write_text(json.dumps({k: tc[k] for k in ["bos_token", "eos_token", "pad_token", "additional_special_tokens"]}, indent=2))
print("SAVED", OUT)
