"""Step 3: export the Llama-layout model to ONNX with a KV cache, lay out the transformers.js folder, quantize to int8, and compare greedy output of ONNX fp32 / int8 with PyTorch.

    uv run --no-project --python 3.12 --with optimum-onnx --with onnxruntime --with torch python scripts/hf/export_web.py
"""
import shutil, subprocess, sys
from pathlib import Path
from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, LlamaForCausalLM

HF, ONNX, OUT = Path("/tmp/web/cerbeto_hf"), Path("/tmp/web/onnx_out"), Path("/tmp/web/model")
subprocess.run(["optimum-cli", "export", "onnx", "--model", str(HF), "--task", "text-generation-with-past", str(ONNX)], check=True)
shutil.rmtree(OUT, ignore_errors=True); (OUT / "onnx").mkdir(parents=True)
for f in ["config.json", "generation_config.json"]: shutil.copy(ONNX / f, OUT / f)
for f in HF.glob("*token*"): shutil.copy(f, OUT / f.name)
shutil.copy(ONNX / "model.onnx", OUT / "onnx/model.onnx")
quantize_dynamic(str(OUT / "onnx/model.onnx"), str(OUT / "onnx/model_quantized.onnx"), weight_type=QuantType.QUInt8)
for f in sorted(OUT.rglob("*")): print(f, f.stat().st_size // 1024, "KB")

tok = AutoTokenizer.from_pretrained(HF)
ref = LlamaForCausalLM.from_pretrained(HF).eval()
runs = {"fp32": ORTModelForCausalLM.from_pretrained(OUT, subfolder="onnx", file_name="model.onnx", use_cache=True),
        "int8": ORTModelForCausalLM.from_pretrained(OUT, subfolder="onnx", file_name="model_quantized.onnx", use_cache=True)}
for q in ["Tell me a story about a cat.", "What is the opposite of big?", "Hi! What is your name?"]:
    ids = tok("<|eos|><|human|> " + q + "\n<|student|>", return_tensors="pt", add_special_tokens=False).input_ids
    r = ref.generate(ids, max_new_tokens=40, do_sample=False)[0, ids.shape[1]:]
    print("\nQ:", q, "\n  torch:", tok.decode(r, skip_special_tokens=True))
    for name, m in runs.items():
        o = m.generate(ids, max_new_tokens=40, do_sample=False)[0, ids.shape[1]:]
        n = min(len(o), len(r))
        print(f"  {name}: same tokens {int((o[:n] == r[:n]).all())} ·", tok.decode(o, skip_special_tokens=True))
