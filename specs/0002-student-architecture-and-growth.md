# 0002 — Student: architecture, growth, pruning, renewal

Status: planned · Depends on: 0001 · Unblocks: 0003, 0005

## 1. TL;DR
We build our own student network: a transformer with a recurrent loop ("thinking in a loop"), starting at ~8M parameters, with operators for growth (neurogenesis), pruning and neuron renewal, growing up to a max. of 30M. Proof: tests showing that growth does not change the network's behaviour, and that training converges on the nursery data.

## 2. Abstract
The specification requires an experimental architecture, iterative reasoning and "full development" instead of mere fine-tuning. Level B of biological fidelity was agreed: a simple neuron, biological developmental mechanisms. At ≤30M parameters full training of all weights takes <1 GB of memory, so adapters (LoRA) are unnecessary — we always change the full weights. Growing a network during its lifetime is known in the literature (Net2Net, GradMax, transformer stacking), but combining it with a growth controller is the research element of the project.

## 3. Objective
**Goals**
- A `Student` model (PyTorch) with three parts: *prelude* (input), *core* (recurrent block repeated K times), *coda* (output).
- Function-preserving growth operators: `grow_depth` (new layer in core), `grow_width` (wider MLP / more heads), `grow_model_dim` (wider residual stream).
- Structured pruning of low-utility MLP neurons; neuron renewal (continual backprop).
- Synaptic consolidation: EWC (diagonal Fisher matrix saved after passing a grade).
- Checkpoint: `safetensors` + `architecture.json` + `growth_log.jsonl` + optimizer state.

**Out of scope:** spiking neurons, learning without backprop, MoE, fast plasticity during chat, active dendrites (project stage 2).

**Success criteria**
- Start ≤ 10M parameters, hard limit 30M (growth beyond the limit is rejected).
- After every growth operator |Δ validation loss| ≤ 0.01 before further training.
- Training + inference of a 30M student at peak ≤ 3 GB RAM.

## 4. Artefact Inventory
| Artefact | Kind | Status |
|---|---|---|
| `brain/cerbeto/model/student.py` | system | to create |
| `brain/cerbeto/model/growth.py` (growth, pruning, renewal) | system | to create |
| `brain/cerbeto/model/consolidation.py` (EWC) | system | to create |
| `brain/cerbeto/model/checkpoint.py` | system | to create |
| `config/student.yaml` (starting size, limit, K) | data | to create |

## 5. Functional Domain Changes
**Starting architecture (proposal to be confirmed by measurement):**
- vocabulary 8192, `d_model` 256, 8 heads, SwiGLU MLP, RMSNorm, RoPE, context 512, tied embeddings;
- prelude 2 layers, core 2 layers × K iterations (training: K sampled 1–6; inference: K adjustable, "think longer"), coda 2 layers;
- ≈ 7–9M parameters.

**Growth rules:**
- `grow_depth`: new core layer with zeroed output projections (attention out, MLP down) → function unchanged.
- `grow_width`: new MLP neurons with input weights per GradMax (or small random) and zero output weights.
- `grow_model_dim`: new residual-stream dimensions zeroed; RMSNorm gain corrected by √(d_new/d_old).
- Optimizer state for new parameters zeroed; for old ones preserved.
- Every operator appends an event to `growth_log.jsonl` (what, where, how many parameters, reason).

**Pruning and renewal:**
- Neuron utility = mean |activation| × ‖output weights‖ (continual backprop, Dohare et al. 2024).
- Renewal: every sleep, reinitialization of ≤ 1% of the least useful neurons (maturity ≥ N steps).
- Pruning: after growth, when a neuron has utility < threshold for 3 sleeps — removal.

**Precision and device:**
- Device selected automatically: MPS → CUDA → CPU.
- fp32 by default; optionally mixed fp16 (autocast + gradient scaling), if it passes gate G6. **bf16 forbidden** (no hardware support on M1).
- Master weights and optimizer state always in fp32; checkpoint in fp32.

**Resilience:** every operator works on a copy; if the condition |Δ loss| ≤ 0.01 is not met, the operation is rejected and logged.

## 6. Critical Path
1. `Student` + forward/generate → 2. checkpoint save/load → 3. training on a small sample of the `s0_stories` shelf → 4. `grow_depth` with a function-preservation test → 5. remaining operators → 6. EWC.

**Completion condition:** an 8M student trained for 30 min on `s0_stories` generates recognizable English sentences; the growth tests pass.

## 7. Control Gates
| # | Type | Condition | Verification | Blocks |
|---|---|---|---|---|
| G1 | Unit | save→load gives identical logits (max diff < 1e-6) | `pytest tests/test_checkpoint.py` | 0003 |
| G2 | Unit | Every growth operator: max |Δ logits| < 1e-4 before training | `pytest tests/test_growth.py` | 0005 |
| G3 | Unit | Parameter count after growth matches the formula; growth > 30M → exception | `pytest tests/test_growth.py` | 0005 |
| G4 | Integration | 30 min of training: validation loss drops ≥ 50% from the start | `cerbeto train --smoke` | 0003 |
| G5 | Integration | Peak process RAM at 30M ≤ 3 GB | `psutil` measurement in a test | 0007 |
| G6 | Integration | fp16-autocast: loss after 30 min within ±2% of the fp32 loss, no NaN/inf; faster by ≥ 15% — otherwise fp32 stays | `cerbeto train --smoke --precision fp16` | precision choice in 0003 |
| N1 | Negative | The model loads no external weights and no external tokenizer | test checks for no network downloads | always |
| N2 | Negative | No model or optimizer tensor has dtype bf16 | test | always |

## 8. Expected Outcome
- **Direct:** a student that can grow, prune itself and renew.
- **Measurable:** parameter count and growth log in every version.
- **Side effects:** recurrence increases the compute cost K-fold for the core.
- **Rollback:** the checkpoint from before the operation (manifesto §4).

## 9. Additional Information
- **Rejected alternatives:** LoRA adapters (unnecessary at 30M); MLX as the training backend (Apple Silicon only; weights in safetensors keep the path open to inference in MLX); spiking networks; bf16 (no hardware support on M1); Core ML / Neural Engine — postponed (the NPU does not support training; inference via Core ML would require reconversion after every growth and unrolling the loop to a fixed K — to be reassessed after 0005).
- **Reference hardware:** MacBook Pro Apple M1 Pro, 32 GB; measured (PyTorch 2.15, MPS, matmul 4096²): CPU fp32 1.5 TFLOPS, GPU fp32 3.6, fp16 4.2, bf16 2.1 TFLOPS.
- **Technical debt:** GradMax may be simplified to a small random initialization in the first version.
- **Sources:** Net2Net (Chen et al., 2015); GradMax (Evci et al., 2022); Stacking Your Transformers (Du et al., 2024); Loss of plasticity in deep continual learning (Dohare et al., Nature 2024); EWC (Kirkpatrick et al., 2017); recurrent-depth transformers (Geiping et al., 2025).
- **Open questions:** order of growth operators (depth vs width) — decided by the experiment in 0005.
