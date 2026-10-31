# Source: MVP specification "AdaptiveMind Zero" (business input) — project Cerbeto

The input document from which specifications 0001–0007 were derived. It is not a change specification — we do not edit it; refinements go into the numbered specifications.

> The goal of the project is to build **from scratch** a proprietary, small neural model (**10–30M parameters**), with **its own architecture, tokenizer and randomly initialized weights**, **without using pre-trained models** as a base, capable of communicating in **natural language**, **multi-step reasoning** and **continual learning**. By default the model will conduct automatic dialogues with one or more external large LLMs acting as **teachers**, who ask questions, explain mistakes, give feedback and gradually increase the difficulty of tasks. **Verified interactions** will be **automatically turned into training data**, and the network parameters updated locally in a cyclical learning process, with **control of forgetting** of earlier skills and **independent assessment of progress**. The system will have **persistent memory**, an **iterative reasoning mechanism**, an **interface for chatting with the student** and a **local administration panel** enabling configuration of multiple teachers (API provider, model, role, instructions, schedule, cost limits), observing dialogues, monitoring reasoning effectiveness, training parameters and resource usage, **comparing successive model versions** and **rolling back updates**. The model architecture will remain experimental and independent of any specific family of neural networks. The project will be optimized for training on a **MacBook Pro with 32 GB** of memory and for local running and further learning on computers with **16 GB RAM**. **The MVP priority will be achieving a measurable improvement in reasoning on new tasks after automatic learning cycles with a teacher**, while maintaining **full control over the weights, data and the student's development process**.

<details><summary>Original (Polish)</summary>

> Celem projektu jest stworzenie od podstaw autorskiego, niewielkiego modelu neuronowego (10–30M parametrów), z własną architekturą, tokenizerem i losowo inicjalizowanymi wagami, bez wykorzystywania gotowych modeli jako podstawy, zdolnego do komunikacji w języku naturalnym, wieloetapowego wnioskowania i ciągłego uczenia się. Model będzie domyślnie prowadził automatyczne dialogi z jednym lub wieloma zewnętrznymi dużymi modelami LLM pełniącymi funkcje nauczycieli, którzy zadają pytania, wyjaśniają błędy, przekazują informacje zwrotne i stopniowo zwiększają trudność zadań. Zweryfikowane interakcje będą automatycznie przekształcane w dane treningowe, a parametry sieci aktualizowane lokalnie w cyklicznym procesie uczenia, z kontrolą zapominania wcześniejszych umiejętności i niezależną oceną postępów. System będzie posiadał trwałą pamięć, iteracyjny mechanizm rozumowania, interfejs rozmowy z uczniem oraz **lokalny panel administracyjny** umożliwiający konfigurowanie wielu nauczycieli (dostawca API, model, rola, instrukcje, harmonogram, limity kosztów), obserwowanie dialogów, monitorowanie skuteczności rozumowania, parametrów treningu i zużycia zasobów, porównywanie kolejnych wersji modelu oraz cofanie aktualizacji. Architektura modelu pozostanie eksperymentalna i niezależna od konkretnej rodziny sieci neuronowych. Projekt zostanie zoptymalizowany pod trening na MacBooku Pro z 32 GB pamięci oraz lokalne uruchamianie i dalsze uczenie na komputerach z 16 GB RAM. **Priorytetem MVP będzie uzyskanie mierzalnej poprawy rozumowania na nowych zadaniach po automatycznych cyklach nauki z nauczycielem**, przy zachowaniu pełnej kontroli nad wagami, danymi i procesem rozwoju ucznia.

</details>

## Decisions from the specification session (2026-10-31)
| # | Decision | Specification |
|---|---|---|
| 1 | Student language: **English**; language-independent pipeline (Polish as a later "new skill" — forgetting test) | 0001, 0004 |
| 2 | Learning "like at school, step by step": **stages 0–3** (nursery, preschool, grade 1, grade 2); no coding, mathematics slowly and in the context of language | 0003, 0004 |
| 3 | Biology at **level B**: developmental mechanisms (growth, pruning, sleep, hippocampus+cortex, consolidation, renewal), not a faithful neuron model | 0002, 0005 |
| 4 | **Full weights** instead of adapters; the student grows from ~8M to max. 30M | 0002 |
| 5 | Teacher: a **large LLM run locally** (Ollama), chosen by a teacher exam; a human as an additional teacher in chat | 0005, 0006 |
| 6 | Nursery on ready-made children's corpora (**"school library"**), not our own scraping; data offline, outside git | 0001 |
| 7 | **PyTorch** (portability); weights in safetensors (path to MLX for inference) | 0002 |
| 8 | Admin panel: **Ruby on Rails**; brain: Python — communication via HTTP API | 0006 |
| 9 | Copies held by people are **independent students**; nursery done centrally | 0007 |
| 10 | Exams: pass **≥ 85%** (id), forgetting gate ≤ 2 pp, transfer measurement, 30M control student | 0004, 0005 |
| 11 | **Automatic daily rhythm** with manual control in the admin panel | 0005 |
| 12 | Training on GPU via MPS; precision fp32 / fp16, **no bf16** (M1 without hardware support) | 0002, 0003 |
| 13 | **Core ML / Neural Engine postponed** (NPU for inference only, cost of reconversion after growth) | 0002 |
| 14 | Default teacher: **`nemotron-3.5-bolt:e4b`** (7.2 GB, ~36 tok/s, the same on 32 and 16 GB); fallback `qwen3.8:27b-mlx` | 0005, 0007 |
| 15 | Day/sleep rhythm **sequential** by default (shared GPU); nursery of both students sequentially, ~20 h on M1 Pro | 0003, 0005 |
| 16 | Project name: **Cerbeto** (Esperanto: "little brain"); public repository `michalasobczak/cerbeto`, package `cerbeto` | — |
