# Czas treningu i podział ról: 4 × RTX 3060 i Mac mini M6

Krok 3 z 6 analizy lokalnego rdzenia rozumowania. Stan: 21 listopada 2026 r. Wejście: warianty i FLOP-y z [kroku 2](model-size-and-training-budget.md), architektury z [kroku 1](architecture-review.md). Wszystkie czasy to szacunki planistyczne. Jedynymi pomiarami są przepustowości z dotychczasowych przebiegów Cerbeto, opisane w sekcji „Kalibracja”.

## Wniosek

1. **4 × 3060 to sprzęt dla modeli do ~300M parametrów od zera i dla każdego dostrajania do ~1,5B.** D (106,8M) na 1–3B tokenów to **4–18 h**, 300M na 4–10B tokenów to **2–7 dni**, 600M na 8–20B tokenów to **8–33 dni**, a 1B na 15–30B tokenów to **1–4 miesiące**. Ostatni wariant wypada z planu. Dodatkowo przy 600M–1B granicą są dane: Cerbeto ma 6,28B unikalnych tokenów w nowym zbiorze ([0016](../specs/0016-ten-billion.md)).
2. **Wszystkie ścieżki z kroku 2 poza pretreningiem mieszczą się w godzinach.** Dotyczy to dostrojenia D, LoRA na Qwen3-0.6B i Llama 3.2 1B oraz QLoRA 8B na jednej karcie. Generowanie danych przez nauczyciela 7B zajmie **5–26 h** na 4 kartach. To więcej niż trening ucznia, co potwierdza obserwację z kroku 2.
3. **Do ~300M wystarczy DDP. 600M wymaga FSDP2/ZeRO albo 8-bitowego optymalizatora, a 1B zawsze FSDP2 z checkpointingiem.** Narzut komunikacji po PCIe bez P2P: ≤5% dla DDP z akumulacją gradientu i ~10–25% dla FSDP przy 1B.
4. **Dla etapu MVP 4 niezależne przebiegi na 4 kartach są cenniejsze niż jeden rozproszony.** Dotyczy to sweepów hiperparametrów, porównań architektur z kroku 1 i ziaren. Każdy przebieg dostrojenia lub LoRA trwa od <1 h do kilku godzin na jednej karcie.
5. **Mac mini M6 (16 GB, 153 GB/s) to maszyna do inferencji, ewaluacji, serwowania RAG i szybkich prototypów w MLX.** Szacunkowo daje ~65–95 tok/s dla Qwen3-0.6B w bf16, ~150–280 tok/s w 4 bitach i ~17–26 tok/s dla modelu 7–8B w 4 bitach. LoRA na 0,6B przy 10M tokenów zajmie 3–5 h, czyli mieści się w jednej nocy. Pretrening na M6 jest ~4× wolniejszy niż na jednej 3060 (pomiar).

## Kalibracja: co już zmierzono w Cerbeto

| Pomiar | Wartość | Źródło |
|---|---|---|
| D 112M (słownik 16k), bf16 autocast, SDPA, bez `torch.compile`, bez checkpointingu, batch 8 × 512, DiLoCo H = 500, 3 workery na kartach 12 GB | **~33 000 tok/s łącznie, ~11 000 tok/s na kartę** | [0016](../specs/0016-ten-billion.md), [0018](../specs/0018-twenty-billion.md), [CHANGELOG](../CHANGELOG.md) |
| To samo w FLOP-ach | 11 000 × 6 × 112M ≈ **7,4 TFLOPS na kartę ≈ 29% MFU** (≈32% z FLOP-ami uwagi) | rachunek własny, szczyt z tabeli niżej |
| D 106,8M na Macu mini M6, PyTorch MPS, bf16, z checkpointingiem aktywacji | **~2 550–2 630 tok/s** (≈1,75 TFLOPS efektywnie wg 6ND) | [0010](../specs/0010-one-billion-tokens.md), [0014](../specs/0014-second-billion.md) |
| 283M na M6, bf16 | ~1 390 tok/s, ~1,4 GB swapu przy 16 GB | [0008](../specs/0008-growth-experiment-100m.md) |
| Lokalny nauczyciel 4B (efektywnie) w Ollama/MLX | ~36 tok/s | [specyfikacja źródłowa](../specs/source-mvp-specification.md) |
| 20B tokenów na D w DiLoCo | ~7 dni | [0018](../specs/0018-twenty-billion.md) |

Kod treningu ma przełącznik `--no-recompute` opisany jako „12 GB cards”. Na tej podstawie przyjmuję, że workery DiLoCo to karty 3060 z tego planu. Repo nie podaje jednak modelu kart ani ich rozkładu na maszyny. [Spec 0015](../specs/0015-diloco.md) mówi o kilku maszynach połączonych **1 GbE**. To odmienna topologia od założenia „4 karty na PCIe w jednej obudowie” i omawiam ją osobno w sekcji o komunikacji.

## Założenia

| Założenie | Wartość | Uzasadnienie |
|---|---|---|
| Szczyt RTX 3060 dla bf16/fp16 z akumulacją fp32 (tak liczy PyTorch) | **25,5 TFLOPS** dense | 28 SM × 1024 FLOP/takt (FP16 z akumulacją FP16) × 1,777 GHz = 51 TFLOPS. Na GeForce Ampere akumulacja FP32 ma połowę tej przepustowości, analogicznie do RTX 3090: 71 wobec 142 TFLOPS. [Specyfikacja 3060](https://www.techspot.com/specs/gpu/223291-nvidia-geforce-rtx-3060-12gb.html) · [GA102 whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf) |
| Pamięć 3060 | 12 GB GDDR6, 192 bit, 360 GB/s, 170 W | [Specyfikacja](https://www.techspot.com/specs/gpu/223291-nvidia-geforce-rtx-3060-12gb.html) |
| MFU pretreningu i pełnego dostrajania | **30% (pomiar Cerbeto) – 45%** (`torch.compile`, większe mikro-batche, sekwencja 1–2k) | Kalibracja powyżej; 45% to cel, nie pomiar |
| MFU dla 10–35M | 15–30% | Małe macierze słabo wykorzystują rdzenie tensorowe |
| MFU dla LoRA (bf16) / QLoRA (NF4) | 20–35% / 12–25% | Dekwantyzacja i małe GEMM adapterów. Do zmierzenia |
| Koszt FLOP | 6ND (pretrening, pełne dostrajanie), 4ND (LoRA), 2N(P+G) (nauczyciel) | [Krok 2](model-size-and-training-budget.md#podstawa-liczb) |
| Checkpointing aktywacji | ×1,33 obliczeń przy pełnym, ~×1,1 przy selektywnym | Jedno dodatkowe przejście w przód. [Korthikanti i in.](https://arxiv.org/abs/2205.05198) |
| Komunikacja między kartami | **3–6 GB/s efektywnego „bus bandwidth” NCCL** | GeForce Ampere nie ma P2P po PCIe, więc NCCL kopiuje przez pamięć hosta. Gniazda x4 z chipsetu dzielą łącze DMI. [Tom's Hardware](https://www.tomshardware.com/news/nvidia-confirms-geforce-cards-lack-p2p-support) · [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html). Do zmierzenia przez `nccl-tests` |
| Mac mini M6 | 16 GB, **153 GB/s** (24/32 GB: 170 GB/s) | [Apple – specyfikacja](https://www.apple.com/mac-mini/specs/) · [9to5Mac](https://9to5mac.com/2026/08/27/m6-mac-mini-three-things-apple-didnt-highlight-in-the-announcement/) |
| Efektywna moc obliczeniowa M6 | trening 1,5–2,0 TFLOPS (pomiar PyTorch MPS), prefill w MLX 3–6 TFLOPS (założenie) | Kalibracja powyżej |

Czasy w tabelach to czysty trening. Do planu trzeba doliczyć **10–20%** na walidację, zapisy checkpointów, restarty i throttling termiczny. Repo ma już pacer temperatury dla karty z wadliwym chłodzeniem ([0015](../specs/0015-diloco.md)).

## Tabela czasów

Zakres „najlepiej” łączy dolny FLOP z kroku 2, MFU 45% i mały narzut komunikacji. Zakres „najgorzej” łączy górny FLOP, MFU 30% i duży narzut. „1 karta” oznacza pojedynczy przebieg w sweepie, bez komunikacji. 1 dzień = 24 h.

### Pretrening od zera

| Wariant (krok 2) | Tokeny | FLOP | Równoległość | Narzut komunikacji / recompute | **4 × 3060** | 1 × 3060 | Mac M6 (MPS) |
|---|---:|---:|---|---|---:|---:|---:|
| 10–35M | 0,2–1B | 1,2e16–2,1e17 | 4 niezależne przebiegi (sweep) | – | **0,1–4 h** | 0,5–15 h | 2–40 h |
| **D 106,8M** | 1–3B | 6,41e17–1,92e18 | DDP lub DiLoCo | 2–5% | **4–18 h** (0,2–0,8 d) | 16–70 h | 3,7–15 d |
| 300M | 4–10B | 7,2e18–1,8e19 | DDP, ewentualnie 8-bit AdamW | 3–8% | **45–177 h (1,9–7,4 d)** | 7–27 d | ≥ 6 tyg., niepraktyczne |
| 600M | 8–20B | 2,88e19–7,2e19 | FSDP2 (ZeRO-2) | 8–20% | **8–33 d** | 1–3,6 mies. | – |
| 1B | 15–30B | 9,0e19–1,8e20 | FSDP2 + checkpointing | 25–65% (recompute + komunikacja) | **28–112 d** | – | – |

Sprawdzenie zgodności: wiersz D przewiduje ~5,8 h na 1B tokenów przy MFU 30% na 4 kartach. Pomiar z 3 kart (33 tys. tok/s) daje 8,4 h, co po przeskalowaniu na 4 karty wynosi ~6,3 h.

### Dostrajanie, destylacja, LoRA/QLoRA

| Wariant (krok 2) | Tokeny | FLOP | Gdzie i jak | **4 × 3060** (DDP) | **1 × 3060** | Mac M6 (MLX) |
|---|---:|---:|---|---:|---:|---:|
| Pełne dostrojenie D | 20–200M | 1,28e16–1,28e17 | 1 karta; sweep LR × ziarno | 0,1–1,2 h | **0,3–4,6 h** | 2–24 h |
| Pełne dostrojenie 300M | 20–200M | 3,6e16–3,6e17 | 1 karta lub DDP | 0,2–3,5 h | **0,9–13 h** | 5–67 h |
| LoRA SmolLM2-135M / 360M | 10–50M | 5,4e15–7,2e16 | 1 karta | ≤1 h | **0,2–4 h** | 0,75–13 h |
| LoRA Qwen2.5-0.5B / 1.5B | 10–50M | 1,96e16–3,08e17 | 1 karta, bf16, bez kwantyzacji | 0,15–4,2 h | **0,6–17 h** | 3–57 h |
| **LoRA Qwen3-0.6B** | 10–50M | 2,4e16–1,2e17 | 1 karta, bf16 | 0,2–1,6 h | **0,75–6,5 h** | **3,3–22 h** |
| LoRA Llama 3.2 1B | 10–50M | 4,92e16–2,46e17 | 1 karta, bf16 | 0,4–3,3 h | **1,5–13 h** | 7–46 h |
| QLoRA 7–8B (np. adaptacja nauczyciela) | 10–50M | ≈4,3e17–2,1e18 (4ND × 1,33 za checkpointing) | 1 karta, NF4 | 5–48 h | **19–193 h** | niepraktyczne |
| Nauczyciel 7B: 50 tys. zadań × 2 próby, 512 + 512 tokenów | 102,4M (prefill + generacja) | 1,43e18 | 4 niezależne instancje vLLM, każda z modelem int4 na swojej karcie | **5–26 h** | 20–104 h | ~2,5 tyg. przy ~25–40 tok/s |

Zakres dla nauczyciela ma dwie granice. Górna (26 h) to rachunek FLOP przy szczycie 25,5 TFLOPS i MFU 15%. Dolna (~5 h) zakłada ciągłe batchowanie w vLLM i ~600–1 200 tok/s generacji na kartę. Model int4 zajmuje ~4,5 GB, a GQA w Qwen2.5-7B to ~57 kB KV na token, więc w 12 GB mieści się ~64–100 równoległych sekwencji po 1k tokenów. Kernele inferencji mogą też akumulować w fp16 (szczyt 51 TFLOPS). Oba końce zakresu trzeba zmierzyć na 200 zadaniach. [vLLM](https://github.com/vllm-project/vllm) · [Qwen2.5-7B config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)

Wniosek z tabeli: najtańszy etap MVP z kroku 2 (D przed i po dostrojeniu, zamrożony Qwen3-0.6B oraz jego LoRA) wymaga łącznie **mniej niż doby pracy kart**. Wąskim gardłem będą dane od nauczyciela i ewaluacja, nie trening.

## Pamięć na karcie 12 GB

Trening mieszany (bf16 autocast, wagi i AdamW w fp32) kosztuje **~16 B na parametr**: wagi fp32 (4 B), gradienty fp32 (4 B) i dwa momenty Adama (8 B). Do tego dochodzi kontekst CUDA (~0,5–0,8 GB), fragmentacja oraz aktywacje. Przy FlashAttention/SDPA aktywacje zajmują ok. **34·h bajtów na token i warstwę** (h to szerokość modelu), plus logity: tokeny × słownik × 4 B. [Korthikanti i in.](https://arxiv.org/abs/2205.05198) · [ZeRO](https://arxiv.org/abs/1910.02054)

| Model | Stany treningu (16 B/param) | Aktywacje (przykład) | DDP na 12 GB? | Co zrobić |
|---|---:|---|---|---|
| D 112M (20 warstw, h = 640) | 1,8 GB | 4k tokenów na mikro-batch: ~1,8 GB + logity 16k: 0,3 GB | **tak, bez checkpointingu** (pomiar: `--no-recompute`) | DDP; większy mikro-batch podniesie MFU |
| 300M (~24 warstwy, h = 1024) | 4,8 GB | 4k tokenów: ~3,4 GB; 8k: ~6,8 GB | **tak**, przy mikro-batchu ≤4–6k tokenów | DDP; selektywny checkpointing albo 8-bit AdamW, by podnieść batch |
| 600M | 9,6 GB | – | **na granicy / nie** | FSDP2 na 4 karty: 2,4 GB stanów na kartę. Albo DDP z 8-bit AdamW (~10 B/param = 6 GB) i checkpointingiem |
| 1B | 16 GB | – | **nie** | FSDP2: 4 GB stanów na kartę + checkpointing |
| 1,5–2B | 24–32 GB | – | nie | FSDP2: 6–8 GB na kartę, krótkie sekwencje. Praktyczny sufit pełnego treningu na 4 × 12 GB |
| ≥3B pełny trening | ≥48 GB | – | nie | Tylko z offloadem na CPU (bardzo wolno). Dla tej skali właściwe jest LoRA/QLoRA |

**Uwaga na duże słowniki gotowych modeli.** Qwen ma ~151k tokenów, więc logity dla 8k tokenów w fp32 zajmują ~4,9 GB. Przy LoRA na Qwen trzeba liczyć stratę kawałkami (chunked cross-entropy, np. kernele Liger) albo zmniejszyć mikro-batch. Słownik 16k w D tego problemu nie ma.

**LoRA i QLoRA na jednej karcie.** Modele 0,5–1,5B w bf16 (1–3 GB wag) z adapterem i checkpointingiem mieszczą się w 12 GB bez kwantyzacji. Dla nich QLoRA tylko spowalnia. QLoRA ma sens od ~4B: model 7–8B w NF4 zajmuje ~5–5,5 GB, a do tego dochodzą adapter, stany Adama adaptera i aktywacje przy sekwencji 2k z checkpointingiem, łącznie ~9–11 GB. Model 14B w NF4 (~8,5 GB) zmieści się tylko przy krótkich sekwencjach. [QLoRA](https://arxiv.org/abs/2305.14314)

## Komunikacja: DDP, FSDP/ZeRO i PCIe bez NVLinka

**DDP** wykonuje all-reduce gradientów raz na krok optymalizatora. Ring all-reduce przesyła 2(n−1)/n × rozmiar gradientów na kartę, czyli dla 4 kart **1,5 × 2N bajtów w bf16**.

| Model | All-reduce bf16 przy 3–6 GB/s | Obliczenia na krok (64k tokenów/kartę, MFU 30–35%) | Narzut |
|---|---:|---:|---:|
| D 112M | 0,06–0,11 s | ~5,8 s | **1–2%** |
| 300M | 0,15–0,3 s | ~13 s | **1–2%** |

Warunek: akumulacja gradientu z `no_sync()`, czyli synchronizacja raz na krok, a nie po każdym mikro-batchu. Domyślnie DDP przy autocast synchronizuje gradienty fp32 (2× więcej bajtów). Hook kompresji bf16 to tanie usprawnienie. [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)

**FSDP2 (`fully_shard`) / ZeRO-3** dzieli wagi, gradienty i stany optymalizatora między karty. Za to na każdy mikro-batch wykonuje all-gather wag w przód, ponownie w tył oraz reduce-scatter gradientów. To ok. **3 × 0,75 × 2N bajtów**, czyli dla 1B ~4,5 GB na mikro-batch, **0,75–1,5 s** przy 3–6 GB/s. Obliczenia dla 8k tokenów na kartę trwają ~5,5 s (bez recompute). Narzut wynosi więc 14–27% bez nakładania, a z prefetchem FSDP2 ~10–20%. Ustawienie `reshard_after_forward=False` daje zachowanie zbliżone do ZeRO-2: pomija all-gather w tył kosztem pamięci na pełne wagi bf16 (2 GB dla 1B). Dla 600M to rozsądny wybór. [FSDP2 `fully_shard`](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) · [torchtitan – FSDP](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)

**Sieć 1 GbE zamiast PCIe.** Jeśli karty siedzą w kilku maszynach, jak w dotychczasowym DiLoCo, all-reduce dla D trwa ~2,9 s na krok, a FSDP dla 1B ~40 s na mikro-batch. Synchroniczny trening między maszynami odpada. Wtedy:
- DDP/FSDP tylko w obrębie jednej maszyny;
- DiLoCo między maszynami. Zmierzony w Cerbeto koszt jakości jest zerowy, a komunikacja co H = 500 kroków ([0015](../specs/0015-diloco.md)).

Przed planowaniem 600M+ trzeba ustalić `nvidia-smi topo -m` i zmierzyć `all_reduce_perf` z [nccl-tests](https://github.com/NVIDIA/nccl-tests).

**Rekomendacja:**

| Model | Konfiguracja |
|---|---|
| ≤300M | DDP (lub obecne DiLoCo) |
| 600M | FSDP2 w trybie ZeRO-2 albo DDP z 8-bit AdamW |
| 1B | FSDP2 + selektywny checkpointing. Tylko po pozytywnym wyniku 300M i z nowymi danymi |

Tensor/pipeline parallel po PCIe bez P2P nie ma sensu w tej skali.

## Sweepy: 4 niezależne eksperymenty zamiast jednego treningu

W fazie MVP przewaga jest po stronie sweepów:
- brak komunikacji;
- awaria karty zatrzymuje 1/4 pracy;
- każdy przebieg ma ten sam kod co na Macu;
- odpowiedzi na pytania z kroków 1–2 wymagają **porównań**, nie jednego dużego modelu.

| Seria (4 karty × 1 przebieg) | Czas jednej serii | Co rozstrzyga |
|---|---:|---|
| D: dostrojenie na odpowiedziach nauczyciela, 4 × (LR, mieszanka) po 20–50M tokenów | ~0,5–1,5 h | Czy D uczy się użycia dowodu (krok 2) |
| Qwen3-0.6B LoRA: rank 8/16/32 × ziarno, 10M tokenów | ~0,75–1,5 h | Zysk adaptera względem zamrożonej bazy |
| Architektury z kroku 1 przy ~35M i 0,5B tokenów: dense / współdzielony blok K = 2 / K = 4 / Mamba-2 | ~2–8 h | Pętla po głębokości przy dopasowanym koszcie |
| Ziarna: 4 × ta sama konfiguracja | jak wyżej | Szum: w Cerbeto ±3 pp od samego ziarna ([0005](../specs/0005-school-day-sleep-exam.md)) |

Sweep na 4 kartach daje w nocy (~10 h) ~**40 kart-godzin**. To np. ~20–50 przebiegów LoRA na Qwen3-0.6B po 10M tokenów albo 5–10 małych pretreningów architektonicznych (35M, 0,5B tokenów). Rozproszony trening ma sens dopiero dla jednego długiego przebiegu: dalszego pretreningu D lub 300M.

## Mac mini M6 z MLX

**Rola:** inferencja i czat, ewaluacja checkpointów w trakcie treningu na GPU, serwowanie RAG (retrieval + model), szybkie prototypy w MLX, LoRA małych modeli przez noc. Dotychczas M6 był główną maszyną treningową. Po przeniesieniu pretreningu na karty zostaje maszyną „produktu i pomiaru”.

**Przepustowość generacji (szacunek, do zmierzenia).** Dekodowanie pojedynczego strumienia ogranicza przepustowość pamięci: tok/s ≈ η × 153 GB/s / (bajty wag na token), przy η ≈ 0,5–0,75. Dla bardzo małych modeli górę wyznacza narzut wywołań. Prefill liczy się w FLOP-ach: tok/s ≈ (3–6 TFLOPS) / 2N.

| Model w MLX | Wagi | Generacja (1 strumień) | Prefill | Odpowiedź RAG z 2k tokenów kontekstu (czas do 1. tokenu) |
|---|---:|---:|---:|---:|
| D 112M, bf16 (przez `to_llama.py` → `mlx_lm.convert`) | 0,22 GB | ~200–450 tok/s | >10 000 tok/s | <0,2 s |
| Qwen3-0.6B, bf16 | ~1,2 GB | ~65–95 tok/s | ~2 500–5 000 tok/s | ~0,4–0,8 s |
| Qwen3-0.6B, 4 bit | ~0,4 GB | ~150–280 tok/s | jw. | jw. |
| Qwen3-1.7B, 4 bit | ~1 GB | ~75–110 tok/s | ~900–1 800 tok/s | ~1–2 s |
| Qwen3-4B, 4 bit | ~2,3 GB | ~33–50 tok/s | ~375–750 tok/s | ~3–5 s |
| 7–8B, 4 bit (nauczyciel) | ~4,5 GB | ~17–26 tok/s (punkt odniesienia z repo: nauczyciel 4B ~36 tok/s) | ~190–375 tok/s | ~5–11 s |

Wniosek dla tezy Cerbeto: **rdzeń ≤0,6B daje na M6 odpowiedź RAG w ułamku sekundy do sekundy**, a model 7–8B potrzebuje kilku do kilkunastu sekund na sam prefill długiego kontekstu. To przemawia za małym rdzeniem z wiedzą z zewnątrz. Pomiar przez `mlx_lm.generate --verbose`, który drukuje osobno tok/s promptu i generacji, trzeba wykonać przed krokiem 5.

**Pamięć 16 GB.** System i aplikacje zajmują ~4–5 GB. Model 7–8B w 4 bitach razem z KV cache dla kilku tysięcy tokenów mieści się, ale nie równolegle z treningiem. Repo już raz odnotowało spowolnienie, gdy Ollama dzieliła GPU z treningiem ([0003](../specs/0003-nursery-stage-0-pretraining.md)).

**LoRA w MLX.** `mlx_lm.lora` obsługuje LoRA/DoRA. Wskazanie modelu skwantyzowanego włącza QLoRA; pamięć ograniczają `--num-layers` i `--grad-checkpoint`. Przy 1,5–2 TFLOPS efektywnie Qwen3-0.6B na 10M tokenów to ~3–5 h, a na 50M tokenów ~17–22 h. Do szybkiego sprawdzenia formatu danych i promptów wystarczy, ale serie porównawcze lepiej puszczać na 3060. [mlx-lm](https://github.com/ml-explore/mlx-lm) · [LoRA w mlx-lm (opis)](https://deepwiki.com/ml-explore/mlx-lm/4.1-lora-fine-tuning)

**Ograniczenie:** `to_llama.py` konwertuje D tylko przy K = 1. Model z pętlą po głębokości (krok 1) wymaga własnej klasy modelu w MLX, zanim da się go serwować na Macu.

## Proponowany stos

| Warstwa | 4 × RTX 3060 (Linux, CUDA) | Mac mini M6 |
|---|---|---|
| Framework | PyTorch (repo: `torch>=2.15`), bf16 autocast, SDPA (FlashAttention-2 na sm_86), `torch.compile` do sprawdzenia przed dłuższym przebiegiem | MLX + `mlx-lm`; PyTorch MPS dla zgodności z obecnym kodem |
| Uruchamianie | `torchrun --nproc_per_node=4` w jednej maszynie; między maszynami obecne `scripts/gpu/diloco.sh` | `mlx_lm.generate`, `mlx_lm.server` (API zgodne z OpenAI) |
| Pretrening D i 300M | Własny kod Cerbeto (`final_test.py`, DiLoCo) + DDP w obrębie maszyny | – |
| Bazowe porównania architektur | [nanoGPT](https://github.com/karpathy/nanoGPT): prosty DDP, łatwy do modyfikacji pod współdzielony blok | Prototyp w MLX |
| 600M–1B | [torchtitan](https://github.com/pytorch/torchtitan): FSDP2, checkpointing, `torch.compile`, rozproszone checkpointy (DCP) | – |
| LoRA/QLoRA, dostrajanie gotowych modeli | [LitGPT](https://github.com/Lightning-AI/litgpt) (LoRA, QLoRA przez bitsandbytes, FSDP) albo PEFT + TRL | `mlx_lm.lora`, `mlx_lm.fuse` |
| Generacja nauczyciela | [vLLM](https://github.com/vllm-project/vllm): 4 niezależne instancje (data parallel), model AWQ/GPTQ int4, bez tensor parallel po PCIe | Ollama (obecny nauczyciel) dla małych zestawów |
| Ewaluacja | – | `scripts/hf/bench.py`, bench1000, test RAG z kroku 5; [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) dla porównań z literaturą |
| Pomiar sprzętu | `scripts/gpu/burn.py`, `nvidia-smi topo -m`, [nccl-tests](https://github.com/NVIDIA/nccl-tests) | `mlx_lm.generate --verbose` |

## Podział ról

| Zadanie | 4 × RTX 3060 | Mac mini M6 |
|---|---|---|
| Pretrening D / 300M (dni) | ✅ jeden długi przebieg (DDP/DiLoCo) | – |
| Sweepy: dostrajanie, LoRA, porównanie architektur | ✅ 4 niezależne przebiegi | pojedyncze prototypy |
| Generowanie danych przez nauczyciela | ✅ masowe (vLLM) | małe zestawy, kontrola jakości |
| QLoRA 7–8B | ✅ 1 karta | – |
| Ewaluacja checkpointów | walidator DiLoCo (1 karta) | ✅ benchmarki, test RAG, czat |
| Serwowanie RAG i demo | – | ✅ `mlx_lm.server` + indeks |
| Prototyp nowej architektury | – | ✅ MLX, potem port na PyTorch |

## Ryzyka

1. **Topologia nieznana z repo.** Jeśli 4 karty są w kilku maszynach z 1 GbE, DDP/FSDP między maszynami odpada (all-reduce D ~2,9 s/krok). Zostaje DiLoCo. Wtedy wiersze 600M/1B są dodatkowo nierealne.
2. **Brak P2P i wąskie gniazda PCIe.** NCCL przez pamięć hosta, karty w gniazdach x4 z chipsetu, możliwe zawieszenia NCCL (`NCCL_P2P_DISABLE=1`). Przyjęte 3–6 GB/s trzeba zmierzyć.
3. **MFU 45% to cel, nie pomiar.** Zmierzone 29–32% dotyczy 112M bez `torch.compile`. Dla 600M–1B z FSDP i checkpointingiem realne może być 25–35%. Każdy wiersz powyżej 300M wymaga 200-krokowego testu przepustowości.
4. **Termika i zasilanie.** 4 × 170 W ciągle przez tydzień; repo już ma kartę z wadliwym chłodzeniem i pacer temperatury, a throttling obniża MFU.
5. **Pamięć 12 GB.** Fragmentacja, duże słowniki gotowych modeli (logity Qwen), OOM przy dłuższych sekwencjach RAG. Uwaga rośnie kwadratowo w FLOP-ach, liniowo w pamięci przy SDPA.
6. **Dane, nie GPU, ograniczają 600M–1B.** 8–30B tokenów przy 6,28B unikalnych to >4 przejścia, gdzie powtórzenia zaczynają kosztować ([Muennighoff i in.](https://arxiv.org/abs/2305.16264)).
7. **Koszt nauczyciela.** Generacja 7B zajmuje 5–26 h na wszystkich kartach i blokuje sweepy. Do tego licencje wyjść modeli nauczycieli.
8. **Mac 16 GB.** Swap przy treningu ~300M, konkurencja Ollama z treningiem, brak klasy MLX dla D z pętlą (K > 1).
9. **Zgodność numeryczna PyTorch ↔ MLX.** Wynik z MLX (kwantyzacja 4 bit) może różnić się od bf16 na CUDA. Ewaluację porównawczą prowadzić na tej samej precyzji.
10. **Zapas czasu.** Tabele pomijają ewaluację, checkpointy i restarty. Do każdego planu doliczyć 10–20%.

## Pierwsze pomiary przed krokiem 6

1. `nvidia-smi topo -m` i `all_reduce_perf -b 64M -e 1G` na 4 kartach: potwierdza topologię i rzeczywiste GB/s dla DDP/FSDP.
2. `scripts/gpu/burn.py` na każdej karcie: rzeczywiste TFLOPS bf16 po rozgrzaniu (szczyt 25,5).
3. 200 kroków D na jednej karcie: z `torch.compile` i bez, mikro-batch 4k / 8k / 16k tokenów. Daje realne MFU dla tabel.
4. 200 kroków LoRA Qwen3-0.6B (bf16) i QLoRA 8B (NF4) na jednej karcie: tok/s i szczyt pamięci.
5. Na M6: `mlx_lm.generate --verbose` dla D (po `to_llama.py`), Qwen3-0.6B bf16/4 bit i nauczyciela z promptem 2k tokenów. Uzupełnia tabelę przepustowości Maca.
