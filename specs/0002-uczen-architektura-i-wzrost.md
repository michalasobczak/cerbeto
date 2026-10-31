# 0002 — Uczeń: architektura, wzrost, przycinanie, odnowa

Status: zaplanowana · Zależy od: 0001 · Odblokowuje: 0003, 0005

## 1. TL;DR
Budujemy własną sieć ucznia: transformer z obiegiem zwrotnym („myślenie w pętli”), startujący z ~8M parametrów, z operatorami wzrostu (neurogeneza), przycinania i odnowy neuronów, rosnący do maks. 30M. Dowód: testy, że wzrost nie zmienia działania sieci, i że trening zbiega na danych ze żłobka.

## 2. Abstrakt
Specyfikacja wymaga architektury eksperymentalnej, iteracyjnego rozumowania i „pełnego rozwoju” zamiast samego dotrenowywania. Ustalono poziom B odwzorowania biologii: prosty neuron, biologiczne mechanizmy rozwoju. Przy ≤30M parametrów pełny trening wszystkich wag zajmuje <1 GB pamięci, więc adaptery (LoRA) są zbędne — zmieniamy zawsze pełne wagi. Wzrost sieci w trakcie życia jest znany w literaturze (Net2Net, GradMax, stacking transformerów), ale połączenie z kontrolerem wzrostu jest elementem badawczym projektu.

## 3. Cel
**Cele**
- Model `Student` (PyTorch) z trzema częściami: *prelude* (wejście), *core* (blok rekurencyjny powtarzany K razy), *coda* (wyjście).
- Operatory wzrostu zachowujące funkcję: `grow_depth` (nowa warstwa w core), `grow_width` (szerszy MLP / więcej głowic), `grow_model_dim` (szerszy strumień rezydualny).
- Przycinanie strukturalne neuronów MLP o niskiej użyteczności; odnowa neuronów (continual backprop).
- Utrwalanie synaps: EWC (diagonalna macierz Fishera zapisywana po zdaniu klasy).
- Checkpoint: `safetensors` + `architecture.json` + `growth_log.jsonl` + stan optymalizatora.

**Poza zakresem:** neurony impulsowe, uczenie bez backpropu, MoE, szybka plastyczność w czasie rozmowy, aktywne dendryty (etap 2 projektu).

**Kryteria sukcesu**
- Start ≤ 10M parametrów, limit twardy 30M (wzrost ponad limit odrzucany).
- Po każdym operatorze wzrostu |Δ strata walidacyjna| ≤ 0,01 przed dalszym treningiem.
- Trening + inferencja ucznia 30M w szczycie ≤ 3 GB RAM.

## 4. Inwentarz artefaktów
| Artefakt | Rodzaj | Stan |
|---|---|---|
| `brain/cerbeto/model/student.py` | system | do utworzenia |
| `brain/cerbeto/model/growth.py` (wzrost, przycinanie, odnowa) | system | do utworzenia |
| `brain/cerbeto/model/consolidation.py` (EWC) | system | do utworzenia |
| `brain/cerbeto/model/checkpoint.py` | system | do utworzenia |
| `config/student.yaml` (rozmiar startowy, limit, K) | dane | do utworzenia |

## 5. Zmiany w domenie funkcjonalnej
**Architektura startowa (propozycja do potwierdzenia pomiarem):**
- słownik 8192, `d_model` 256, 8 głowic, MLP SwiGLU, RMSNorm, RoPE, kontekst 512, embeddingi wiązane;
- prelude 2 warstwy, core 2 warstwy × K iteracji (trening: K losowane 1–6; inferencja: K regulowane, „myśl dłużej”), coda 2 warstwy;
- ≈ 7–9M parametrów.

**Reguły wzrostu:**
- `grow_depth`: nowa warstwa core z wyzerowanymi projekcjami wyjściowymi (attention out, MLP down) → funkcja niezmieniona.
- `grow_width`: nowe neurony MLP z wagami wejściowymi wg GradMax (lub małe losowe) i zerowymi wyjściowymi.
- `grow_model_dim`: nowe wymiary strumienia rezydualnego wyzerowane; korekta wzmocnienia RMSNorm o √(d_new/d_old).
- Stan optymalizatora dla nowych parametrów zerowany; dla starych zachowany.
- Każdy operator dopisuje zdarzenie do `growth_log.jsonl` (co, gdzie, ile parametrów, przyczyna).

**Przycinanie i odnowa:**
- Użyteczność neuronu = średnia |aktywacja| × ‖wagi wyjściowe‖ (continual backprop, Dohare i in. 2024).
- Odnowa: co sen reinicjalizacja ≤ 1% najmniej użytecznych neuronów (dojrzałość ≥ N kroków).
- Przycinanie: po wzroście, gdy neuron ma użyteczność < progu przez 3 sny — usunięcie.

**Precyzja i urządzenie:**
- Urządzenie wybierane automatycznie: MPS → CUDA → CPU.
- Domyślnie fp32; opcjonalnie mieszana fp16 (autocast + skalowanie gradientu), jeśli przejdzie bramkę G6. **bf16 zabronione** (brak sprzętowego wsparcia na M1).
- Wagi główne i stan optymalizatora zawsze w fp32; checkpoint w fp32.

**Odporność:** każdy operator działa na kopii; przy niespełnionym warunku |Δ strata| ≤ 0,01 operacja jest odrzucana i logowana.

## 6. Ścieżka krytyczna
1. `Student` + forward/generate → 2. checkpoint save/load → 3. trening na małej próbce półki `s0_stories` → 4. `grow_depth` z testem zachowania funkcji → 5. pozostałe operatory → 6. EWC.

**Warunek ukończenia:** uczeń 8M uczony 30 min na `s0_stories` generuje rozpoznawalne angielskie zdania; testy wzrostu przechodzą.

## 7. Bramki kontrolne
| # | Typ | Warunek | Weryfikacja | Blokuje |
|---|---|---|---|---|
| G1 | Unit | save→load daje identyczne logity (max diff < 1e-6) | `pytest tests/test_checkpoint.py` | 0003 |
| G2 | Unit | Każdy operator wzrostu: max |Δ logity| < 1e-4 przed treningiem | `pytest tests/test_growth.py` | 0005 |
| G3 | Unit | Licznik parametrów po wzroście zgodny z wzorem; wzrost > 30M → wyjątek | `pytest tests/test_growth.py` | 0005 |
| G4 | Integration | 30 min treningu: strata walidacyjna spada ≥ 50% od startu | `cerbeto train --smoke` | 0003 |
| G5 | Integration | Szczyt RAM procesu przy 30M ≤ 3 GB | pomiar `psutil` w teście | 0007 |
| G6 | Integration | fp16-autocast: strata po 30 min w granicy ±2% straty fp32, bez NaN/inf; szybciej o ≥ 15% — inaczej zostaje fp32 | `cerbeto train --smoke --precision fp16` | wybór precyzji w 0003 |
| N1 | Negatywna | Model nie ładuje żadnych zewnętrznych wag ani tokenizera | test sprawdza brak pobrań sieciowych | zawsze |
| N2 | Negatywna | Żaden tensor modelu ani optymalizatora nie ma typu bf16 | test | zawsze |

## 8. Oczekiwany rezultat
- **Bezpośredni:** uczeń, który potrafi rosnąć, przycinać się i odnawiać.
- **Mierzalny:** liczba parametrów i log wzrostu w każdej wersji.
- **Skutki uboczne:** rekurencja zwiększa koszt obliczeń K-krotnie dla core.
- **Wycofanie:** checkpoint sprzed operacji (manifest §4).

## 9. Informacje dodatkowe
- **Odrzucone alternatywy:** adaptery LoRA (zbędne przy 30M); MLX jako backend treningu (tylko Apple Silicon; wagi w safetensors zostawiają drogę do inferencji w MLX); sieci impulsowe; bf16 (brak wsparcia sprzętowego na M1); Core ML / Neural Engine — odłożone (NPU nie obsługuje treningu; inferencja przez Core ML wymagałaby rekonwersji po każdym wzroście i rozwinięcia pętli do stałego K — do ponownej oceny po 0005).
- **Sprzęt referencyjny:** MacBook Pro Apple M1 Pro, 32 GB; zmierzone (PyTorch 2.15, MPS, matmul 4096²): CPU fp32 1,5 TFLOPS, GPU fp32 3,6, fp16 4,2, bf16 2,1 TFLOPS.
- **Dług techniczny:** GradMax może zostać uproszczony do małej inicjalizacji losowej w pierwszej wersji.
- **Źródła:** Net2Net (Chen i in., 2015); GradMax (Evci i in., 2022); Stacking Your Transformers (Du i in., 2024); Loss of plasticity in deep continual learning (Dohare i in., Nature 2024); EWC (Kirkpatrick i in., 2017); recurrent-depth transformers (Geiping i in., 2025).
- **Pytania otwarte:** kolejność operatorów wzrostu (głębia vs szerokość) — rozstrzyga eksperyment w 0005.
