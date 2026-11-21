# Cerbeto — analiza rdzenia rozumowania i plan MVP

## Streszczenie

**Uruchomić wariant A w trzy tygodnie.** Porównać istniejący Cerbeto D z zamrożonym Qwen3-0.6B na wspólnym lokalnym RAG i kalkulatorze. Dostroić D na 50 mln tokenów, a Qwena przez LoRA w bf16 na 25 mln. Główny test sprawdzi, czy D rozwiązuje nowe zadania bez narzędzi, poprawnie reaguje na podmianę i usunięcie dowodu oraz z RAG osiąga jakość porównywalną z większym modelem closed-book przy niższym koszcie energetycznym.

**Wariant B** uruchomić, jeśli poprawnie dostarczone dowody nadal sprawiają trudność po dostrojeniu. Zacząć od kontrolowanego eksperymentu przy 106,8 mln parametrów. **Wariant C** prowadzić jako niezależny eksperyment z TRM na syntetycznych zadaniach o sprawdzalnych rozwiązaniach. Jego wynik może uzasadnić włączenie solvera jako narzędzia Cerbeto.

## Kontekst i teza

Cerbeto ma działać lokalnie. Model ma interpretować instrukcje, składać przesłanki, zlecać obliczenia i oceniać, czy dowody wystarczają do odpowiedzi. Fakty podlegające aktualizacji dostarcza indeks dokumentów lub narzędzie. Podmiana faktu przy niezmienionym pytaniu pozwoli sprawdzić, czy odpowiedź rzeczywiście zależy od dowodu. Test z odłożonymi kombinacjami reguł sprawdzi generalizację procedur.

Punktem wyjścia jest rodzina D, pierwotnie 106,8 mln parametrów. Około 2 mld przetworzonych tokenów dotyczy wcześniejszego etapu. README wskazuje dziś `bil10` jako domyślny checkpoint (około 12,5 mld tokenów łącznie), a specyfikacja opisuje zaplanowane 20 mld kolejnych tokenów w `bil20` (około 32,5 mld łącznie po ukończeniu). Raport sprzętowy opisuje późniejszą konfigurację D jako około 112 mln parametrów ze słownikiem 16 tys. Rachunki dla treningu od zera zachowują referencyjne 106,8 mln. Na dev porównać `bil2-avg`, `bil10` i, jeśli dostępny, `v6`. Zapisać hash wybranego checkpointu, tokenizer, rozmiar i limit kontekstu. Historyczne testy rozmów służą do opisu stanu projektu. [README](../README.md) · [Historia D](background.md) · [bil10](../specs/0016-ten-billion.md) · [bil20](../specs/0018-twenty-billion.md)

## Architektury

| Rodzina | Rola w Cerbeto |
|---|---|
| Gęsty Transformer | D jako kontrola własnych wag, Qwen3-0.6B jako referencja gotowego modelu. Pełna uwaga pozwala zestawiać i cytować odległe fragmenty, a koszt części uwagowej rośnie kwadratowo z długością podczas prefillu. |
| Transformer ze współdzielonym blokiem, Universal Transformer | Pierwszy eksperyment architektoniczny w B. Wspólne wagi zmniejszają liczbę parametrów, a kolejne zastosowania bloku zwiększają koszt obliczeń. Porównywać przy podobnych FLOP i raportować p95 oraz liczbę iteracji. |
| Mamba-2 i RWKV-7 | Stan o stałym rozmiarze względem długości historii sprzyja strumieniowaniu. Zmierzyć przy długim kontekście odczyt konkretnego faktu, cytowania i odporność na dystraktory. Uwzględnić koszt integracji PyTorch i MLX. |
| Hybrydy Mamba–uwaga | Zamba/Zamba2 używają globalnej uwagi, Samba uwagi lokalnej, a Jamba łączy stan, uwagę i ekspertów. Własna prosta hybryda z nielicznymi warstwami globalnej uwagi jest zasadna, jeśli pomiar wykaże, że kontekst ogranicza RAG. |
| HRM i TRM | Iteracyjnie poprawiają odpowiedź w zadaniach o ustrukturyzowanym stanie. MVP ograniczyć do Sudoku i labiryntów z odłożonymi rozmiarami lub regułami. Wyniki na siatkach nie rozstrzygają o jakości tekstowego RAG. |
| Rzadki MoE | Mniej aktywnych wag na token przy większej całkowitej pojemności. Pamięć ekspertów, routing i komunikacja pogarszają atrakcyjność na 12 GB na kartę i przy braku NVLinka. Odłożyć. |

Macierz 11 architektur, wyniki poniżej 1 mld parametrów i źródła eksperymentów zawiera [przegląd architektur](architecture-review.md).

## Rozmiar, budżet i czas treningu

Dla pełnego treningu gęstego modelu przyjęto około 6 × parametry × przetworzone tokeny FLOP części macierzowej, a dla małego adaptera około 4 × parametry × tokeny. Przedziały pomijają m.in. osobny koszt uwagi, ewaluacji i zapisu. Orientacyjne 20 tokenów na parametr z Chinchilli daje 2,14 mld dla D i 6 mld dla 300M. To punkt odniesienia dla pretreningu językowego. [Budżet treningu](model-size-and-training-budget.md) · [Chinchilla](https://arxiv.org/abs/2203.15556)

| Przebieg | Przetworzone tokeny | FLOP | Czas jednego przebiegu |
|---|---:|---:|---:|
| D 106,8M od zera | 1–3 mld | 6,41 × 10¹⁷–1,92 × 10¹⁸ | 4–18 h na 4 GPU |
| Transformer 300M od zera | 4–10 mld | 7,2 × 10¹⁸–1,8 × 10¹⁹ | 45–177 h na 4 GPU (1,9–7,4 dnia) |
| Transformer 600M od zera | 8–20 mld | 2,88 × 10¹⁹–7,2 × 10¹⁹ | 8–33 dni na 4 GPU |
| Transformer 1B od zera | 15–30 mld | 9,0 × 10¹⁹–1,8 × 10²⁰ | 28–112 dni na 4 GPU |
| Pełne dostrojenie istniejącego D | 20–200 mln | 1,28 × 10¹⁶–1,28 × 10¹⁷ | 0,3–4,6 h na 1 GPU |
| LoRA Qwen3-0.6B | 10–50 mln | 2,4 × 10¹⁶–1,2 × 10¹⁷ | 0,75–6,5 h na 1 GPU |
| Nauczyciel 7B, 50 tys. zadań, po 2 próby | 102,4 mln (wejście i wyjście) | 1,43 × 10¹⁸ | 5–26 h na 4 niezależnych GPU |

Czasy są szacunkami z kroku 3. Dodać 10–20% rezerwy na zapisy, ewaluację i restarty oraz uwzględnić niezależnie liczbę prób. W przykładzie pełnej destylacji 7B uczeń D zużywa około 6,56 × 10¹⁶ FLOP, więc większość obliczeń przypada na generowanie nadzoru. Ten przykład liczy 512 tokenów wejścia i 512 wyjścia w każdej próbie. W A skala generowania zależy od udziału śladów, ich długości i odsetka zaakceptowanych odpowiedzi. [Czasy i założenia sprzętowe](training-time-and-hardware-roles.md) · [Budżet destylacji](model-size-and-training-budget.md)

### Sprzęt i koszty

- Cztery RTX 3060 mają po 12 GB VRAM bez NVLinka. Krok 3 zakłada 25,5 TFLOPS bf16 na kartę i 30–45% MFU. Punkt odniesienia to około 33 tys. tok/s w konfiguracji D 112M z trzema workerami DiLoCo, około 29% MFU. Górny kraniec 45% jest celem do sprawdzenia. Czasy 4 GPU zależą od topologii, rozmiaru partii, długości sekwencji i temperatury.
- W MVP uruchamiać niezależne próby na poszczególnych kartach. Dla B stosować DDP w jednym hoście po pomiarze NCCL i sprawdzeniu VRAM. Synchronizować raz na krok optymalizatora przy akumulacji gradientów. Przy hostach połączonych przez 1 GbE korzystać z istniejącego DiLoCo i przeliczyć czas według pomiaru. Dla 600M rozważyć FSDP2 lub 8-bitowy AdamW, a dla 1B FSDP2 z checkpointingiem aktywacji. Obie skale wypadają z MVP.
- Mac mini M6 obsługuje indeks, ewaluację i lokalną inferencję w MLX. Sprawdzić faktyczną pamięć, konwersję D oraz wpływ kwantyzacji. Dla Qwen3-0.6B krok 3 szacuje dekodowanie na 65–95 tok/s w bf16 i 150–280 tok/s w 4 bitach, a dla 7–8B w 4 bitach na 17–26 tok/s. To szacunki dekodowania, a pomiar odpowiedzi obejmie także wyszukanie, prefill, narzędzia i generację.
- W pierwszym tygodniu wykonać `nvidia-smi topo -m`, `nccl-tests` dla wspólnego hosta, po 200 kroków D i LoRA Qwen na jednej karcie, pilotaż nauczyciela na 200 zadaniach, jeśli będzie użyty, oraz pomiar prefillu, generacji, pamięci i opóźnienia na Macu. [Raport sprzętowy](training-time-and-hardware-roles.md) · [DiLoCo w Cerbeto](../specs/0015-diloco.md) · [Apple](https://www.apple.com/mac-mini/specs/) · [MLX LM](https://github.com/ml-explore/mlx-lm)

Koszt poniżej podano w kartogodzinach oraz orientacyjnych kWh samych GPU przy założeniu 170 W na aktywną kartę. To przeliczenie planistyczne, bez narzutu czasu i poboru hosta. Koszt w PLN wyliczyć ze zmierzonych kWh całej maszyny i lokalnej stawki PLN/kWh. Budżet zakłada dostępny sprzęt i lokalne modele.

## Dane

| Mieszanka | Tokeny | Priorytety |
|---|---:|---|
| Pełne SFT D | 50 mln | 19 mln RAG i narzędzia, 21 mln własne zadania weryfikowane automatycznie, 5 mln krótkie zweryfikowane ślady, 5 mln język i otwarte dane matematyczne. |
| LoRA Qwen | 25 mln | Te same proporcje i rodziny przykładów. Porównywać także liczbę przykładów, ponieważ modele mają różne tokenizery. |
| D od zera | 2 mld | Mieszanka z kroku 4. 500 mln tokenów języka i formatu oraz 1,5 mld zadań, dowodów, narzędzi i pozostałych składników. Liczyć oddzielnie tokeny unikalne i przetworzone. |

Curriculum obejmuje format i pojedyncze operacje, składanie kroków oraz trudniejsze przypadki z szumem i brakującym dowodem (25%, 45%, 30% tokenów). Generatory obejmują arytmetykę, logikę, siatki, Sudoku, planowanie i programy. Każdy przykład ma weryfikator lub sprawdzalne przesłanki. Ślad nauczyciela musi przejść kontrolę wyniku i istotnych kroków. Zestawić wariant bez śladów z wariantem z krótkimi śladami przy podobnym budżecie, korzystając najpierw z rozwiązań solvera i próbki z lokalnego nauczyciela 7B. Rozdzielić zbiory po rodzinach reguł przed generacją, zablokować test przed wyborem konfiguracji, usuwać duplikaty po kanonizacji i prowadzić manifest licencji i pochodzenia. [Strategia danych](data-strategy.md)

## Ewaluacja i decyzja

Porównać D przed i po SFT, zamrożonego i dostrojonego Qwena, SmolLM2-135M jako małą referencję i większe modele Qwen3-1.7B oraz Llama 3.2 3B closed-book. Jedną większą referencję wybrać na dev. Dla RAG zacząć od lokalnego BM25 w SQLite FTS5, wybrać na dev top-3 lub top-5 i dostarczać identyczne fragmenty wszystkim modelom. Główne konfiguracje to closed-book, dowód wzorcowy, rzeczywisty RAG, szum, brak i podmiana dowodu. Osobno porównać RAG i kalkulator w układzie 2 × 2.

Ocenić cztery rodziny syntetyczne (arytmetyka, logika, Sudoku, labirynty), każdą na 200 przykładach ID i 200 OOD. Dodać 200 par z podmienionym dowodem oraz po 400 pytań z Natural Questions, HotpotQA i MuSiQue. GSM8K, podzbiory BIG-Bench Hard i ARC-AGI-2 są dodatkowymi odniesieniami. Raportować dokładność, EM/F1, pokrycie dowodów, poprawność cytowań i wywołań, błędy, abstencję, czas pierwszego tokena, p50/p95 end-to-end, RAM/VRAM i energię na 1000 zapytań. Porównania sparować, podać 95% przedziały ufności i zachować manifest oraz rekordy przebiegu. [Protokół ewaluacji](evaluation-protocol.md)

| Bramka dla dostrojonego D | Próg z kroku 5 |
|---|---|
| Rozumowanie | Makrodokładność ≥70% ID i ≥50% OOD, każda rodzina OOD ≥30%, dolna granica 95% CI przewagi OOD nad heurystyką >0 pp. |
| Wartość RAG | Zysk ≥10 pp makro-F1 wobec D closed-book, dolna granica 95% CI zysku >0 pp, wzrost na co najmniej dwóch z trzech zbiorów. |
| Użycie dowodu | ≥70% w pełni poprawnych par z cytowaniami, ≥80% poprawnych abstencji po usunięciu dowodu, spadek F1 pod szumem ≤10 pp. |
| Jakość użytkowa | ≥50% makro-F1 i ≥40% F1 na każdym zbiorze. Dolna granica 95% CI różnicy wobec większej referencji ≥−5 pp. W audycie ≥85% odpowiedzi w pełni wspartych dowodem i ≤10% z twierdzeniem sprzecznym lub bez oparcia. |
| Koszt lokalny | p95 end-to-end ≤5 s i energia na 1000 zapytań ≤75% wyniku większej referencji, na tej samej maszynie. |

Zamrozić progi, próbki i ustawienia przed testem. Jeśli kontekst wzorcowy pomaga, a RAG gubi potrzebne fragmenty, poprawić indeks i dobór kontekstu. Jeśli przy kompletnym dowodzie D przegrywa z małym Qwenem, wariant A nadal testuje system z zewnętrzną wiedzą, a poparcie dla własnego checkpointu wymaga osobnego wyniku. Jeśli nawet Qwen zawodzi przy pełnym dowodzie, sprawdzić format i trudność na dev. Zaliczenie wszystkich bramek wspiera tezę w badanym zakresie.

## Rekomendowane warianty i plany MVP

### A. Istniejący D i mały model gotowy z lokalnym RAG — 3 tygodnie

**Koszt.** Przy budżetach 50 mln i 25 mln tokenów odpowiednio około 3,20 × 10¹⁶ FLOP dla D i 6,0 × 10¹⁶ dla LoRA. Pełne przedziały z kroku 3 dają łącznie 1,05–11,1 kartogodz. dla pary przebiegów (około 0,18–1,89 kWh GPU), a równoległe próby mogą skrócić czas kalendarzowy. Pełny przykład generowania nauczyciela zużywa dodatkowo 20–104 kartogodz. (około 3,4–17,7 kWh GPU). Zacząć od LoRA w bf16, a QLoRA sprawdzić, jeśli pamięć ogranicza wymagany kontekst lub partię.

1. **Tydzień 1:** zbudować adapter wspólnego interfejsu, BM25, kalkulator i parser cytowań. Wykonać pomiary sprzętu i pilotaż dev na 20 pytaniach na zbiór i 20 parach. Uruchomić zamrożone referencje.
2. **Tydzień 2:** wygenerować i zweryfikować mieszankę, dostroić D i Qwena, porównać krótkie ślady z samą odpowiedzią i sprawdzić regresję formatu oraz zadań bez narzędzi.
3. **Tydzień 3:** zamrozić ustawienia, uruchomić test i audyt odpowiedzi, zmierzyć energię oraz opóźnienie na RTX 3060 i osobno wdrożenie na Macu. Złożyć wyniki względem pięciu bramek.

**Ryzyka.** Krótki kontekst D może uciąć przesłanki, nauczyciel może generować błędne kroki, a dostrojenie może pogorszyć zdolności modelu. Diagnostykę oprzeć na trybie wzorcowym, weryfikacji śladów i porównaniu z zamrożonym D.

### B. Własny Transformer 106,8–300M od zera — 4 tygodnie

**Koszt.** Bieg kontrolny D na 2 mld tokenów to około 1,28 × 10¹⁸ FLOP. Zakres referencyjny 1–3 mld to 4–18 h na 4 GPU, czyli 16–72 kartogodz. i około 2,7–12,2 kWh GPU. Opcjonalne 300M wymaga 45–177 h na 4 GPU, czyli 180–708 kartogodz. i około 30,6–120,4 kWh GPU. Każda dodatkowa próba osobno zużywa swój budżet pretreningu i dostrajania.

1. **Tydzień 1:** potwierdzić, gdzie A zawodzi przy pełnym dowodzie. Sprawdzić topologię, korpus, throughput i limity pamięci. Przygotować gęstą kontrolę i prototyp współdzielonego bloku.
2. **Tydzień 2:** wykonać krótkie próby na tej samej mieszance i długości kontekstu. Zachować identyczną punktację, liczyć wszystkie zastosowania bloku i porównać przy podobnych FLOP.
3. **Tydzień 3:** uruchomić gęsty przebieg 106,8M na 2 mld tokenów. Rozważyć 300M od dolnego krańca 4 mld tokenów, jeśli pomiar pojemności, VRAM i czasu to uzasadnia. Zestawić wyniki z istniejącym D oraz rozważyć opisane w README powiększenie checkpointu jako kontrolę.
4. **Tydzień 4:** dostroić wybrany model, powtórzyć kluczowe ablacje A i porównać makro-F1, OOD, p95 i koszt. Próbę hybrydy Mamba–uwaga dopuścić, jeśli w A długi kontekst był zmierzonym ograniczeniem.

**Ryzyka.** Górna granica 300M konkuruje z czasem na ewaluację, a liczba tokenów może przekroczyć zasób unikalnych danych. W nowej półce treningowej z kroku 3 jest 6,28 mld unikalnych tokenów. Jeśli topologia ogranicza przepustowość lub jakość nie rośnie na dev, zakończyć MVP na skali D.

### C. TRM jako eksperymentalny solver — 2–3 tygodnie

**Koszt.** Limit pilotażu to 24 kartogodz. na jednej RTX 3060 (około 4,1 kWh GPU). To decyzja planistyczna dla prototypu, a czas i FLOP zmierzyć po krótkim przebiegu. Zacząć od konfiguracji rzędu 7 mln parametrów według konwencji publikacji TRM i policzyć wszystkie uczone wagi, w tym ewentualne embeddingi zadań. HRM rzędu 27 mln pozostaje możliwą kontynuacją.

1. **Tydzień 1:** uruchomić generatory Sudoku i labiryntów, solvery do punktacji oraz mały Transformer jako kontrolę. Rozdzielić instancje i odłożyć całe klasy trudności na OOD.
2. **Tydzień 2:** porównać jedną aktualizację z kilkoma aktualizacjami stanu oraz z kontrolą przy podobnym koszcie. Zmierzyć poprawność, FLOP i czas na zadanie, przy jawnej liczbie iteracji i prób.
3. **Opcjonalny tydzień 3:** powtórzyć wybraną próbę na drugim ziarnie i zmierzyć p95. Przy powtarzalnej przewadze OOD w budżecie obliczeń podłączyć solver jako narzędzie i zmierzyć poprawność zlecania zadań przez D.

**Ryzyka.** Przewaga może zależeć od augmentacji, wielokrotnych prób, przecieku szablonów lub kosztownej iteracji. Zachować identyczne dane i budżet prób oraz ujawnić koszt pełnego przebiegu. Wnioski ograniczyć do badanych rodzin zadań.

## Ryzyka i otwarte pytania

| Pytanie | Jak rozstrzygnąć |
|---|---|
| Czy karty działają w jednym hoście, czy przez 1 GbE? Czy jest PCIe P2P i stabilne chłodzenie? | Zmapować hosty, sprawdzić NCCL i temperatury, zmierzyć pojedynczą kartę i konfigurację docelową przed zatwierdzeniem B. |
| Czy D wykorzystuje pełny dowód i utrzymuje format w polskim oraz angielskim? | Zmierzyć użyteczną długość kontekstu i dodać odłożony panel PL na pytaniach syntetycznych i źródłach lokalnych. Główne progi protokołu zachować dla zamrożonego zestawu. |
| Czy potrzeba większego modelu, lepszych danych czy lepszego retrievera? | Porównać dowód wzorcowy, faktyczne pokrycie dowodów i interwencje na parach. Obserwować błędy D i Qwena na tych samych pytaniach. |
| Jaki jest koszt nadzoru i ryzyko przecieku? | Zapisać przepustowość i odsetek przyjętych śladów, pochodzenie i licencje oraz deduplikację według rodzin, również dla promptów nauczyciela. |
| Czy jakość utrzymuje się po kwantyzacji i integracji z MLX? | Porównać przed i po konwersji na tych samych pytaniach, z kontrolą pamięci, energii i opóźnienia całej ścieżki. |

## Źródła

### Repozytorium i raporty cząstkowe

- [Stan projektu i lista checkpointów](../README.md) · [Historia i założenia](background.md) · [Dziennik zmian](../CHANGELOG.md)
- [DiLoCo](../specs/0015-diloco.md) · [bil10](../specs/0016-ten-billion.md) · [talk10](../specs/0017-talk10.md) · [bil20](../specs/0018-twenty-billion.md)
- [1. Przegląd architektur](architecture-review.md)
- [2. Rozmiar i budżet treningu](model-size-and-training-budget.md)
- [3. Czas treningu i podział ról sprzętu](training-time-and-hardware-roles.md)
- [4. Strategia danych](data-strategy.md)
- [5. Protokół ewaluacji](evaluation-protocol.md)

### Źródła zewnętrzne

Wykaz obejmuje odnośniki z pięciu raportów cząstkowych. Każdy adres występuje raz w wykazie, w grupie pierwszego raportu, który go przytacza.

#### Architektury

- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)
- [SmolLM2 arXiv](https://arxiv.org/abs/2502.02737)
- [135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [SSD](https://arxiv.org/abs/2405.21060)
- [Kod i wagi](https://github.com/state-spaces/mamba)
- [Tabela wyników](https://openreview.net/attachment?id=5kKdgokvCj&name=pdf)
- [MLX](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mamba2.py)
- [Publikacja](https://arxiv.org/abs/2503.14456)
- [Tabela COLM](https://openreview.net/pdf/74011cffc1082ac9394afde92802c810c51baabf.pdf)
- [Kod](https://github.com/RWKV/RWKV-LM)
- [MLX-LM #580](https://github.com/ml-explore/mlx-lm/pull/580)
- [Jamba](https://arxiv.org/abs/2403.19887)
- [Jamba 1.5](https://arxiv.org/abs/2408.12570)
- [API](https://aws.amazon.com/marketplace/pp/prodview-fnof2toffbfbg)
- [MLX](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/jamba.py)
- [Zamba](https://arxiv.org/abs/2405.16712)
- [Zamba2](https://arxiv.org/abs/2411.15242)
- [Kod](https://github.com/Zyphra/Zamba2)
- [ArXiv](https://arxiv.org/abs/2406.07522)
- [Kod](https://github.com/microsoft/Samba)
- [HRM](https://arxiv.org/abs/2506.21734)
- [Kod](https://github.com/sapientinc/HRM)
- [Ablacja](https://arcprize.org/blog/hrm-analysis)
- [Leaderboard](https://arcprize.org/leaderboard/community)
- [HRM-Text](https://arxiv.org/abs/2605.20613)
- [Kod HRM-Text](https://github.com/sapientinc/HRM-Text)
- [HRM-Text, tabela 2](https://arxiv.org/html/2605.20613)
- [ArXiv](https://arxiv.org/abs/2510.04871)
- [Kod](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- [ArXiv](https://arxiv.org/abs/1807.03819)
- [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
- [Looped Transformers](https://arxiv.org/abs/2502.17416)
- [RecursiveGPT](https://arxiv.org/abs/2608.26973)
- [Kod MoR](https://github.com/raymin0223/mixture_of_recursions)
- [Huginn](https://arxiv.org/abs/2502.05171)
- [Kod Huginn](https://github.com/seal-rg/recurrent-pretraining)
- [Mikro-MoE](https://arxiv.org/abs/2605.13769)
- [Switch](https://arxiv.org/abs/2101.03961)
- [MobileMoE](https://arxiv.org/abs/2605.27358)
- [OLMoE](https://arxiv.org/abs/2409.02060)
- [BFCL](https://gorilla.cs.berkeley.edu/leaderboard)
- [RULER](https://arxiv.org/abs/2404.06654)
- [Zoology](https://arxiv.org/abs/2312.04927)
- [RAG](https://arxiv.org/abs/2005.11401)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)

#### Rozmiar i budżet

- [Chinchilla](https://arxiv.org/abs/2203.15556)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [Beyond Chinchilla-Optimal](https://proceedings.mlr.press/v235/sardana24a.html)
- [TinyLlama](https://arxiv.org/abs/2401.02385)
- [TinyLlama](https://arxiv.org/html/2401.02385v2)
- [SmolLM](https://huggingface.co/blog/smollm)
- [SmolLM2](https://arxiv.org/html/2502.02737)
- [MobileLLM](https://arxiv.org/html/2402.14905v2)
- [Phi-1](https://arxiv.org/pdf/2306.11644)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Rodzina Qwen2.5](https://qwenlm.github.io/blog/qwen2.5-llm/)
- [0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- [Karta Meta](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [Destylacja sekwencyjna](https://arxiv.org/abs/1606.07947)

#### Sprzęt i oprogramowanie

- [Specyfikacja 3060](https://www.techspot.com/specs/gpu/223291-nvidia-geforce-rtx-3060-12gb.html)
- [GA102 whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf)
- [Korthikanti i in.](https://arxiv.org/abs/2205.05198)
- [Tom's Hardware](https://www.tomshardware.com/news/nvidia-confirms-geforce-cards-lack-p2p-support)
- [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
- [Apple – specyfikacja](https://www.apple.com/mac-mini/specs/)
- [9to5Mac](https://9to5mac.com/2026/08/27/m6-mac-mini-three-things-apple-didnt-highlight-in-the-announcement/)
- [vLLM](https://github.com/vllm-project/vllm)
- [Qwen2.5-7B config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)
- [ZeRO](https://arxiv.org/abs/1910.02054)
- [FSDP2 `fully_shard`](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [torchtitan – FSDP](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
- [nccl-tests](https://github.com/NVIDIA/nccl-tests)
- [LoRA w mlx-lm (opis)](https://deepwiki.com/ml-explore/mlx-lm/4.1-lora-fine-tuning)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [torchtitan](https://github.com/pytorch/torchtitan)
- [LitGPT](https://github.com/Lightning-AI/litgpt)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Muennighoff i in.](https://arxiv.org/abs/2305.16264)

#### Dane i licencje

- [ARC-AGI-1](https://github.com/fchollet/ARC-AGI)
- [OR-Tools](https://github.com/google/or-tools)
- [ALCE](https://arxiv.org/abs/2305.14627)
- [SQuAD v2](https://aclanthology.org/P18-2124/)
- [Karta xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
- [Karta nauczyciela](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- [Zbiór](https://huggingface.co/datasets/roneneldan/TinyStories)
- [Umowa CDLA](https://cdla.dev/sharing-1-0/)
- [Publikacja](https://arxiv.org/abs/2305.07759)
- [Zbiór](https://github.com/openai/grade-school-math)
- [Licencja](https://github.com/openai/grade-school-math/blob/master/LICENSE)
- [Karta NVIDIA](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
- [Sekcja licencji DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1#7-license)
- [NuminaMath 1.5](https://huggingface.co/datasets/project-numina/NuminaMath-1.5)
- [SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2)
- [OpenAI Services Agreement](https://openai.com/policies/services-agreement/)

#### Ewaluacja

- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)
- [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2)
- [Procedura decontamination w lm-eval](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md)
- [Natural Questions](https://github.com/google-research-datasets/natural-questions)
- [DPR i korpus Wikipedii](https://github.com/facebookresearch/DPR)
- [HotpotQA](https://hotpotqa.github.io/)
- [MuSiQue](https://github.com/StonyBrookNLP/musique)
- [SQLite FTS5](https://www.sqlite.org/fts5.html)
- [Dokumentacja CLI](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)
- [Przewodnik tworzenia zadań](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md)
