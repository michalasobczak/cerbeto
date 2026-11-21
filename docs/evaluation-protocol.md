# Protokół ewaluacji rdzenia rozumowania Cerbeto

Krok 5 analizy Cerbeto. Punktem wyjścia jest istniejący checkpoint D (106,8 mln parametrów) oraz wariant dostrojony według [strategii danych](data-strategy.md). Test ma ustalić, czy w określonych zadaniach mały model potrafi rozwiązywać nowe problemy bez wyszukiwania i korzystać z zewnętrznej wiedzy na tyle skutecznie, by przy niższym koszcie dorównać większemu modelowi odpowiadającemu z pamięci. Wniosek dotyczy badanych zadań, checkpointów i budżetów. Protokół mierzy zależność odpowiedzi od dowodu. Stwierdzenie, że w wagach jest mniej wiedzy dzięki określonemu treningowi, wymaga dodatkowego porównania dwóch treningów o jednakowym budżecie.

## Hipotezy i porównania

| Pytanie | Porównanie |
|---|---|
| Czy rdzeń generalizuje? | D po dostrojeniu w nowych zadaniach syntetycznych bez RAG i narzędzi, względem D przed dostrojeniem, małego modelu referencyjnego i prostej heurystyki dla każdej rodziny. |
| Czy wiedza z zewnątrz poprawia odpowiedzi? | Ten sam checkpoint D, ta sama lista pytań i ten sam sposób generowania. Porównać brak kontekstu, kontekst wzorcowy i kontekst odzyskany z dokumentów. |
| Czy odpowiedź rzeczywiście zależy od treści dowodu? | Pary z identycznym pytaniem i dwiema wersjami dokumentów, które uzasadniają różne odpowiedzi. Dodatkowo usunąć niezbędny dowód i zmierzyć poprawną abstencję. |
| Czy mały system ma przewagę praktyczną? | D z RAG względem gotowych małych modeli z tym samym RAG i większych modeli closed-book. Porównać jakość, energię, opóźnienie i pamięć całego systemu. |

| Model lub system | Konfiguracje |
|---|---|
| Cerbeto D przed i po dostrojeniu | Oba checkpointy closed-book, z kontekstem wzorcowym i z tym samym RAG. D po dostrojeniu przechodzi pełne ablacje. D po dostrojeniu w trybie closed-book jest główną kontrolą o identycznym rozmiarze i identycznych wagach. |
| SmolLM2-135M-Instruct i Qwen3-0.6B | Modele zamrożone, closed-book oraz z tym samym RAG i kontekstem wzorcowym. Qwen w trybie bez thinking w głównym porównaniu. [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) · [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Qwen3-1.7B i Llama 3.2 3B Instruct | Większe punkty odniesienia closed-book. Na wspólnym zbiorze dev z góry wybrać jeden jako główną referencję do bramki jakości i kosztu. Wyniki obu podać na teście. [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) · [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Proste kontrole | Heurystyki dopasowane do rodzin zadań, np. najczęstsza odpowiedź tam, gdzie ma sens, oraz dla QA ekstraktor wybierający z topowego fragmentu krótki ciąg z największym dopasowaniem leksykalnym do pytania. Solver referencyjny generuje etykiety i sprawdza poprawność. |

Jeśli krok 6 ma porównać **strategie treningu**, dodać dwa dostrojenia tego samego D z identycznego checkpointu, z tą samą liczbą tokenów, kroków i ziarnami. W kontroli zastąpić tokeny ćwiczeń użycia dowodów przykładami pytań i odpowiedzi bez kontekstu z tej samej dystrybucji treningowej. Ocenić obie wersje z RAG i bez niego. To oddziela wpływ ćwiczenia użycia dowodów od samego dostępu do dokumentów przy inferencji.

## Zbiory i punktacja

### Rozumowanie bez dostępu do zewnętrznej wiedzy

| Zbiór | Przebieg |
|---|---|
| Syntetyczny test Cerbeto | Główny test. 4 rodziny po 200 przykładów w teście ID i 200 w teście OOD na rodzinę. Arytmetyka wieloetapowa, logika relacji, Sudoku i labirynty. Osobne 100 przykładów dev na rodzinę. ID zachowuje zakres trudności treningu przy rozłącznych instancjach. OOD odkłada całe kompozycje reguł i zakresy trudności, np. dodatkowe przejście w relacji lub większą siatkę. Punktacja przez solver. Sudoku wymaga zachowania podanych cyfr i poprawności wszystkich ograniczeń, labirynt legalnej najkrótszej ścieżki według BFS. Zapisywać wynik na rodzinę, poziom trudności i makrośrednią. [Strategia danych](data-strategy.md) |
| GSM8K | Publiczny test, dokładność końcowej liczby po normalizacji. Stały wariant zadania z lm-eval-harness, identyczna liczba demonstracji dla modeli. Interpretować z uwzględnieniem języka, arytmetyki i ekspozycji na dane publiczne. [GSM8K](https://github.com/openai/grade-school-math) |
| BIG-Bench Hard | Raportować osobno podzbiór `logical_deduction_*`, `tracking_shuffled_objects_*`, `dyck_languages` i `multistep_arithmetic_two`. Dokładność wg skryptu zadania, makrośrednia po rodzinach. [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) |
| ARC-AGI-2 | Publiczne zadania ewaluacyjne, siatki w ustalonej serializacji tekstowej. Zadanie zaliczone, gdy wszystkie wyjściowe siatki są poprawne. Raportować jedną próbę i osobno ewentualne dwie, licząc także koszt drugiej. To pomiar lokalnego wariantu protokołu, a publiczne zadania mogą być znane modelom. [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) |

Podzielić dane syntetyczne przed treningiem według szablonów i rodzin reguł. Wykrywać duplikaty po kanonizacji, w tym po podstawieniu nazw, symetriach siatek i zmianach powierzchniowych tekstu. Test wygenerować z wersjonowanego generatora, przechować oddzielnie od nauczyciela i dostrajania oraz zablokować identyfikatory i hashe przed wyborem konfiguracji. Publiczne zbiory sprawdzić względem dostępnych materiałów treningowych, w tym mieszanek i promptów nauczyciela. Dla gotowych modeli ekspozycja na te zbiory ma status nieustalony, jeśli nie da się jej zweryfikować. [Procedura decontamination w lm-eval](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md)

### Wiedza z zewnątrz

Zamrozić po 400 pytań z dostępnego splitu walidacyjnego każdego zbioru na potrzeby lokalnego testu i po 100 rozłącznych na dev. Próbkować deterministycznie. Usunąć pokrycie pytań i ich parafraz między dev, testem i danymi dostrajania, także między zbiorami. W każdej głównej próbie oceniać te same pytania w trybach closed-book, open-book z retrieverem i open-book z dowodami wzorcowymi. Przypadki nieodpowiadalne raportować osobno.

| Zbiór | Przygotowanie i pomiar |
|---|---|
| Natural Questions | Użyć pytań z krótką odpowiedzią, zachować alternatywne etykiety i osobno raportować EM oraz tokenowy F1 dla krótkich odpowiedzi po ustalonej normalizacji. W trybie wzorcowym użyć tekstu strony obejmującego udokumentowany dowód. W RAG wyszukiwać w zamrożonym korpusie Wikipedii, z wersją i mapowaniem fragmentów do odpowiedzi. [Natural Questions](https://github.com/google-research-datasets/natural-questions) · [DPR i korpus Wikipedii](https://github.com/facebookresearch/DPR) |
| HotpotQA | Próbkować pytania bridge i comparison. W trybie wzorcowym podać fragmenty z oznaczonymi zdaniami dowodowymi, bez oznaczeń wskazujących odpowiedź w samym prompcie. Raportować answer EM/F1, supporting-fact F1 oraz joint F1, gdy identyfikatory zdań są mapowalne. W MVP wyszukiwać w puli akapitów całego użytego splitu, a dodatkowo można raportować oficjalny wariant distractor jako osobny wynik. [HotpotQA](https://hotpotqa.github.io/) |
| MuSiQue | MuSiQue-Ans, warstwy według 2–4 kroków. W MVP wyszukiwać w puli akapitów całego użytego splitu, a wzorcowo przekazać wszystkie wymagane ogniwa. Raportować answer EM/F1 i support F1 na poziomie akapitu. MuSiQue-Full służy dodatkowo do pomiaru abstencji na dostarczonych zestawach akapitów zgodnie z etykietami dostępności odpowiedzi w zbiorze. [MuSiQue](https://github.com/StonyBrookNLP/musique) |

Indeksować treść dokumentów i tytuły. Wykluczyć pola pytań, odpowiedzi, dekompozycji i adnotacji. Dla HotpotQA i MuSiQue wynik MVP odnosi się do zamrożonej puli benchmarku. Wyszukiwanie w pełnej Wikipedii to osobny pomiar open-domain. Przed testem sprawdzić pokrycie dowodów w każdym indeksie. Brak dowodu w korpusie liczyć w wyniku end-to-end i oznaczyć osobno w diagnostyce.

## RAG, narzędzia i ablacje

Zacząć od lokalnego BM25, np. SQLite FTS5, z ustalonym podziałem na fragmenty, identyfikatorami i deterministycznym rozstrzyganiem remisów. Porównać na dev `k = 3, 5`, po czym zamrozić k. Zapisać listy zwróconych fragmentów, a następnie dostarczyć wszystkim modelom identyczny tekst w identycznej kolejności. Dopasować wspólny budżet wejścia do najkrótszego limitu checkpointów po doliczeniu pytania, szablonu i rezerwy na odpowiedź. Przypadki, w których komplet dowodów nie mieści się w trybie wzorcowym, oznaczyć i podać ich udział. Główna różnica RAG względem closed-book obejmuje wszystkie pytania. [SQLite FTS5](https://www.sqlite.org/fts5.html)

| Warunek | Zmiana i pomiar |
|---|---|
| Bez kontekstu | Wyłączyć retriever, pamięć rozmowy i narzędzia. Użyć tego samego pytania oraz specyfikacji formatu odpowiedzi z pustą sekcją dowodów. W głównym porównaniu QA instrukcja dopuszcza odpowiedź z pamięci, gdy sekcja jest pusta, i wymaga wskazania użytych fragmentów, gdy model opiera się na dowodach. |
| Kontekst wzorcowy | Podać wszystkie wymagane fragmenty. Analizować błędy generatora tam, gdzie komplet dowodów mieści się w budżecie. |
| RAG | Zamrożone top-k. Mierzyć trafienie przynajmniej jednego i wszystkich wymaganych dowodów po przycięciu kontekstu. Warunkowy wynik przy pełnym pokryciu pozwala zlokalizować błędy generatora. |
| Szum | Utrzymać dowody i dodać podobne tematycznie dystraktory w stałym budżecie. Porównywać ze wzorcowym kontekstem uzupełnionym neutralnymi fragmentami o zbliżonej długości. Zrównoważyć pozycję dowodu. |
| Brak dowodu | Usunąć dowód jednoetapowy lub jedno konieczne ogniwo i wypełnić miejsce dystraktorem bez odpowiedzi. W eksperymencie podmiany i usuwania dowodu użyć tej samej instrukcji oparcia odpowiedzi na dostarczonych dowodach albo abstencji, niezależnie od wariantu dokumentów. |
| Podmiana dowodu | 200 par z fikcyjnymi encjami, po połowie jedno- i wieloetapowych. Zmienić fakt w dokumencie, obliczyć nową etykietę i wymagać obu poprawnych odpowiedzi wraz ze wspierającymi cytowaniami. Nad tymi samymi pytaniami wykonać kontrolę closed-book i osobny wariant z usuniętym dowodem. |
| Narzędzia | Na ustalonym podzbiorze arytmetyki i QA z obliczeniami sprawdzić układ 2 × 2: RAG włączony lub wyłączony oraz lokalny kalkulator dostępny lub niedostępny. Dozwolony jest jeden wywoływalny kalkulator z limitem dwóch wywołań. Rejestrować błędne argumenty, wynik narzędzia, poprawność końcowej odpowiedzi i koszt. Zadania syntetyczne w głównej bramce pozostają bez narzędzi. |

## Metryki i sposób pomiaru

- **Jakość odpowiedzi.** Accuracy dla zadań weryfikowanych solverem, answer EM/F1 dla QA. Ścisły parser rozróżnia odpowiedź, deklarację braku danych i identyfikatory cytowań. Zachować surową odpowiedź. Błędny format, limit czasu i nieprawidłowa abstencja otrzymują zero w metryce jakości. Raportować ich częstość osobno. Główny wynik QA to makrośrednia tokenowego F1 z trzech zbiorów.
- **Wierność kontekstowi.** Odsetek merytorycznych odpowiedzi, których wszystkie istotne twierdzenia i niezbędne ogniwa wnioskowania uzasadnia cytowany tekst. Podawać precision i recall cytowań tam, gdzie istnieją mapowalne etykiety. Poprawność odpowiedzi według klucza mierzyć osobno. W podmienionym świecie obowiązuje klucz zgodny z dowodem w danej wersji.
- **Halucynacje i abstencja.** Raportować odsetek odpowiedzi zawierających sprzeczność z dowodem lub twierdzenie bez wsparcia w trybach kontekstowych. Mianowniki to osobno wszystkie pytania i wszystkie udzielone odpowiedzi merytoryczne. W closed-book weryfikować odpowiedź i dodatkowe twierdzenia wobec etykiet lub źródeł, a brak cytowania oceniać wyłącznie w trybie wymagającym cytatów. Raportować pokrycie odpowiedzi, poprawną abstencję przy braku dowodu i nadmierną abstencję przy pełnym dowodzie.
- **Audyt ręczny.** Na tym samym losowym panelu 120 pytań QA (po 40 na zbiór) dwóch oceniających niezależnie koduje odpowiedzi porównywanych systemów w głównych trybach. Ukryć nazwę modelu i wymieszać kolejność. Rozbieżności rozstrzygnąć według z góry zapisanej rubryki, podać zgodność oceniających oraz mianowniki. Dla syntetycznych podmian użyć weryfikatora z dokładnymi etykietami i ID dowodów.
- **Koszt i opóźnienie.** Na tej samej RTX 3060 i hoście zacząć od batch size 1, po identycznym rozgrzaniu, bez cache odpowiedzi i w tej samej precyzji, jeśli modele się mieszczą. Zmierzyć p50/p95 od pytania do końca odpowiedzi, czas pierwszego tokena, czasy retrievera, prefillu, generowania i narzędzi, szczytową VRAM/RAM oraz energię hosta z GPU. Podać energię i koszt prądu na 1000 zapytań, energię na poprawną odpowiedź, koszt zbudowania indeksu i zajętość dysku. Koszt prądu to zmierzone kWh × lokalna taryfa. Na Macu powtórzyć pomiar wdrożeniowy oddzielnie, z podaniem backendu, precyzji i kwantyzacji. [Podział ról sprzętu](training-time-and-hardware-roles.md)
- **Niepewność.** Użyć 10 000 replik bootstrapu i przedziałów 95%. Różnice liczyć sparowanie na tych samych pytaniach, warstwując po zbiorze, a w wariancie syntetycznym po rodzinie i losując całe szablony. Podmiany losować jako całe pary. Dla energii i opóźnienia porównywać te same pytania z co najmniej trzema przeplatanymi powtórzeniami na konfigurację w podzbiorze pomiarowym. Jeśli mianownik jest zerowy, oznaczyć metrykę jako nieokreśloną.

## Kryteria decyzji MVP

Progi są założeniami projektowymi do zamrożenia przed testem. Pilotaż na dev ma sprawdzić format, wykonalność i liczebność. Jeśli przedziały z pilotażu wskazują, że 400 pytań na zbiór daje zbyt dużą niepewność dla marginesu 5 pp, zwiększyć próbę testową przed jej otwarciem.

| Warunek zaliczenia | Próg |
|---|---|
| Samodzielne rozumowanie | D po dostrojeniu osiąga ≥70% makroaccuracy ID i ≥50% OOD, a każda rodzina OOD osiąga ≥30%. Dolna granica 95% CI różnicy makroaccuracy OOD względem heurystyk przekracza 0 pp. |
| Wartość RAG | D z rzeczywistym retrieverem zyskuje ≥10 pp makro-F1 względem tego samego D closed-book. Dolna granica 95% CI różnicy przekracza 0 pp, a wynik punktowy rośnie na co najmniej dwóch z trzech zbiorów. |
| Użycie dowodu | ≥70% kompletnie poprawnych par podmian z cytowaniami, ≥80% poprawnych abstencji po usunięciu koniecznego dowodu oraz spadek odpowiedzi F1 pod szumem ≤10 pp względem kontroli o zbliżonej długości. |
| Jakość użytkowa | D z RAG osiąga ≥50% makro-F1 i ≥40% F1 na każdym zbiorze. Dolna granica 95% CI różnicy wobec wybranego większego modelu closed-book wynosi co najmniej −5 pp. W audycie ≥85% udzielonych odpowiedzi z RAG ma pełne wsparcie, a ≤10% zawiera twierdzenie sprzeczne z kontekstem lub pozbawione wsparcia. |
| Koszt lokalny | Na tej samej maszynie p95 end-to-end ≤5 s, a zużycie energii na 1000 zapytań ≤75% wyniku większej referencji closed-book. Oba przebiegi obejmują pełną ścieżkę odpowiedzi w swoim trybie. |

Zaliczenie wszystkich warunków wspiera tezę dla badanego MVP. Jeżeli górny kraniec CI zysku RAG leży poniżej 10 pp, test odrzuca zakładaną skalę zysku dla tej konfiguracji. Jeżeli górny kraniec CI różnicy wobec większego modelu leży poniżej −5 pp, test odrzuca założenie o porównywalnej jakości. Pozostałe niespełnione warunki blokują zaliczenie MVP. Wyniki blisko progów z szerokimi przedziałami pozostają nierozstrzygające. Wysoki wynik z kontekstem wzorcowym i niski z RAG przy słabym pokryciu dowodów wskazuje na indeks lub retriever. Niski wynik mimo pełnego pokrycia albo brak zmiany odpowiedzi po podmianie dowodu wskazuje na generator lub jego dostrojenie. Jeżeli gotowy mały model z tym samym RAG jest lepszy przy podobnym koszcie, wynik wspiera architekturę systemu i osłabia uzasadnienie własnego checkpointu.

## Uruchomienie i odtwarzalność

1. W przypiętej rewizji [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) sprawdzić identyfikatory zadań przez `lm-eval ls`. Użyć istniejących konfiguracji dla GSM8K i BBH, a dla ARC-AGI-2 i prób syntetycznych dodać lokalne YAML oraz funkcje punktacji. Przykładowe wywołanie dla modelu HF podano poniżej. [Dokumentacja CLI](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md) · [Przewodnik tworzenia zadań](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md)
2. D wczytywać przez istniejące [ładowanie checkpointu](../brain/cerbeto/model/checkpoint.py) z właściwym tokenizerem. Zaprojektować adapter generacji do harnessu i prostą pętlę Python dla par dowodów, retrievera, narzędzi i telemetrii. Użyć wspólnego formatu wyników z adaptera i pętli. Przed pełnym przebiegiem zweryfikować na dev zgodność deterministycznej generacji z bezpośrednim wywołaniem modelu.
3. Zacząć od 20 pytań na zbiór i 20 par dowodów z dev. Ręcznie sprawdzić parser, komplet dowodów po przycięciu, cytowania i izolację trybów. Następnie przejść przez dev, zamrozić konfigurację i uruchomić test. Dobierać prompty tylko na dev. Stosować jednakową treść instrukcji i demonstracji, z właściwą obudową dla każdego modelu, generację zachłanną i porównywalny limit wyjścia w ramach zadania. Główna próba QA i syntetyczna jest zero-shot. Zapisać wyniki benchmarków publicznych z dokładną konfiguracją demonstracji.
4. Manifest przebiegu zawiera commit repo i narzędzi, hash wag i tokenizera, wersje danych i indeksu, listy ID, ziarna, prompty, parametry dekodowania, sprzęt i precyzję. JSONL na pytanie zawiera ID przykładu i wariantu, treść wejścia lub odtwarzalne odwołanie, listy dowodów przed i po przycięciu, wynik narzędzia, surową i sparsowaną odpowiedź, ocenę oraz czasy i liczniki. Zapis z pomiaru energii dołączyć na poziomie serii. Raport końcowy zawiera sparowane różnice, CI, wyniki per zbiór i rodzinę, komórki ablacji oraz decyzję dla każdej bramki.

```bash
lm-eval run --model hf \
  --model_args pretrained=Qwen/Qwen3-0.6B,revision=PINNED_MODEL_SHA \
  --tasks gsm8k --device cuda:0 --batch_size 1 \
  --output_path runs/eval/qwen06-gsm8k --log_samples
```

Przykład ilustruje uruchomienie zadania GSM8K. Docelowy manifest określa także szablon promptu i tryb Qwen dla wspólnej konfiguracji. Protokół definiuje eksperyment, który trzeba wykonać, aby uzyskać wyniki.
