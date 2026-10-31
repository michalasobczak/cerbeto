# 0004 — Program nauczania (etapy 1–3) i egzaminy

Status: zaplanowana · Zależy od: 0001 · Odblokowuje: 0005

## 1. TL;DR
Definiujemy program szkoły (przedszkole, klasa 1, klasa 2) jako generatory zadań z odpowiedzią znaną programowi, z regulowaną trudnością, oraz egzaminy w trzech wariantach: znane typy, trudniejsze niż w treningu i nowe kombinacje. Dowód: testy generatorów i stała, zahashowana pula egzaminacyjna.

## 2. Abstrakt
Uczeń ma uczyć się „jak człowiek w szkole, po kolei”: bez kodowania, z matematyką wprowadzaną powoli i zawsze w kontekście języka. Priorytetem MVP jest mierzalna poprawa rozumowania na **nowych** zadaniach, dlatego prawda musi pochodzić z kodu (manifest §3), a egzaminy muszą zawierać zadania spoza rozkładu treningowego. Ustalono zakres etapów 0–3.

## 3. Cel
**Cele — program:**
| Etap | Umiejętności (generatory) |
|---|---|
| 1 — przedszkole | pytania o świat z mini-bazy wiedzy (kolory, zwierzęta i ich odgłosy, części ciała), przeciwieństwa, większe/mniejsze, kategorie („Is a cat an animal?”) |
| 2 — klasa 1 | czytanie ze zrozumieniem (historyjka z szablonu → pytanie kto/co/gdzie), liczenie obiektów do 20, dodawanie i odejmowanie w zakresie 10 |
| 3 — klasa 2 | kolejność zdarzeń, przyczyna–skutek, dodawanie/odejmowanie do 100, zadania tekstowe jednokrokowe, porównania i relacje (2–3 elementy) |

**Cele — egzaminy (dla każdej umiejętności):**
- `id` — te same typy i trudność co trening;
- `ood_hard` — większe liczby / dłuższe historyjki / więcej elementów niż w treningu;
- `ood_combo` — nowe połączenia znanych elementów (np. liczenie wewnątrz historyjki).
- Pomiar przenoszenia: wynik na egzaminach klasy N+1 przed rozpoczęciem jej nauki.

**Poza zakresem:** etap 4+, kodowanie, wiedza encyklopedyczna spoza mini-bazy.

**Kryteria sukcesu**
- Każda umiejętność ma generator z parametrem trudności i deterministycznym `seed`.
- Pule egzaminacyjne: 300 zadań na umiejętność na wariant, zamrożone i zahashowane.

## 4. Inwentarz artefaktów
| Artefakt | Rodzaj | Stan |
|---|---|---|
| `config/curriculum.yaml` (etapy, umiejętności, progi, trudność) | dane | do utworzenia |
| `brain/cerbeto/curriculum/generators/*.py` | system | do utworzenia |
| `brain/cerbeto/curriculum/world.yaml` (mini-baza wiedzy przedszkola) | dane | do utworzenia |
| `brain/cerbeto/exams/grader.py` (normalizacja i ocena odpowiedzi) | system | do utworzenia |
| `data/exams/*.jsonl` + `hashes.txt` | dane | do utworzenia |

## 5. Zmiany w domenie funkcjonalnej
**Format zadania:** `{skill, stage, difficulty, seed, context, question, answer, answer_type, steps[]}`; `steps` to rozwiązanie wzorcowe krok po kroku generowane przez kod (zalążek śladu rozumowania).

**Reguły:**
- Zadanie zalicza się, gdy znormalizowana odpowiedź ucznia (po tokenie `<|answer|>`) równa się `answer` (liczby: wartość; tekst: lemat/synonim z listy dozwolonych).
- **Zdanie klasy:** ≥ 85% na `id` i ≥ 70% na `ood_hard` dla każdej umiejętności etapu [progi `ood` DO WERYFIKACJI po pomiarze bazowym].
- **Bramka zapominania:** spadek na egzaminie którejkolwiek wcześniejszej klasy > 2 pp → wersja odrzucona.
- Ziarna egzaminacyjne z zakresu zarezerwowanego; generator treningowy odmawia ich użycia.
- Program nauczania rośnie trudnością w obrębie klasy (`difficulty` 1→5) wg wyników ucznia.

**Odporność:** zmiana generatora zmienia hash puli → egzaminy wersjonowane (`exam_set_version`), wyniki porównywalne tylko w obrębie wersji.

## 6. Ścieżka krytyczna
1. Format zadania + grader → 2. generatory klasy 1 (czytanie, liczenie) → 3. przedszkole → 4. klasa 2 → 5. zamrożenie pul egzaminacyjnych.

**Warunek ukończenia:** `cerbeto exam build` tworzy wszystkie pule; `cerbeto exam run --student <wersja>` zwraca tabelę wyników per umiejętność i wariant.

## 7. Bramki kontrolne
| # | Typ | Warunek | Weryfikacja | Blokuje |
|---|---|---|---|---|
| G1 | Unit | Każdy generator: 1000 zadań, 100% odpowiedzi zgodnych z niezależnym rozwiązaniem testowym | `pytest tests/curriculum/` | 0005 |
| G2 | Unit | Ten sam seed → identyczne zadanie | `pytest` | 0005 |
| G3 | Unit | Grader: tabela 200 przypadków brzegowych (spacje, wielkość liter, liczby słownie) | `pytest tests/test_grader.py` | 0005 |
| G4 | Integration | Rozłączność: 0 wspólnych hashy między pulą egzaminacyjną a 1M zadań treningowych | `cerbeto exam check-leak` | 0005 |
| G5 | Manual | 30 zadań na klasę przeczytanych przez człowieka — sensowne dla dziecka | przegląd | 0005 |
| N1 | Negatywna | Generator treningowy rzuca wyjątek dla ziarna egzaminacyjnego | test | zawsze |

## 8. Oczekiwany rezultat
- **Bezpośredni:** program szkoły i niezależny egzaminator w kodzie.
- **Mierzalny:** wyniki per umiejętność × wariant × wersja ucznia.
- **Skutki uboczne:** brak.
- **Wycofanie:** poprzednia wersja `curriculum.yaml` / puli egzaminacyjnej.

## 9. Informacje dodatkowe
- **Odrzucone alternatywy:** arytmetyka/kodowanie od początku (sprzeczne z „po kolei, jak w szkole”); ocena przez LLM jako główny egzaminator.
- **Dług techniczny:** mini-baza wiedzy przedszkola ręcznie spisana — mały zakres.
- **Źródła:** curriculum learning (Bengio i in., 2009); GSM8K jako wzór zadań tekstowych (Cobbe i in., 2021).
- **Pytania otwarte:** progi `ood` po pomiarze bazowym.
