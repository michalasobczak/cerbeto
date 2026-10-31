# 0005 — Szkoła: dzień, sen, egzamin, wzrost

Status: zaplanowana · Zależy od: 0002, 0003, 0004 · Odblokowuje: 0006, 0007

## 1. TL;DR
Automatyczna pętla nauki: **dzień** (lekcje z dużym LLM-nauczycielem, zapis przeżyć w hipokampie) → **sen** (konsolidacja do wag z powtórkami i utrwalaniem) → **egzamin** (bramki) → awans wersji / cofnięcie / wzrost. Dowód: po N cyklach wynik ucznia na egzaminach `ood` klasy rośnie, a zapominanie mieści się w 2 pp.

## 2. Abstrakt
To jest rdzeń priorytetu MVP: mierzalna poprawa rozumowania na nowych zadaniach po automatycznych cyklach nauki z nauczycielem. Nauczycielem jest **duży LLM uruchomiony lokalnie** (Ollama, endpoint zgodny z OpenAI), dzięki czemu nauka działa offline i bez kosztów API; interfejs pozwala dopiąć zewnętrznych dostawców. Rytm dzień/sen odwzorowuje komplementarne systemy uczenia (hipokamp + kora, McClelland 1995). Człowiek rozmawiający z uczniem jest dodatkowym nauczycielem, ale lekcje prowadzi LLM.

## 3. Cel
**Cele**
- **Nauczyciele** z rolami: *wychowawca* (prowadzi lekcję, zadaje pytania, stopniuje trudność), *korepetytor* (tłumaczy błąd), *egzaminator pomocniczy* (ocenia odpowiedzi otwarte; nigdy nie widzi pul egzaminacyjnych). Konfiguracja: dostawca, endpoint, model, rola, instrukcje, harmonogram, limity (tokeny/dzień, koszt/dzień dla dostawców płatnych).
- **Egzamin dla nauczyciela** przed dopuszczeniem: rozwiązuje zadania klas 1–3 i generuje wyjaśnienia; wymagane ≥ 98% poprawnych odpowiedzi i ≥ 95% wyjaśnień zgodnych z odpowiedzią z generatora.
- **Lekcja:** generator daje zadanie + prawdę → wychowawca formułuje je naturalnie (warianty językowe) → uczeń odpowiada (z `<|think|>`) → kod ocenia → przy błędzie korepetytor tłumaczy, uczeń próbuje ponownie → zapis przeżycia.
- **Hipokamp:** baza przeżyć (SQLite) z pochodzeniem (`teacher|human`), stanem weryfikacji, wynikiem, zaskoczeniem (strata ucznia).
- **Sen:** trening na zweryfikowanych przeżyciach dnia + powtórki (bibliotekа + wcześniejsze klasy, domyślnie 50/50) + kara EWC; odnowa neuronów; opcjonalnie przycinanie.
- **Egzamin po śnie** + bramki 0004 → awans lub automatyczne cofnięcie.
- **Kontroler wzrostu:** brak poprawy ≥ 1 pp na egzaminie bieżącej klasy przez 3 cykle i wynik poniżej progu zdania → operator wzrostu (0002) → nowa wersja.
- **Rytm:** automatyczny; ustawienia: lekcji na dzień, start snu (po N lekcjach lub o godzinie), pauza, „ucz teraz”, „śpij teraz”.
- **Uczeń kontrolny** przechodzi ten sam program bez wzrostu.

**Poza zakresem:** szybka plastyczność w rozmowie, dendryty, RL z sygnałem nagrody (etap 2 projektu), dzielenie się wiedzą między kopiami.

**Kryteria sukcesu (priorytet MVP)**
- Uczeń rosnący zdaje klasy 1–3 według progów 0004.
- Wynik `ood_hard` + `ood_combo` każdej klasy rośnie o ≥ 20 pp względem pomiaru bazowego (0003).
- Zapominanie: żadna wcześniejsza klasa nie traci > 2 pp w zaakceptowanej wersji.
- Raport porównawczy: uczeń rosnący vs kontrolny (wynik, liczba cykli, czas, parametry).

## 4. Inwentarz artefaktów
| Artefakt | Rodzaj | Stan |
|---|---|---|
| `brain/cerbeto/teachers/` (klient OpenAI-compatible, role, prompty, egzamin nauczyciela, limity) | system | do utworzenia |
| `brain/cerbeto/school/day.py`, `sleep.py`, `scheduler.py`, `growth_controller.py` | system | do utworzenia |
| `brain/cerbeto/memory/hippocampus.py` (SQLite) | system | do utworzenia |
| `brain/cerbeto/versions/` (rejestr wersji, awans, cofnięcie, porównanie) | system | do utworzenia |
| `brain/cerbeto/api/` (FastAPI — API mózgu dla panelu) | system | do utworzenia |
| `config/teachers.yaml`, `config/rhythm.yaml`, `config/prompts/*.md` | dane | do utworzenia |
| Ollama 0.40 + **domyślny nauczyciel `nemotron-3.5-bolt:e4b`** (9B, rodzina Gemma 4, nvfp4, runner MLX, 7,2 GB) | system zewn. | istnieje — dopuszczenie po egzaminie nauczyciela |
| Nauczyciel rezerwowy `qwen3.8:27b-mlx` (18 GB) | system zewn. | istnieje |
| `data/brain.sqlite` (hipokamp, lekcje, metryki, wersje) | dane | do utworzenia |

## 5. Zmiany w domenie funkcjonalnej
**Stany ucznia:** `idle → day → sleep → exam → (promoted | rolled_back) → [growth] → day…`; `paused` z każdego stanu.

**Reguły:**
- Przeżycie trafia do snu tylko, gdy `verified = true`: odpowiedź końcowa z wyjaśnienia nauczyciela zgadza się z prawdą z generatora; wypowiedź człowieka weryfikuje egzaminator pomocniczy + kod (jeśli dotyczy zadania z generatora).
- Do treningu trafia ślad poprawny (wyjaśnienie → odpowiedź); błędna odpowiedź ucznia tylko jako kontekst w formacie samopoprawy.
- Zaskoczenie (wysoka strata ucznia na poprawnym śladzie) zwiększa wagę przykładu w śnie (odpowiednik sygnału dopaminowego, ograniczony do ×3).
- Program sam podnosi trudność w klasie, gdy skuteczność na lekcjach > 80% na danym poziomie, i obniża przy < 40%.
- Limity nauczycieli egzekwowane twardo; po przekroczeniu nauczyciel jest wstrzymany do następnej doby.
- Nauczyciel i trening ucznia dzielą to samo GPU, więc **domyślnie rytm jest sekwencyjny** (dzień, potem sen) na każdym sprzęcie; na 32 GB równoległość dozwolona tylko jako opcja.
- **Domyślny nauczyciel:** `nemotron-3.5-bolt:e4b`. Sonda na M1 Pro (2026-10-31): 5/5 poprawnych (zadanie tekstowe, kolejność zdarzeń, relacje, dodawanie do 100, wyjaśnienie błędu), ~36 tok/s, ~36 tokenów na wyjaśnienie → ~1–2 s na lekcję, kilkadziesiąt tysięcy lekcji na dobę. Ten sam model mieści się w profilu 16 GB.
- Rezerwowy `qwen3.8:27b-mlx`: szacunkowo ~10–12 tok/s na M1 Pro (ograniczenie przepustowości pamięci) — używany, jeśli nemotron nie zda egzaminu nauczyciela lub jako drugi głos przy konsensusie.

**Odporność:** brak nauczyciela (Ollama wyłączona) → dzień wstrzymany, sen i egzaminy działają; przerwany sen → wersja nie powstaje, uczeń zostaje na poprzedniej.

**Dane:** przeżycia ~1–3 KB każde; 10k lekcji/dzień ≈ 30 MB/dzień.

## 6. Ścieżka krytyczna
1. Klient nauczyciela + egzamin nauczyciela → 2. pojedyncza lekcja end-to-end (klasa 1, liczenie) → 3. hipokamp → 4. sen + egzamin + bramki + wersje → 5. pełny cykl automatyczny dla klasy 1 → 6. kontroler wzrostu → 7. API mózgu → 8. klasy 1–3 dla obu uczniów.

**Warunek ukończenia:** 3 automatyczne cykle dzień→sen→egzamin bez interwencji, z co najmniej jednym awansem wersji i poprawą `ood` klasy 1.

## 7. Bramki kontrolne
| # | Typ | Warunek | Weryfikacja | Blokuje |
|---|---|---|---|---|
| G1 | Unit | Weryfikator odrzuca 100% wyjaśnień ze sprzeczną odpowiedzią (zestaw 100 przypadków) | `pytest tests/school/test_verify.py` | sen |
| G2 | Unit | Bramka zapominania odrzuca wersję przy spadku > 2 pp | `pytest tests/school/test_gates.py` | awans |
| G3 | Integration | Lekcja end-to-end z atrapą nauczyciela zapisuje przeżycie | `pytest tests/school/test_day.py` | G5 |
| G4 | Contract | API mózgu zgodne z `specs/contracts/brain-api.yaml` (OpenAPI) | test kontraktowy | 0006 |
| G5 | E2E | 3 cykle automatyczne z prawdziwym nauczycielem; ≥ 1 awans; `ood` klasy 1 rośnie | `cerbeto school run --cycles 3` | 0007 |
| G6 | E2E | Cofnięcie: po wymuszonej regresji aktywna jest poprzednia wersja, logity identyczne | test | 0006 |
| N1 | Negatywna | Nauczyciel nigdy nie otrzymuje zadania z puli egzaminacyjnej (log promptów vs hashe) | `cerbeto exam check-leak --prompts` | zawsze |
| N2 | Negatywna | Nauczyciel po przekroczeniu limitu nie dostaje żadnego wywołania | test | zawsze |

## 8. Oczekiwany rezultat
- **Bezpośredni:** uczeń uczy się sam, cyklicznie, z dużym lokalnym LLM.
- **Mierzalny:** krzywa wyników `id/ood` per klasa w czasie, zapominanie, liczba wzrostów, porównanie z kontrolnym.
- **Skutki uboczne:** długotrwałe obciążenie GPU (nauczyciel 27B + trening ucznia).
- **Wycofanie:** dowolna wcześniejsza wersja z rejestru; pauza szkoły.

## 9. Informacje dodatkowe
- **Odrzucone alternatywy:** zewnętrzne API jako domyślny nauczyciel (koszty, brak offline — pozostaje opcją); LLM jako sędzia poprawności zadań weryfikowalnych.
- **Dług techniczny:** jedna kolejka lekcji (bez wielu nauczycieli równolegle) w pierwszej wersji.
- **Źródła:** Complementary Learning Systems (McClelland i in., 1995); STaR (Zelikman i in., 2022); EWC; continual backprop (Dohare i in., 2024).
- **Pytania otwarte:** brak co do wyboru nauczyciela — `nemotron-3.5-bolt:e4b` domyślny na 32 GB i 16 GB, o ile zda egzamin nauczyciela (sonda 5/5 to nie egzamin).
