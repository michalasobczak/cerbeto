# 0001 — Szkolna biblioteka i tokenizer

Status: w realizacji · Zależy od: — · Odblokowuje: 0002, 0003

## 1. TL;DR
Pobieramy offline „szkolną bibliotekę” (BabyLM, TinyStories, Simple English Wikipedia, FineWeb-Edu), dzielimy ją na półki według poziomu trudności i trenujemy własny tokenizer BPE (8192 tokenów). Wynik potwierdza raport biblioteki (półki, liczba tokenów, licencje) oraz testy tokenizera.

## 2. Abstrakt
Projekt startuje od pustego katalogu. Uczeń ma uczyć się „jak człowiek w szkole, po kolei”, a etap 0 (żłobek) wymaga 200–300 mln tokenów prostego języka angielskiego. Lokalny nauczyciel (30–60 tok/s) nie jest w stanie wygenerować takiej ilości w rozsądnym czasie, więc żłobek opiera się na gotowych korpusach o profilu dziecięcym. Rozważano własny scraping internetu — odrzucony (jakość „języka dorosłych”, ryzyko prawne przy dystrybucji modelu, koszt budowy pipeline'u); zamiast tego korzystamy z gotowego, oczyszczonego FineWeb-Edu filtrowanego do poziomu klas 1–3. Tokenizer musi być własny (manifest §1) i mały, bo przy 8–30M parametrów embeddingi przy słowniku 32k zjadałyby połowę budżetu.

## 3. Cel
**Cele**
- Półki biblioteki dostępne offline w `data/library/` (poza gitem), pobierane skryptem `scripts/download_library.sh`.
- Każda półka ma manifest: źródło, licencja, rozmiar, liczba słów/tokenów, poziom trudności.
- „Bibliotekarz”: ocena trudności tekstu (Flesch-Kincaid Grade Level) i przypisanie fragmentów do półek etapów 0–3.
- Własny tokenizer BPE (byte-level, 8192 tokenów + tokeny specjalne ról i rozumowania).
- Zbiory tokenów gotowe do treningu (`uint16`, memmap) z podziałem train/valid.

**Poza zakresem**
- Generatory zadań szkolnych (0004), nauczyciel LLM (0005), własny crawler WWW.
- Gutenberg/Wikijunior jako osobne pobrania — Gutenberg i Simple Wiki są już w BabyLM; Wikijunior [DO USTALENIA] w razie potrzeby.

**Kryteria sukcesu**
- ≥ 250 mln tokenów na półkach etapu 0 (żłobek) po tokenizacji.
- Średnia długość w tokenach na słowo angielskie ≤ 1,4 na zbiorze walidacyjnym TinyStories.
- 100% półek ma manifest z licencją.

## 4. Inwentarz artefaktów
| Artefakt | Rodzaj | Stan |
|---|---|---|
| `scripts/download_library.sh` | system | istnieje |
| `data/library/babylm/` — BabyLM-2026-Strict (CHILDES, BNC spoken, Switchboard, OpenSubtitles, Gutenberg dziecięcy, Simple Wiki), ~100M słów, MIT | dane | istnieje (pobierane) |
| `data/library/tinystories/` — TinyStoriesV2-GPT4 train/valid, CDLA-Sharing-1.0 | dane | istnieje (pobierane) |
| `data/library/simplewiki/` — Wikipedia simple 20231101 (parquet), CC BY-SA | dane | istnieje (pobierane) |
| `data/library/fineweb_edu/` — 1 shard sample-10BT (~2 GB), ODC-By | dane | istnieje (pobierane) |
| `brain/cerbeto/library/` — manifesty półek, bibliotekarz (czytelność), budowa półek | system | do utworzenia |
| `brain/cerbeto/tokenizer/` — trening i ładowanie BPE | system | do utworzenia |
| `config/library.yaml` — definicja półek i progów trudności | dane | do utworzenia |
| `data/shelves/*.bin` + `manifest.json` | dane | do utworzenia |
| Projekt Python `brain/` (uv, Python 3.12, PyTorch) | system | do utworzenia |

## 5. Zmiany w domenie funkcjonalnej
**Stan obecny:** brak danych i tokenizera.

**Stan docelowy — półki:**
| Półka | Zawartość | Etap |
|---|---|---|
| `s0_speech` | CHILDES, Switchboard, BNC spoken, OpenSubtitles | 0 — żłobek |
| `s0_stories` | TinyStories V2 | 0 — żłobek / 1 — przedszkole |
| `s1_books` | Gutenberg dziecięcy (BabyLM) | 1–2 |
| `s2_world` | Simple Wiki (BabyLM + pełny zrzut), fragmenty FK ≤ 4 | 2–3 |
| `s3_edu` | FineWeb-Edu, fragmenty FK ≤ 4 i ocena edu ≥ 3 | 3 |

**Reguły:**
- Fragment (akapit) trafia na półkę tylko, jeśli jego poziom FK mieści się w progu półki; reszta jest odrzucana i liczona w raporcie.
- Deduplikacja dokładna (hash znormalizowanego akapitu) między półkami.
- Tokenizer uczony na próbie zrównoważonej między półkami (żeby nie przeuczyć się na napisach filmowych).
- Tokeny specjalne: `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|teacher|>`, `<|student|>`, `<|human|>`, `<|think|>`, `<|/think|>`, `<|answer|>`.

**Odporność:** pobieranie jest wznawialne (`curl -C -`); budowa półek jest idempotentna (manifest z hashami wejść — ponowne uruchomienie pomija gotowe).

**Dane:** surowe ~5 GB; półki po tokenizacji ~0,6–1 GB.

## 6. Ścieżka krytyczna
1. Pobranie półek (skrypt) → 2. projekt `brain/` z uv → 3. bibliotekarz + budowa półek tekstowych → 4. trening tokenizera → 5. tokenizacja do `.bin` → 6. raport biblioteki.

**Warunek ukończenia:** `uv run cerbeto library report` wypisuje wszystkie półki z licencją i liczbą tokenów, a suma etapu 0 ≥ 250 mln.

## 7. Bramki kontrolne
| # | Typ | Warunek zaliczenia | Weryfikacja | Blokuje |
|---|---|---|---|---|
| G1 | Unit | Tokenizer: `decode(encode(x)) == x` dla 10 000 losowych akapitów | `pytest tests/test_tokenizer.py` | 0002 |
| G2 | Unit | Bibliotekarz: FK znanych tekstów wzorcowych w tolerancji ±0,5 | `pytest tests/test_librarian.py` | budowę półek |
| G3 | Integration | Każda półka ma `manifest.json` z polami `source, license, tokens, fk_max` | `pytest tests/test_shelves.py` | 0003 |
| G4 | Integration | Tokenów etapu 0 ≥ 250 mln; tokenów/słowo ≤ 1,4 | `cerbeto library report` | 0003 |
| G5 | Manual | Losowe 20 fragmentów z każdej półki czytelnych dla ~8-latka | przegląd człowieka | 0003 |
| N1 | Negatywna | Żaden plik z `data/` nie jest śledzony przez git | `git ls-files data/` zwraca pusto | commit |
| N2 | Negatywna | Tokenizer nie zawiera tokenów spoza słownika 8192 + specjalnych | test G1 | 0002 |

## 8. Oczekiwany rezultat
- **Bezpośredni:** biblioteka i tokenizer gotowe offline.
- **Mierzalny:** raport biblioteki (tokeny per półka, odrzucone fragmenty, licencje).
- **Skutki uboczne:** ~6 GB zajętości dysku w `data/`.
- **Wycofanie:** usunięcie `data/shelves/` i `brain/cerbeto/tokenizer/artifacts/`; surowe dane mogą zostać.

## 9. Informacje dodatkowe
- **Odrzucone alternatywy:** własny scraping WWW; generowanie żłobka przez lokalnego nauczyciela (miesiące); tokenizer 32k (koszt parametrów).
- **Dług techniczny:** FK to miara przybliżona dla dialogów (CHILDES) — dla półek mowy próg FK nie jest stosowany.
- **Zależności zewnętrzne:** Hugging Face (jednorazowe pobranie).
- **Źródła:** BabyLM Challenge (Findings, arXiv 2504.08165); TinyStories (Eldan & Li, 2023); FineWeb-Edu (Penedo i in., 2024).
- **Pytania otwarte:** czy dorzucić Wikijunior jako półkę 1–2.
