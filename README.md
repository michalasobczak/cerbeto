# Cerbeto

*Cerbeto* (esperanto: *cerbo* „mózg” + *-et-* zdrobnienie) — „móżdżek”.

Autorski, mały model neuronowy (8→30M parametrów, od losowych wag), który uczy się angielskiego i rozumowania „jak dziecko w szkole”: żłobek → przedszkole → klasa 1 → klasa 2, prowadzony przez duży lokalny LLM-nauczyciel, w rytmie dzień (lekcje) / sen (konsolidacja) / egzamin, rosnąc w trakcie nauki.

Projekt prowadzony jest metodyką **CDD** (Context-Driven Development): ten plik opisuje stan „tu i teraz”, `specs/` — zamiar każdej zmiany, `CHANGELOG.md` — co faktycznie dostarczono, testy — dowód. Zasady implementacyjne: [MANIFESTO.md](MANIFESTO.md).

## Stan obecny (2026-10-31)
- Specyfikacje 0001–0007 spisane; realizowana jest **0001** (biblioteka i tokenizer).
- Biblioteka pobrana offline do `data/library/` (poza gitem): BabyLM-2026-Strict, TinyStories V2, Simple English Wikipedia, FineWeb-Edu (1 shard) — ok. 4,8 GB.
- Kod mózgu i panelu: jeszcze nie istnieje.
- Sprzęt referencyjny: MacBook Pro M1 Pro 32 GB — PyTorch używa GPU przez MPS (fp32/fp16, bez bf16).
- Domyślny nauczyciel: `nemotron-3.5-bolt:e4b` w lokalnej Ollamie (sonda: 5/5, ~36 tok/s).

## Domena
| Pojęcie | Znaczenie |
|---|---|
| Uczeń | własna sieć (transformer z obiegiem zwrotnym), rośnie od ~8M do 30M parametrów |
| Uczeń kontrolny | ta sama nauka, stały rozmiar 30M — dowód, czy wzrost pomaga |
| Nauczyciel | duży LLM (lokalnie przez Ollama) w roli wychowawcy, korepetytora lub egzaminatora pomocniczego |
| Biblioteka / półki | korpusy tekstów dobrane do poziomu (etapy 0–3) |
| Lekcja | zadanie z generatora (prawda z kodu) + dialog z nauczycielem |
| Hipokamp | szybka pamięć przeżyć dnia (SQLite) |
| Sen | konsolidacja przeżyć do wag + powtórki + utrwalanie (EWC) + odnowa neuronów |
| Egzamin | niezależna ocena kodem: `id`, `ood_hard`, `ood_combo`; bramka zapominania ≤ 2 pp |
| Wersja | niezmienny stan ucznia po każdej zmianie wag; awans albo cofnięcie |

## Architektura funkcjonalna
```
                 ┌──────────────── panel/ (Rails 8, :3100) ────────────────┐
                 │ nauczyciele · szkoła na żywo · postępy · mózg · wersje · │
                 │ biblioteka · rozmowa (człowiek poprawia ucznia)          │
                 └───────────────────────┬──────────────────────────────────┘
                                         │ HTTP (specs/contracts/brain-api.yaml)
┌──────────────────────────── brain/ (Python + PyTorch, :8700) ──────────────────────────┐
│ library/tokenizer ─▶ model (uczeń, wzrost, EWC) ◀─ school: dzień ─▶ sen ─▶ egzamin      │
│                         ▲                         │   ▲                 │              │
│                     versions ◀────────────────────┘   │ hipokamp        curriculum +   │
│                                                       │ (SQLite)        exams (kod)    │
└───────────────────────────────────────────────────────┼────────────────────────────────┘
                                                        │ OpenAI-compatible
                                                Ollama (nauczyciel, lokalnie)
```

## Ścieżka krytyczna MVP
0001 biblioteka i tokenizer → 0002 uczeń → 0003 żłobek + pomiar bazowy → 0004 program i egzaminy → 0005 szkoła (priorytet MVP: wzrost `ood` po cyklach) → 0006 panel → 0007 dystrybucja offline.

## Struktura repozytorium
```
README.md  MANIFESTO.md  CHANGELOG.md
specs/            specyfikacje zmian (NNNN-slug.md) + źródło i kontrakty
scripts/          skrypty (pobieranie biblioteki, pakowanie)
brain/            mózg — Python (do utworzenia, 0001+)
panel/            panel — Rails (do utworzenia, 0006)
config/           program nauczania, architektura, rytm, nauczyciele (do utworzenia)
data/ runs/ checkpoints/   poza gitem
```

## Uruchomienie (stan obecny)
Pobranie biblioteki (wznawialne, ~5 GB):
```bash
./scripts/download_library.sh
```
