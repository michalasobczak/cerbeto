# Manifest inżynierski — Cerbeto

Zasady implementacyjne obowiązujące w każdej specyfikacji (`specs/`). Specyfikacja może od nich odstąpić tylko jawnie, w sekcji „Informacje dodatkowe → Dług techniczny”.

## 1. Uczeń jest nasz od zera
- Żadnych gotowych wag ani gotowych tokenizerów jako punktu startu. Wagi ucznia są inicjalizowane losowo z zapisanym `seed`.
- Gotowe **dane** (korpusy, benchmarki) są dozwolone, jeśli mają znaną licencję zapisaną w manifeście półki.
- Gotowe **modele** wolno używać wyłącznie jako nauczycieli i egzaminatorów pomocniczych — nigdy jako część ucznia.

## 2. Biologia na poziomie systemu (poziom B)
- Pojedynczy neuron jest uproszczony (standardowe warstwy sieci); biologię odwzorowujemy w mechanizmach rozwoju i uczenia: wzrost, przycinanie, sen/konsolidacja, hipokamp + kora, utrwalanie synaps, odnowa neuronów.
- Każdy mechanizm „biologiczny” musi mieć odpowiednik opisany w literaturze ML (źródło w specyfikacji) i własną metrykę w panelu.
- Nie wymyślamy koła: najpierw szukamy znanej techniki, dopiero potem projektujemy własną.

## 3. Prawda pochodzi z kodu
- Poprawność odpowiedzi ucznia rozstrzyga program (generator zna odpowiedź), nie LLM.
- LLM-nauczyciel tłumaczy, pyta, poprawia — jego treść trafia do danych tylko po weryfikacji zgodności z prawdą z generatora.
- Zestawy egzaminacyjne są generowane z zarezerwowanych ziaren (`seed`) i nigdy nie trafiają do nauczyciela ani do danych treningowych (kontrola przez hash).

## 4. Każda zmiana wag jest wersją
- Każdy sen, wzrost, przycinanie i cofnięcie tworzy nową, niezmienną wersję ucznia (wagi + architektura + stan optymalizatora + log zdarzeń).
- Nowa wersja zostaje aktywna tylko po przejściu bramek egzaminacyjnych; w przeciwnym razie automatyczny powrót do poprzedniej.
- Wszystko jest odtwarzalne: konfiguracja, `seed`, hash danych, wersja kodu (commit) zapisane przy każdej wersji.

## 5. Lokalnie i offline
- Cały system działa bez internetu po jednorazowym pobraniu biblioteki i nauczyciela.
- Budżety: trening i inferencja ucznia muszą mieścić się w 16 GB RAM razem z systemem; twardy limit ucznia 30M parametrów.
- Dane (`data/`), przebiegi (`runs/`) i checkpointy (`checkpoints/`) nigdy nie trafiają do gita.

## 6. Podział technologii
- **Mózg** (uczeń, trening, szkoła, egzaminy): Python 3.12 + PyTorch, środowisko `uv`. Urządzenie: MPS → CUDA → CPU (wykrywane automatycznie).
- **Panel i rozmowa**: Ruby on Rails 8 (Hotwire), komunikacja z mózgiem wyłącznie przez HTTP API mózgu.
- **Nauczyciele**: dowolny endpoint zgodny z OpenAI Chat Completions; domyślnie lokalna Ollama.
- Wagi w formacie `safetensors` (przenośne; umożliwia późniejszy backend MLX do inferencji).
- Precyzja obliczeń: fp32 lub mieszana fp16 (autocast). **Bez bf16** — Apple M1 nie ma sprzętowego bf16 (pomiar na M1 Pro: bf16 2,1 TFLOPS vs fp16 4,2 TFLOPS).
- Neural Engine (Core ML) nie jest używany w MVP — trening wyłącznie na GPU (MPS) / CPU.

## 7. Kod
- Kod, identyfikatory i komentarze po angielsku; dokumentacja projektu (`README`, `specs/`, `CHANGELOG`) po polsku.
- Najprostsze rozwiązanie spełniające specyfikację; abstrakcja dopiero przy drugim użyciu.
- Konfiguracja w plikach YAML w repo (program nauczania, architektura, rytm dnia), sekrety w `.env` poza gitem.
- Testy: `pytest` (mózg), `rails test` (panel). Bramka z typem „Unit/Integration” w specyfikacji = test automatyczny w repo.

## 8. Proces (CDD)
- Każda zmiana zaczyna się od specyfikacji w `specs/` (9 sekcji) i kończy wpisem w `CHANGELOG.md` oraz aktualizacją `README.md`.
- Testy są dowodem tego, co opisuje CHANGELOG.
- Specyfikacje są numerowane `NNNN-slug.md` i nigdy nie są usuwane; zmiana kierunku = nowa specyfikacja odwołująca się do starej.
