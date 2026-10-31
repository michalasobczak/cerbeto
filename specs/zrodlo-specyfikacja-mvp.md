# Źródło: specyfikacja MVP „AdaptiveMind Zero” (wejście biznesowe) — projekt Cerbeto

Dokument wejściowy, z którego wyprowadzono specyfikacje 0001–0007. Nie jest specyfikacją zmiany — nie edytujemy go; doprecyzowania trafiają do specyfikacji numerowanych.

> Celem projektu jest stworzenie od podstaw autorskiego, niewielkiego modelu neuronowego (10–30M parametrów), z własną architekturą, tokenizerem i losowo inicjalizowanymi wagami, bez wykorzystywania gotowych modeli jako podstawy, zdolnego do komunikacji w języku naturalnym, wieloetapowego wnioskowania i ciągłego uczenia się. Model będzie domyślnie prowadził automatyczne dialogi z jednym lub wieloma zewnętrznymi dużymi modelami LLM pełniącymi funkcje nauczycieli, którzy zadają pytania, wyjaśniają błędy, przekazują informacje zwrotne i stopniowo zwiększają trudność zadań. Zweryfikowane interakcje będą automatycznie przekształcane w dane treningowe, a parametry sieci aktualizowane lokalnie w cyklicznym procesie uczenia, z kontrolą zapominania wcześniejszych umiejętności i niezależną oceną postępów. System będzie posiadał trwałą pamięć, iteracyjny mechanizm rozumowania, interfejs rozmowy z uczniem oraz **lokalny panel administracyjny** umożliwiający konfigurowanie wielu nauczycieli (dostawca API, model, rola, instrukcje, harmonogram, limity kosztów), obserwowanie dialogów, monitorowanie skuteczności rozumowania, parametrów treningu i zużycia zasobów, porównywanie kolejnych wersji modelu oraz cofanie aktualizacji. Architektura modelu pozostanie eksperymentalna i niezależna od konkretnej rodziny sieci neuronowych. Projekt zostanie zoptymalizowany pod trening na MacBooku Pro z 32 GB pamięci oraz lokalne uruchamianie i dalsze uczenie na komputerach z 16 GB RAM. **Priorytetem MVP będzie uzyskanie mierzalnej poprawy rozumowania na nowych zadaniach po automatycznych cyklach nauki z nauczycielem**, przy zachowaniu pełnej kontroli nad wagami, danymi i procesem rozwoju ucznia.

## Decyzje z sesji specyfikacyjnej (2026-10-31)
| # | Decyzja | Specyfikacja |
|---|---|---|
| 1 | Język ucznia: angielski; pipeline niezależny od języka (polski jako późniejsza „nowa umiejętność” — test zapominania) | 0001, 0004 |
| 2 | Nauka „jak w szkole, po kolei”: etapy 0–3 (żłobek, przedszkole, klasa 1, klasa 2); bez kodowania, matematyka powoli i w kontekście języka | 0003, 0004 |
| 3 | Biologia na poziomie B: mechanizmy rozwoju (wzrost, przycinanie, sen, hipokamp+kora, utrwalanie, odnowa), nie wierny model neuronu | 0002, 0005 |
| 4 | Pełne wagi zamiast adapterów; uczeń rośnie od ~8M do maks. 30M | 0002 |
| 5 | Nauczyciel: duży LLM uruchomiony lokalnie (Ollama), wybierany egzaminem nauczyciela; człowiek jako dodatkowy nauczyciel w rozmowie | 0005, 0006 |
| 6 | Żłobek na gotowych korpusach dziecięcych („szkolna biblioteka”), nie własny scraping; dane offline, poza gitem | 0001 |
| 7 | PyTorch (przenośność); wagi w safetensors (droga do MLX dla inferencji) | 0002 |
| 8 | Panel: Ruby on Rails; mózg: Python — komunikacja przez HTTP API | 0006 |
| 9 | Kopie u ludzi są niezależnymi uczniami; żłobek robiony centralnie | 0007 |
| 10 | Egzaminy: zdanie ≥ 85% (id), bramka zapominania ≤ 2 pp, pomiar przenoszenia, uczeń kontrolny 30M | 0004, 0005 |
| 11 | Rytm dnia automatyczny z ręcznym sterowaniem w panelu | 0005 |
| 12 | Trening na GPU przez MPS; precyzja fp32 / fp16, **bez bf16** (M1 bez sprzętowego wsparcia) | 0002, 0003 |
| 13 | Core ML / Neural Engine odłożone (NPU tylko do inferencji, koszt rekonwersji po wzroście) | 0002 |
| 14 | Domyślny nauczyciel: `nemotron-3.5-bolt:e4b` (7,2 GB, ~36 tok/s, ten sam na 32 i 16 GB); rezerwa `qwen3.8:27b-mlx` | 0005, 0007 |
| 15 | Rytm dzień/sen domyślnie sekwencyjny (wspólne GPU); żłobek obu uczniów sekwencyjnie, ~20 h na M1 Pro | 0003, 0005 |
| 16 | Nazwa projektu: **Cerbeto** (esperanto: „móżdżek”); repozytorium publiczne `michalasobczak/cerbeto`, pakiet `cerbeto` | — |
