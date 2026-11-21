# Raport z testu całej rozmowy głosowej · panel Cerbeto ([spec 0019](../specs/0019-voice-chat-panel.md))

Data: 2026-11-21 · krok 4 epika „Panel rozmowy głosowej z modelem Cerbeto”.

## Co i jak sprawdzono
- **Model:** `v6-1032M` (107 mln parametrów, snapshot HF `77f2610c…`), CPU, K = 4, T = 0, wczytanie 1,2–1,7 s.
- **Mowa:** te same programy i wagi co w trybie głosowym Agentaro: `whisper-cli` 1.9.5 z `ggml-large-v3-turbo-q5_0`
  (STT), Piper z głosem `en_US-lessac-medium` (TTS).
- **Wypowiedzi admina:** nagrania testowe. Pytania syntetyzuje inny głos Pipera (`en_GB-alan-medium`), dostają po
  0,3 s ciszy z obu stron i są zmniejszane do 16 kHz mono tak samo jak w funkcji `wav()` panelu.
- **Droga:** `scripts/voice_e2e.py` wysyła żądania HTTP do **prawdziwej klasy obsługi `cerbeto serve`** przez gniazdo w
  pamięci (sandbox tej sesji nie pozwala otworzyć portu): `/voice/transcribe` → `/v1/reply` → `/voice/speak`
  zdanie po zdaniu, czyli tak jak robi to panel.
- **Ograniczenia środowiska:** brak GPU (Metal), więc Whisper działał na CPU. Nie było przeglądarki, mikrofonu ani
  głośnika.

Polecenie (z `brain/`):
```bash
CERBETO_WHISPER_MODEL=… CERBETO_PIPER_BIN=… CERBETO_PIPER_VOICE=…/en_US-lessac-medium.onnx \
  uv run python ../scripts/voice_e2e.py --checkpoint <folder v6> --admin-voice …/en_GB-alan-medium.onnx --out e2e.json
```

## Wyniki

| Sprawdzenie | Wynik |
|---|---|
| Transkrypcja, 10 pytań | **9/10 dokładnie.** Dziesiąte: „two plus three” zapisane jako „2 plus 3”, sens zachowany. |
| Długa wypowiedź (17 s) | dokładnie, 4,0–4,2 s |
| Cisza (2 s) | `""` → panel pokazuje „Didn't catch that”, nic nie idzie do modelu |
| Odpowiedź jako tekst i mowa | 10/10 tur: odpowiedź w JSON, Piper zwraca WAV, czytana jest tylko `answer`, nigdy myśl |
| Kilka tur z pamięcią | 3 tury („opposite of up” → „And the opposite of left?” → „Thank you, goodbye”). Historia jest trzymana, a `/v1/reset` ją czyści. |
| Drugie pytanie w trakcie odpowiedzi | **409 po 0,1 ms**, bez kolejki. Pierwsze pytanie kończy się kodem 200. |
| Połączenia poza komputer | **0** poza `127.0.0.1` z procesu serwera (audit hook `socket.connect`). Strona nie ma zewnętrznych adresów. Sandbox, który filtruje cały ruch sieciowy, nie zgłosił żadnego połączenia Whispera ani Pipera. |
| Testy jednostkowe | `test_serve.py` i `test_voice.py`: 35 przeszło, 1 pominięty (prawdziwy port). Cały zestaw: 483 przeszły i 6 nie przeszło. Te 6 to `test_diloco.py`, które w sandboxie nie mogą otworzyć portu, tak samo jak przed zmianami. |

### Czasy (10 tur, CPU, drugi przebieg po poprawkach)

| Etap | Mediana | Zakres |
|---|---|---|
| STT (Whisper, CPU, 8 wątków) | **1,66 s** | 1,52–1,92 s |
| Odpowiedź modelu | **0,46 s** | 0,19–0,59 s |
| Pierwsze zdanie TTS (Piper) | **0,24 s** | 0,16–0,27 s |
| **Od końca wypowiedzi do pierwszego dźwięku** | **2,38 s** | 1,96–2,59 s |

G4 (≤ 5 s na CPU) jest spełnione z zapasem. Pomiar nie obejmuje ~1 s ciszy, po której panel uznaje wypowiedź za
zakończoną, ani dekodowania audio w przeglądarce (kilka ms). Z GPU Whisper powinien być szybszy.

## Poprawione błędy
1. **Whisper przerywał pracę, gdy GPU (Metal) nie dał mu pamięci** (`GGML_ASSERT(buffer) failed`, SIGABRT), zamiast
   przejść na CPU. Każde rozpoznanie mowy kończyło się błędem 503. Teraz `Voice.transcribe` powtarza próbę z `-ng`
   i do końca sesji zostaje na CPU. Można to też wymusić przez `CERBETO_WHISPER_NO_GPU=1`.
2. **Whisper był wolny na krótkich pytaniach**, bo zawsze liczył 30-sekundowe okno: ~7,7 s na pytanie, pierwszy
   dźwięk do 11,3 s, więc G4 nie było spełnione. Teraz okno jest dopasowane do nagrania (≥ 10 s), a Whisper używa
   do 8 wątków: **1,5–1,7 s**, ten sam tekst.
3. **ONNX Runtime pod Piperem (1.32) uruchamia telemetrię Microsoftu (1DS).** Programy mowy dostają teraz
   `ORT_DISABLE_TELEMETRY=1` (warunek N2).
4. **Zdanie zapisane przez Whispera dwa razy** („And the opposite of left? And the opposite of left?”) trafiało do
   modelu podwójnie. Teraz zdanie identyczne z poprzednim jest usuwane.
5. **Esc nie przerywał odczytu po kliknięciu mikrofonu**, bo przycisk zostawał w fokusie, a skróty pomijały
   przyciski. Esc jest teraz obsługiwany przed tym sprawdzeniem.
6. **Dymek „pisze…” zostawał na zawsze**, gdy serwer zniknął w trakcie odpowiedzi, a sprawdzenie zdrowia wcześniej
   przełączyło panel w Offline. Dymek jest teraz zawsze usuwany.

Na każdą zmianę w Pythonie (1–4) jest test w `tests/test_voice.py`. Poprawki w JS (5–6) sprawdziłem w kodzie i w
JavaScriptCore (składnia), bez przeglądarki.

## Czego nie dało się sprawdzić automatycznie: kroki dla admina
Sandbox nie pozwala otworzyć portu ani uruchomić przeglądarki, a ta sesja nie ma mikrofonu ani głośnika. Te kroki
trzeba wykonać ręcznie, w swoim terminalu i przeglądarce (ok. 10 minut):

1. **Start.** W `brain/` uruchom `uv run cerbeto serve --student final --version v6-1032M --device cpu` (albo
   `--checkpoint <folder z HF>`). Na wyjściu muszą być dwie linie `ready`: rozpoznawanie mowy i głos. Gdy głos ma
   status `MISSING`, pobierz `en_US-lessac-medium` według `docs/voice-panel.md`.
2. **Bez sieci.** Wyłącz Wi-Fi i odłącz kabel. Otwórz `http://127.0.0.1:8766` i zezwól na mikrofon.
3. **Pierwsza tura.** Naciśnij mikrofon (albo Spację) i powiedz po angielsku „What is the opposite of hot?”.
   Sprawdź: (a) Twój dymek ma ten tekst, (b) pod nim jest „heard in … s”, (c) dymek Cerbeto ma odpowiedź,
   (d) **z głośnika słychać odpowiedź** i czytane zdanie jest podświetlone, (e) pod odpowiedzią jest
   „first sound … s after you stopped”. Wartość powinna wynosić ≤ 5 s, a tutaj ~2,4 s + ~1 s ciszy.
4. **Kilka tur.** Zaznacz „memory” i zadaj 3 pytania pod rząd. Kliknij „New conversation” i sprawdź, czy rozmowa
   zaczyna się od nowa.
5. **Przerwanie odczytu.** Zadaj pytanie z dłuższą odpowiedzią (np. „Tell me about the sun.”). W trakcie czytania
   naciśnij **Stop**, a przy następnej odpowiedzi **Esc** zaraz po kliknięciu mikrofonu myszą. Dźwięk ma ucichnąć
   od razu, a panel wrócić do „Ready”.
6. **Model wyłączony.** W terminalu serwera naciśnij Ctrl+C. W ciągu ≤ 5 s panel pokazuje „Offline” z poleceniem
   startu, a mikrofon jest wyłączony. Uruchom serwer ponownie: panel wraca do „Ready” sam, bez przeładowania. Powtórz
   Ctrl+C **w trakcie „Thinking”**: panel powinien pokazać problem albo Offline bez wiszącego dymka „…”.
7. **Echo.** Przy włączonym „hands-free” sprawdź, czy głos z głośnika nie wraca jako Twoja kolejna wypowiedź.
   Przy głośnikach bez słuchawek odpowiedź nie powinna pojawić się w Twoim dymku.
8. **Nic nie wychodzi na zewnątrz (opcjonalnie).** W trakcie rozmowy uruchom
   `sudo lsof -nP -i | grep -E 'whisper|piper|python' | grep -v 127.0.0.1`. Wynik ma być pusty.

Wynik (OK lub błąd, a przy błędzie także czasy z kroku 3) wpisz w `specs/0019-voice-chat-panel.md` → Results jako G3/G5.

## Otwarte problemy
- **Jakość odpowiedzi v6 jest słaba.** Poprawnie odpowiada na pytania o przeciwieństwa (hot → cold, big → small), ale
  „capital of France” → „The name is isor…”, „spider legs” → „I don't know.”. To ograniczenie modelu (107M), nie
  panelu. Warto powtórzyć test z `bil10` po pobraniu jego wag (`--version`).
- **Szum bez mowy daje wymyślone zdanie.** Na 2 s szumu Whisper zapisał „So, let's go.”, które trafiłoby do modelu.
  W panelu nagrywanie zaczyna się dopiero, gdy dźwięk przez ≥ 150 ms jest wyraźnie głośniejszy od tła, więc stały
  szum zwykle nie uruchomi nagrania. Serwer sam tego jednak nie odfiltruje. Możliwa poprawka: VAD Whispera
  (`--vad`, wymaga modelu Silero) albo próg energii po stronie serwera.
- **Krótkie słowa bywają mylone.** W jednym przebiegu „the opposite of up” zostało zapisane jako „the opposite
  above”. W drugim było poprawne (głos testowy Pipera nie brzmi za każdym razem tak samo).
- **Nie zmierzono z GPU.** Wszystkie czasy STT pochodzą z CPU. Z Metal powinno być szybciej, do potwierdzenia w
  kroku 3 instrukcji ręcznej.
- **Panel nie jest w Agentaro** (zmiana z kroku 3). Kontrakt `/v1/*` jest gotowy na widok w Agentaro. G2 (testy
  `CerbetoClient`) czeka na ten widok.
- **Rozmowa tylko po angielsku**, bo model i jego tokenizer są angielskie.
