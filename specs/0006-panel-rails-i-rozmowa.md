# 0006 — Panel administracyjny (Rails) i rozmowa z uczniem

Status: zaplanowana · Zależy od: 0005 (API mózgu) · Odblokowuje: 0007

## 1. TL;DR
Lokalna aplikacja Rails 8 do zarządzania szkołą: nauczyciele, podgląd lekcji na żywo, postępy, „mózg” ucznia, wersje z porównaniem i cofaniem, biblioteka oraz rozmowa z uczniem, w której człowiek może go poprawiać. Dowód: testy systemowe Rails na atrapie API mózgu i test E2E z prawdziwym mózgiem.

## 2. Abstrakt
Specyfikacja źródłowa wymaga lokalnego panelu do konfiguracji wielu nauczycieli, obserwacji dialogów, monitorowania rozumowania, treningu i zasobów, porównywania wersji i cofania. Wybrano Ruby on Rails. Mózg (Python) jest jedynym źródłem prawdy o uczniu, lekcjach i wersjach; panel komunikuje się z nim wyłącznie przez HTTP API (manifest §6), dzięki czemu uczeń działa także bez panelu.

## 3. Cel
**Widoki:**
| Widok | Zawartość |
|---|---|
| Nauczyciele | lista, dodaj/edytuj (dostawca, endpoint, model, rola, instrukcje, harmonogram, limity), „egzamin nauczyciela” z wynikiem |
| Szkoła na żywo | stan rytmu (dzień/sen/egzamin/pauza), bieżące lekcje strumieniowane (Turbo Streams), przyciski pauza / ucz teraz / śpij teraz |
| Postępy | wyniki egzaminów per klasa × wariant w czasie, zapominanie, przenoszenie, uczeń rosnący vs kontrolny |
| Mózg | liczba parametrów, historia wzrostu/przycinania/odnowy, utrata/strata, zużycie RAM/GPU/CPU |
| Wersje | lista, porównanie dwóch wersji na tych samych zadaniach (różnice odpowiedzi), cofnięcie |
| Biblioteka | półki, przeczytane tokeny, licencje |
| Rozmowa | czat z wybraną wersją ucznia, regulacja „myśl dłużej” (K), podgląd `<|think|>`, przycisk „popraw” → przeżycie `human` w hipokampie |

**Poza zakresem:** dostęp spoza localhost, wielu użytkowników, uwierzytelnianie.

**Kryteria sukcesu:** każda funkcja panelu ze specyfikacji źródłowej dostępna w ≤ 2 kliknięciach z menu; odświeżanie „na żywo” ≤ 2 s opóźnienia.

## 4. Inwentarz artefaktów
| Artefakt | Rodzaj | Stan |
|---|---|---|
| `panel/` — Rails 8.1, Ruby 3.4, SQLite (tylko preferencje panelu), Hotwire, Propshaft | system | do utworzenia |
| `panel/app/services/brain_client.rb` | system | do utworzenia |
| `specs/contracts/brain-api.yaml` (OpenAPI, współdzielony z 0005) | dane | do utworzenia |
| Port panelu: 3100; port API mózgu: 8700 (localhost) | system | do utworzenia |

## 5. Zmiany w domenie funkcjonalnej
**Reguły:**
- Panel nie przechowuje danych ucznia; konfiguracja nauczycieli zapisywana przez API mózgu (`config/teachers.yaml` po stronie mózgu).
- Cofnięcie wersji i usunięcie nauczyciela wymagają potwierdzenia w UI.
- Poprawka człowieka w rozmowie tworzy przeżycie `origin=human, verified=pending`.
- Mózg niedostępny → panel pokazuje stan „mózg offline” i działa w trybie tylko do odczytu ostatnich danych z pamięci podręcznej.

**Odporność:** wywołania API z timeoutem 5 s; strumień lekcji wznawia się po zerwaniu.

## 6. Ścieżka krytyczna
1. Kontrakt API → 2. szkielet Rails + `BrainClient` → 3. Rozmowa (najszybszy dowód działania ucznia) → 4. Szkoła na żywo → 5. Nauczyciele → 6. Postępy i Mózg → 7. Wersje → 8. Biblioteka.

**Warunek ukończenia:** z panelu można dodać nauczyciela, przeegzaminować go, uruchomić dzień, obejrzeć lekcję na żywo, zobaczyć wynik egzaminu po śnie i cofnąć wersję.

## 7. Bramki kontrolne
| # | Typ | Warunek | Weryfikacja | Blokuje |
|---|---|---|---|---|
| G1 | Unit | `BrainClient` mapuje wszystkie odpowiedzi i błędy kontraktu | `rails test test/services` | G3 |
| G2 | Contract | Atrapa API generowana z `brain-api.yaml`; testy panelu przechodzą na atrapie | `rails test` | G3 |
| G3 | E2E | Scenariusz z warunku ukończenia na prawdziwym mózgu | `rails test:system` | 0007 |
| G4 | Manual | Przegląd UI przez właściciela | sesja akceptacyjna | 0007 |
| N1 | Negatywna | Panel nasłuchuje wyłącznie na 127.0.0.1 | test konfiguracji | zawsze |
| N2 | Negatywna | Cofnięcie bez potwierdzenia nie wywołuje API | test systemowy | zawsze |

## 8. Oczekiwany rezultat
- **Bezpośredni:** pełna obsługa szkoły z przeglądarki.
- **Mierzalny:** wszystkie metryki 0003–0005 widoczne na wykresach.
- **Skutki uboczne:** wymaganie Ruby na komputerach użytkowników (patrz 0007).
- **Wycofanie:** panel jest opcjonalny — mózg działa bez niego (CLI).

## 9. Informacje dodatkowe
- **Odrzucone alternatywy:** panel w FastAPI/HTMX (wybrano Rails); wspólna baza danych Rails + Python (podwójne właścicielstwo schematu).
- **Dług techniczny:** brak uwierzytelniania (tylko localhost).
- **Źródła:** —
- **Pytania otwarte:** biblioteka wykresów (Chartkick vs Chart.js przez importmap).
