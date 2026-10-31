# 0007 — Paczka dystrybucyjna: dalszy rozwój ucznia u ludzi (offline)

Status: zaplanowana · Zależy od: 0003, 0005, 0006 · Odblokowuje: —

## 1. TL;DR
Paczka z uczniem „po żłobku”, programem nauczania, generatorami, porcjami powtórek, narzędziem nauki i panelem, która na komputerze z 16 GB RAM pozwala uczniowi dalej się rozwijać (lekcje, sen, wzrost) całkowicie offline z lokalnym nauczycielem. Dowód: instalacja na czystym komputerze 16 GB i 3 cykle nauki bez internetu.

## 2. Abstrakt
Etap 0 jest robiony raz, centralnie. Dalszy rozwój — łącznie z rośnięciem sieci — ma się odbywać u ludzi, bez kosztów API. Ustalono, że kopie są niezależnymi uczniami („osobne dzieci”); format dziennika lekcji przygotowuje możliwość przyszłego dzielenia się doświadczeniami (nie wagami).

## 3. Cel
**Cele**
- Paczka: wagi + architektura + log wzrostu, `curriculum.yaml`, generatory, pule egzaminacyjne, półki powtórek (tokeny, ~kilkaset MB), manifest licencji, mózg (Python), panel (Rails).
- Instalator: `uv` (Python 3.12 + PyTorch), Ollama + nauczyciel `nemotron-3.5-bolt:e4b` (7,2 GB — ten sam co u autora), Ruby/Rails dla panelu.
- Profil 16 GB: rytm sekwencyjny (nauczyciel i sen na zmianę), sen w nocy.
- Eksport dziennika lekcji w formacie przenośnym (JSONL, bez wag).

**Poza zakresem:** łączenie wag między kopiami, uczenie federacyjne, aktualizacje online.

**Kryteria sukcesu:** czysty Mac 16 GB: instalacja ≤ 30 min (bez czasu pobierania), 3 cykle offline, szczytowe RAM ≤ 14 GB.

## 4. Inwentarz artefaktów
| Artefakt | Rodzaj | Stan |
|---|---|---|
| `scripts/package.sh`, `scripts/install.sh` | system | do utworzenia |
| `dist/adaptivemind-zero-<wersja>.tar.zst` | dane | do utworzenia |
| `LICENSES/` (manifest licencji półek) | dane | do utworzenia |
| Profil `config/profiles/16gb.yaml` | dane | do utworzenia |

## 5. Zmiany w domenie funkcjonalnej
**Reguły:** wersja startowa z paczki jest tylko do odczytu (zawsze można do niej wrócić); limit ucznia 30M egzekwowany; brak jakichkolwiek wywołań sieci poza `localhost` po instalacji.

**Odporność:** brak Ollamy → szkoła w trybie „tylko sen i egzaminy”; brak Ruby → CLI.

## 6. Ścieżka krytyczna
1. Profil 16 GB i pomiar RAM → 2. skrypt pakowania → 3. instalator → 4. test na czystej maszynie offline.

**Warunek ukończenia:** G3 zaliczona.

## 7. Bramki kontrolne
| # | Typ | Warunek | Weryfikacja | Blokuje |
|---|---|---|---|---|
| G1 | Integration | Paczka zawiera wszystkie artefakty z manifestu i licencje | `scripts/package.sh --verify` | G3 |
| G2 | Integration | Szczyt RAM w profilu 16 GB ≤ 14 GB (nauczyciel + uczeń + panel) | pomiar | G3 |
| G3 | E2E | Czysty komputer 16 GB, sieć odłączona po instalacji, 3 cykle z awansem wersji | test ręczny z checklistą | wydanie |
| N1 | Negatywna | Brak ruchu sieciowego poza localhost w trakcie cykli | monitor sieci (`nettop`/`lsof`) | wydanie |

## 8. Oczekiwany rezultat
- **Bezpośredni:** każdy może rozwijać własnego ucznia offline.
- **Mierzalny:** czas instalacji, RAM, wyniki egzaminów po cyklach u użytkownika.
- **Skutki uboczne:** ~6–8 GB na dysku (nauczyciel + paczka).
- **Wycofanie:** odinstalowanie; wersja startowa zawsze dostępna.

## 9. Informacje dodatkowe
- **Odrzucone alternatywy:** wspólny model zasilany przez kopie (scalanie wag różnie urośniętych sieci — nierozwiązane).
- **Dług techniczny:** instalacja wymaga dwóch środowisk (Python i Ruby).
- **Pytania otwarte:** Windows/Linux w MVP [DO USTALENIA]; budżet RAM 16 GB: nauczyciel 7,2 GB + uczeń ≤ 3 GB + panel + system — do potwierdzenia bramką G2.
