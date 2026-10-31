# 0003 — Żłobek: nauka języka (etap 0) i pomiar bazowy

Status: zaplanowana · Zależy od: 0001, 0002 · Odblokowuje: 0005

## 1. TL;DR
Uczymy ucznia mówić po angielsku na półkach etapu 0 (mowa do dziecka → proste historyjki), równolegle z uczniem kontrolnym o stałym rozmiarze 30M. Dowód: egzamin żłobka (BLiMP + spójność historyjek) i zapisany pomiar bazowy dla wszystkich klas 1–3.

## 2. Abstrakt
Uczeń startuje z losowymi wagami, więc zanim zacznie lekcje z nauczycielem musi opanować język. To najcięższy obliczeniowo etap (≥ 250M tokenów) i odbywa się raz, centralnie, na komputerze 32 GB; do ludzi trafia uczeń, który już mówi (0007). Kolejność danych naśladuje rozwój: najpierw mowa kierowana do dziecka, potem historyjki. Żeby udowodnić tezę projektu, równolegle trenujemy ucznia kontrolnego bez wzrostu (30M od początku).

## 3. Cel
**Cele**
- Pętla treningowa: AdamW, cosine LR z rozgrzewką, mieszana precyzja tam, gdzie wspiera ją urządzenie, gradient accumulation, wznawianie po przerwaniu.
- Program żłobka: faza A `s0_speech` (~30% tokenów), faza B mieszanka `s0_speech` + `s0_stories` (~70%).
- Uczeń rosnący (start ~8M, wzrost według 0002 przy plateau) i kontrolny (30M stały) na tych samych danych w tej samej kolejności.
- Egzamin żłobka i pomiar bazowy: wyniki obu uczniów na egzaminach klas 1–3 (0004) **przed** szkołą.

**Poza zakresem:** lekcje z nauczycielem, hipokamp, sen.

**Kryteria sukcesu**
- BLiMP (wersja filtrowana BabyLM) ≥ 65% [DO WERYFIKACJI po pierwszym pomiarze].
- Spójność historyjek: ≥ 70% kontynuacji ocenionych przez nauczyciela-egzaminatora jako gramatyczne i spójne (skala 1–5, próg ≥ 3).
- Całość etapu 0 dla obu uczniów ≤ 24 h na MacBooku Pro M1 Pro 32 GB.

**Budżet czasu (szacunek dla M1 Pro, ~300M tokenów, realne wykorzystanie GPU ~1–1,5 TFLOPS):**
| Uczeń | Szacowany czas |
|---|---|
| Rosnący (~8M + pętla K) | 5–7 h |
| Kontrolny (30M stały) | 12–15 h |
| Razem | ~20 h — mieści się w limicie, mały zapas |

Uczniowie trenowani **sekwencyjnie** (najpierw rosnący, potem kontrolny), np. w dwie noce; jednoczesny trening dzieliłby to samo GPU bez zysku. Precyzja wg 0002 (fp32 lub fp16 po bramce G6, bez bf16).

## 4. Inwentarz artefaktów
| Artefakt | Rodzaj | Stan |
|---|---|---|
| `brain/cerbeto/train/loop.py` | system | do utworzenia |
| `brain/cerbeto/train/nursery.py` (program etapu 0) | system | do utworzenia |
| `brain/cerbeto/exams/blimp.py` | system | do utworzenia |
| `data/library/blimp/` (BabyLM-community/BabyLM-BLIMP-Filtered) | dane | do utworzenia |
| `config/nursery.yaml` | dane | do utworzenia |
| `checkpoints/{growing,control}/v0001…` | dane | do utworzenia |
| `runs/nursery/metrics.jsonl` | dane | do utworzenia |

## 5. Zmiany w domenie funkcjonalnej
**Reguły:**
- Uczeń kontrolny różni się wyłącznie brakiem wzrostu/przycinania; seed danych identyczny.
- Wzrost w żłobku dozwolony, gdy strata walidacyjna spada < 1% przez 3 kolejne okna ewaluacji.
- Checkpoint co 30 min oraz na końcu każdej fazy; każdy checkpoint to wersja (manifest §4).
- Egzamin żłobka na zbiorze walidacyjnym, którego uczeń nie widział.

**Odporność:** przerwanie (zamknięcie laptopa) → wznowienie od ostatniego checkpointu z dokładnym miejscem w strumieniu danych.

**Dane:** ~2 × 300 MB checkpointów, metryki w JSONL.

## 6. Ścieżka krytyczna
1. Pętla treningowa + wznawianie → 2. faza A na uczniu rosnącym (krótki przebieg testowy) → 3. egzamin BLiMP → 4. pełny przebieg obu uczniów w tle → 5. pomiar bazowy klas 1–3.

**Warunek ukończenia:** obaj uczniowie mają wersję „po żłobku” z wynikami egzaminu i pomiarem bazowym zapisanymi w `runs/nursery/`.

## 7. Bramki kontrolne
| # | Typ | Warunek | Weryfikacja | Blokuje |
|---|---|---|---|---|
| G1 | Unit | Wznowienie daje ten sam następny batch i tę samą stratę (±1e-5) | `pytest tests/test_resume.py` | pełny przebieg |
| G2 | Integration | BLiMP ≥ 65% (uczeń rosnący) | `cerbeto exam nursery` | 0005 |
| G3 | Integration | Spójność historyjek ≥ 70% | `cerbeto exam nursery --stories` | 0005 |
| G4 | Integration | Pomiar bazowy klas 1–3 zapisany dla obu uczniów | plik `runs/nursery/baseline.json` | 0005 |
| G5 | Manual | 20 losowych wypowiedzi ucznia czytelnych dla człowieka | przegląd | 0007 |
| N1 | Negatywna | Żaden przykład egzaminacyjny (BLiMP, klasy 1–3) nie występuje w danych treningowych | kontrola hashy | G2–G4 |

## 8. Oczekiwany rezultat
- **Bezpośredni:** uczeń mówiący prostym angielskim.
- **Mierzalny:** BLiMP, perpleksja, spójność, pomiar bazowy klas 1–3.
- **Skutki uboczne:** kilkanaście godzin pełnego obciążenia GPU.
- **Wycofanie:** dowolny checkpoint żłobka.

## 9. Informacje dodatkowe
- **Odrzucone alternatywy:** jeden uczeń bez grupy kontrolnej (nie da się dowieść korzyści ze wzrostu).
- **Dług techniczny:** ocena spójności przez LLM jest miękka — tylko jako metryka pomocnicza; twarde bramki to BLiMP i egzaminy kodowe.
- **Źródła:** BabyLM Challenge; BLiMP (Warstadt i in., 2020).
- **Pytania otwarte:** próg BLiMP po pierwszym pomiarze; jeśli realny czas przekroczy 24 h — skrócenie żłobka ucznia kontrolnego do tej samej liczby tokenów co rosnący albo powrót do rozważenia MLX.
