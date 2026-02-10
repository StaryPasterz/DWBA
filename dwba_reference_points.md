# Dane referencyjne do testów DWBA (na bazie Fig. 1–5)

Poniżej masz **sensowny zestaw danych referencyjnych (“golden points”)** do smoke-testów / regresji DWBA na bazie tych wykresów. Uprzedzam wprost: to są **wartości zgrubnie zdigitowane z rastra** (screen), więc traktuj je jako **testy tolerancyjne** (rzędu kilku–kilkunastu %). Jeśli chcesz testy „na serio” (≤2–3%), to i tak musisz te same krzywe **zdigitować narzędziem typu WebPlotDigitizer** i zapisać CSV.

## Definicje (co testujesz)

- **TCS nieskalibrowane**: \(\sigma_{\text{DWBA}}(E)\) (krzywa kropkowana).
- **TCS skalibrowane** (“Tong”, krzywa ciągła): \(\sigma_{\text{cal}}(E)\).
- **Czynnik normalizacji**:
  \[
  N(E)\equiv \frac{\sigma_{\text{cal}}(E)}{\sigma_{\text{DWBA}}(E)}.
  \]
- **DCS** (w pracy: „calibrated DWBA DCS”): typowo zakładają
  \[
  \left(\frac{d\sigma}{d\Omega}\right)_{\text{cal}}(\theta;E)=N(E)\left(\frac{d\sigma}{d\Omega}\right)_{\text{DWBA}}(\theta;E),
  \]
  czyli jak chcesz mieć „nieskalibrowane DCS”, to dzielisz przez \(N(E)\).

**Jednostki:** na wykresach są „a.u.” (w praktyce zwykle \(a_0^2\)); jeśli potrzebujesz SI:
\[
1\,a_0^2=(5.29177210903\times 10^{-11}\,\text{m})^2 = 2.8002852054\times 10^{-21}\,\text{m}^2.
\]

---

## 1) Referencje TCS (H, Fig. 1)

### H: \(1s\to 2s\) (Fig. 1a) — punkty do testów

Wartości (a.u.):

```text
# H 1s->2s  (Fig.1a)    E[eV],  sigma_cal(Tong), sigma_DWBA,  N = cal/DWBA
50,   0.2529, 0.3706, 0.6825
100,  0.1382, 0.2029, 0.6812
200,  0.07647,0.1000, 0.7647
500,  0.03529,0.03824,0.9231
```

**Jak testować w kodzie:**
- policz \(\sigma_{\text{DWBA}}(E)\),
- policz \(\sigma_{\text{cal}}(E)\) (Twoja kalibracja),
- sprawdź \(N(E)\) i/lub bezpośrednio \(\sigma\).

**Tolerancja (realistyczna dla rastra):** startowo ±10% na \(\sigma\), ±10% na \(N\).

### H: \(1s\to 2p\) (Fig. 1b) — punkty do testów

```text
# H 1s->2p  (Fig.1b)    E[eV],  sigma_cal(Tong), sigma_DWBA,  N = cal/DWBA
100,  1.6618, 2.2059, 0.7533
200,  1.0735, 1.5294, 0.7019
500,  0.5735, 0.7206, 0.7959
```

**Uwaga praktyczna:** okolice progu (dziesiątki eV) mają ostre zmiany, więc do testu regresji numerycznej lepiej używać **≥100 eV**, jak wyżej.

---

## 2) Referencje TCS (He\(^+\), Fig. 2) — jak to sensownie zrobić

Na Fig. 2 masz analogicznie \(1s\to 2s\) i \(1s\to 2p\) dla He\(^+\) + te same definicje \(N(E)\).

Z racji tego, że to raster i krzywe są cieńsze (małe wartości), **nie podaję tu “pseudo-dokładnych” liczb**, bo łatwo o błąd rzędu ×2 przy ręcznym odczycie. Zamiast tego proponuję **twardy, powtarzalny protokół**, który da Ci dobre CSV do testów:

### Protokół digitizacji (minimizes bullshit)

1. WebPlotDigitizer → typ osi: **2D (X lin, Y lin)** dla TCS, i osobno **prawa oś** dla \(N(E)\) jeśli chcesz ją digitizować bez ratio.
2. Kalibracja osi: kliknij (0,0), (1000,0) na X oraz (0, ymax), (0,0) na Y.
3. Digitizuj dwie krzywe: DWBA (kropkowana) i Tong (ciągła).
4. W CSV zapisuj kolumny: `E_eV, sigma_dwba, sigma_cal, N_ratio`.
5. Zrób punkty testowe dokładnie na siatce energii, którą i tak liczysz w kodzie (np. 50–1000 eV co 50 eV).

### Rekomendowane energie do “lock”
- 100, 200, 500, 1000 eV (stabilne, daleko od progu).

Jeśli mimo wszystko chcesz “od ręki” smoke-test dla He\(^+\): zrób test **kształtu**, tzn. czy \(\sigma_{\text{cal}}\) i \(\sigma_{\text{DWBA}}\) mają właściwą kolejność i monotoniczność oraz czy \(N(E)\) rośnie i saturuje do ~1 (tak wynika z krzywej \(Tong/DWBA\)).

---

## 3) Referencje DCS — co konkretnie testować (Fig. 3–5)

Masz dwa typy wykresów:
- **Fig. 3**: porównanie DCS skalibrowanego DWBA z eksperymentem dla H do \(n=2\) przy **50 eV** i **100 eV** (osobno wkłady \(1s\to2s\), \(1s\to2p\), suma \(1s\to(2s+2p)\)).
- **Fig. 4**: DCS skalibrowane dla H, \(1s\to2s\) i \(1s\to2p\), dla energii 15, 25, 50, 100 eV.
- **Fig. 5**: analogicznie dla He\(^+\) (45, 60, 80, 100 eV).

### Minimalny, sensowny zestaw punktów DCS do testów kodu

Żeby test nie był “na pałę”, wybierz kąty, gdzie:
- jest forward peak (małe \(\theta\)),
- jest reżim „średni”,
- jest ogon/backward.

**Proponowane kąty testowe:**
\[
\theta \in \{10^\circ,\, 30^\circ,\, 60^\circ,\, 90^\circ,\, 120^\circ,\, 150^\circ\}.
\]

**Proponowane przypadki (na start):**
- H \(1s\to2p\): \(E=\) 50 eV oraz 100 eV (Fig. 4b),
- H \(1s\to2s\): \(E=\) 50 eV oraz 100 eV (Fig. 4a),
- (opcjonalnie) suma \(1s\to(2s+2p)\) przy 50 i 100 eV (Fig. 3) do porównania z eksperymentem.

### Jak spiąć “skalibrowane vs nieskalibrowane” w DCS w Twoich testach

1. Z TCS bierz \(N(E)\) (najlepiej jako ratio \(\sigma_{\text{cal}}/\sigma_{\text{DWBA}}\) w tym samym \(E\)).
2. Liczysz surowe DCS z DWBA: \((d\sigma/d\Omega)_{\text{DWBA}}(\theta;E)\).
3. Skalibrowane robisz przez mnożenie przez \(N(E)\).
4. Testujesz oba:
   - surowe DCS vs (zdigitowane surowe, jeśli masz),
   - skalibrowane DCS vs (zdigitowane z Fig. 4–5).

### Tolerancje

- DCS na log-osi: testuj **błąd względny w logarytmie**, np.
  \[
  \Delta = \left|\log_{10} D_{\text{code}} - \log_{10} D_{\text{ref}}\right|
  \]
  i ustaw np. \(\Delta \le 0.05\) (≈12% względnie) dla danych z rastra, \(\Delta\le 0.02\) po porządnej digitizacji.

---

## 4) Co warto zrobić, żeby testy były “twarde”

Jeśli chcesz, żeby to była porządna walidacja, a nie „na oko”:

1. **Digitizuj** (jednorazowo) Fig. 1–5 do CSV:
   - TCS: DWBA i Tong (kalibracja),
   - DCS: dla wybranych energii i stanów,
   - (Fig. 3) dodatkowo punkty eksperymentalne dla sumy.
2. Zapisz w repo `tests/reference/`:
   - `H_1s2s_TCS.csv`, `H_1s2p_TCS.csv`, …
   - `H_1s2p_DCS_E50.csv` itd.
3. Testy automatyczne:
   - sprawdź wartości w punktach siatki,
   - sprawdź asymptotykę (np. zachowanie ogona),
   - sprawdź, że \(N(E)=\sigma_{\text{cal}}/\sigma_{\text{DWBA}}\) zgadza się z Twoim pipeline.

---
