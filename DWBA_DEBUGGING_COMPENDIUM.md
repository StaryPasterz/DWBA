# DWBA Phase Instability & Calculation Errors — Kompendium Wiedzy

## 📋 Streszczenie Sesji

Sesja skupiła się na systematycznym diagnozowaniu i naprawianiu błędów obliczeń DWBA, które powodowały:
- Niepoprawne kształty i wielkości DCS/TCS dla Wodoru i Helu
- Ostrzeżenia "Phase unstable for L=X"
- Błędy asymptotyki Coulomba dla celów jonowych (He⁺)

---

## 🔴 WYKRYTE BŁĘDY KRYTYCZNE

### 1. Match Point Ignoruje Potencjał Odśrodkowy
**Lokalizacja**: `continuum.py:318` (`_find_match_point`)

**Problem**: Kryterium asymptotyczności używało tylko potencjału krótkozasięgowego:
```python
if abs(2.0 * U) < threshold * k2:  # ❌ Brak centrifugal!
```

**Fizyka**: Dla wysokich L, bariera odśrodkowa `V_cent = l(l+1)/r²` dominuje nad U(r) nawet przy dużych r. Match point wybierany był zbyt wcześnie, gdzie fala nie była jeszcze asymptotyczna.

**Poprawka**:
```python
V_cent = l * (l + 1) / (r * r)
V_eff = abs(2.0 * U) + V_cent
if V_eff < threshold * k2:  # ✅ Pełny V_eff
```

---

### 2. Asymptotyka Coulomba Używana Bez Sprawdzenia Ważności
**Lokalizacja**: `continuum.py:399` (`_extract_phase_logderiv_coulomb`)

**Problem**: `_coulomb_FG_asymptotic()` była wywoływana bez sprawdzenia warunku ważności:
- Wymaga: `ρ >> max(L, |η|)` gdzie `ρ = k×r`, `η = -z/k`
- Dla He⁺ przy L=30, E=100eV: potrzeba ρ > 90, ale r_m dawało tylko ρ ≈ 80

**Poprawka**: Dodano sprawdzenie i warning:
```python
rho_min_required = 3.0 * max(l, abs(eta))
if rho_m < rho_min_required:
    logger.warning(f"L={l}: Coulomb asymptotic may be inaccurate...")
```

---

### 3. Punkt Diagnostyczny Poza Regionem Asymptotycznym
**Lokalizacja**: `continuum.py:1924` (diagnostyka fazy)

**Problem**: Diagnostyka porównywała fazę w `idx_match` vs `idx_match - 5`:
- Dla wysokich L, punkt `idx_match - 5` wypadał PRZED granicą 2.5×r_turn
- Porównywanie faz w różnych regionach (asymptotycznym vs przejściowym) dawało ~3 rad różnicy

**Poprawka**: Zmiana z `idx_match - 5` na `idx_match + 10`:
```python
idx_alt = idx_match + 10  # Dalej w regionie asymptotycznym
```

---

### 4. Auto-Skalowanie r_max Nie Uwzględnia Kryterium Coulomba
**Lokalizacja**: `grid.py:305` (`compute_required_r_max`)

**Problem**: Funkcja używała tylko turning point: `r_max >= safety × (L+0.5)/k`
Dla He⁺ brakowało: `r_max >= 3×max(L, |η|)/k`

**Poprawka**: Dodano parametr `z_ion` i kryterium Coulomba:
```python
def compute_required_r_max(k_au, L_max_target, safety_factor=2.5, z_ion=0.0):
    r_turn = safety_factor * (L_max_target + 0.5) / k_au
    if abs(z_ion) > 1e-6:
        eta = abs(z_ion) / k_au
        r_coulomb = 3.0 * max(L_max_target, eta) / k_au
    r_max = max(r_turn, r_coulomb)
```

---

### 5. L_max dla Skalowania Siatki Nie Odpowiada Faktycznemu L (W TRAKCIE)
**Lokalizacja**: `DW_main.py:896-907`

**Problem**: `calculate_optimal_grid_params` używała `L_max_proj` (=5), ale:
- `driver.py` oblicza `L_dynamic = k×8+5` at runtime
- `chi_f_cache` używa `L_max_proj + 15`
- Dla E=300eV: L_dynamic ≈ 42, chi_f ≈ 57!

**Poprawka**: Wprowadzono `L_max_effective = max(int(k*8.0)+20, L_max_proj+15)` we wszystkich wywołaniach `calculate_optimal_grid_params` (interaktywne, wsadowe, jonizacja). Dodatkowo naprawiono błąd składni (wcięcie) w `run_pilot_calibration`.

---

## 🟡 PROBLEMY POTENCJALNE (Zidentyfikowane, Odroczone)

### Filon Linear Interpolation
**Lokalizacja**: `oscillatory_integrals.py:2013`

**Problem**: `np.interp` na siatce logarytmicznej może wprowadzać błędy.

### Phase Sampling dla Celów Jonowych
**Lokalizacja**: `oscillatory_integrals.py:918`

**Poprawka**: Dodano `eta_total` parametr do `check_phase_sampling()`.

---

## ✅ ZWERYFIKOWANE JAKO POPRAWNE

| Element | Lokalizacja | Status |
|---------|-------------|--------|
| L_max_projectile handling | driver.py:594-637 | ✅ OK |
| Simpson weights on log grid | grid.py:446-489 | ✅ OK |
| idx_limit bound state extent | dwba_matrix_elements.py | ✅ OK |
| Normalizacja (2/π) factor | dwba_coupling.py:353 | ✅ OK |

---

## 📐 KLUCZOWE WZORY FIZYCZNE

### Turning Point
```
r_turn(L) = √(L(L+1)) / k
```
Dla bezpiecznego match point: `r_m > 2.5 × r_turn`

### Coulomb Sommerfeld Parameter
```
η = -z_ion / k
```

### Warunek Ważności Asymptotyki Coulomba
```
ρ = k × r > 3 × max(L, |η|)
```

### Korekcje O(1/ρ) dla Coulomba (NIST DLMF §33.11)
```
θ = θ_base - L(L+1)/(2ρ)           # Korekcja fazy
A = 1 + λ/(4ρ²), λ = L(L+1) - 2η²  # Korekcja amplitudy
```

### L_max Dynamiczny (driver.py)
```
L_dynamic = k × 8 + 5
chi_f używa: L_max_proj + 15
```

---

## 📁 ZMODYFIKOWANE PLIKI

| Plik | Zmiany |
|------|--------|
| `continuum.py` | `_find_match_point()`, `_coulomb_FG_asymptotic()`, `_extract_phase_logderiv_coulomb()`, diagnostics |
| `grid.py` | `compute_required_r_max()` z z_ion |
| `oscillatory_integrals.py` | `check_phase_sampling()` z eta_total |
| `DW_main.py` | `calculate_optimal_grid_params()` z z_ion, L_max_effective |

---

## 📝 NIEROZWIĄZANE

1. **L_max_effective** nie jest jeszcze w pełni propagowany do wszystkich wywołań
2. **run_pilot_calibration** może używać innej ścieżki - wymaga audytu
