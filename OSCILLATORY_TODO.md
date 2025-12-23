# Oscillatory Integrals - TODO List

## Status Legend
- [ ] Not started
- [/] In progress
- [x] Completed
- [~] Skipped (with reason)

---

## Phase 1: Simple Fixes (Low Risk) ✅ COMPLETED

### 1.1 Remove Forced Match Point Minimum
- [x] Remove `idx_limit = max(idx_limit, N_grid // 2)` line 208
- [x] Allow physical match point to be used directly
- [x] Add fallback only if idx_match is invalid (< 10% of grid)

### 1.2 Disable Tail for Ionization
- [x] Check `isinstance(bound_f, BoundOrbital)` before applying tail
- [x] Skip analytical tail when bound_f is ContinuumWave
- [x] Add comment explaining why

### 1.3 Remove Exchange Tail (Arbitrary 0.5)
- [x] Remove exchange tail code from CPU version
- [~] GPU version already uses standard method (no exchange tail was there)
- [x] Document: "Exchange tail not implemented due to theoretical complexity"

### 1.4 GPU/CPU Consistency Documentation
- [x] Add clear comment that oscillatory mode uses sum() like standard
- [x] GPU uses standard method for consistency
- [x] Update README to document this
- [x] Remove dead bound_overlap variable from GPU version

---

## Phase 2: Critical Math Fixes (Medium Risk) ✅ COMPLETED

### 2.1 Fix Double Weighting in Adaptive
- [x] Changed adaptive to use sum() not trapz()
- [x] Inputs are pre-weighted, so np.dot() is correct
- [x] Now consistent with "standard" method

### 2.2 Fix Multipole Moment (∫r^L instead of ∫1)
- [x] Changed to `moment_L = np.sum(w * (r ** L) * u_f * u_i)`
- [x] Pass moment_L to _analytical_multipole_tail

### 2.3 Fix Dipole Envelope (1/r² not 1/r)
- [x] Changed A_env = 1/r_m to 1/r_m² for L=1
- [x] Updated comments explaining kernel decay

---

## Phase 3: Physics Fixes (Higher Risk) ✅ COMPLETED

### 3.1 Add Coulomb Phase Terms ✅ COMPLETED
- [x] Added eta and sigma_l fields to ContinuumWave dataclass
- [x] Updated solve_continuum_wave to compute η = -z_ion/k and σ_l = arg(Γ(l+1+iη))
- [x] Updated _analytical_dipole_tail with Coulomb phase parameters
- [x] Updated _analytical_multipole_tail with Coulomb phase parameters
- [x] Updated radial_ME_all_L (CPU) to extract and pass Coulomb params
- [x] Updated radial_ME_all_L_gpu (GPU) to extract and pass Coulomb params

### 3.2 Fix Inner Integral Oscillations (Exchange) ✅ COMPLETED
- [x] Added 'filon_exchange' method to oscillatory_kernel_integral_2d
- [x] Applies CC to BOTH inner and outer integrals
- [x] Vectorized kernel interpolation using searchsorted+linear interp
- [x] Exchange integrals in radial_ME_all_L now use filon_exchange

### 3.3 Integrate Filon/Clenshaw-Curtis into Main Path ✅ COMPLETED
- [x] Added 'filon' method to oscillatory_kernel_integral_2d
- [x] Uses generate_phase_nodes() with Δφ = π/2
- [x] Clenshaw-Curtis (5 nodes) on each sub-interval
- [x] Direct and exchange integrals both use filon method
- [x] Per instruction: "rozbij całkę na przedziały, na których faza robi stały przyrost"

---

## Phase 4: Verification & Documentation ✅ COMPLETED

### 4.1 Unit Tests
- [x] Test CC weights: sum = 1.0 for all n (verified)
- [x] Test Filon integration: consistent results, 1ms for 1000 points
- [x] Test Coulomb phase: ratio ~1.28 for dipole, ~1.05 for L=2

### 4.2 Documentation
- [x] Update README with corrected formulas
- [x] Document limitations and assumptions
- [x] Created verify_oscillatory.py test script

---

## Original Instruction Compliance Checklist

From original instruction:
- [x] Filon-type quadrature for oscillatory integrals
- [x] Analytical tail treatment beyond match point
- [x] Phase-aware domain splitting (φ(r_{j+1}) - φ(r_j) = const)
- [x] Phase sampling diagnostic

---

## Notes

- Each fix should be tested before proceeding
- Maintain backward compatibility via `use_oscillatory_quadrature` flag
- If `use_oscillatory_quadrature=False`, original behavior preserved

---

## Performance Optimizations Applied ✅

| Optimization | Before | After | Improvement |
|--------------|--------|-------|-------------|
| CC weights vectorized | O(n²) Python | O(n) NumPy | ~10x |
| CC reference caching | Computed each call | Module-level constant | ~25% faster |
| Filon inner loop | Python for-loop | Batched interpolation | ~5x |
| filon_exchange kernel | Per-row interp | searchsorted + vectorized | ~3x |

**Verified Performance:**
- Filon (1000 points): 0.77 ms
- Filon Exchange (500 points): 2.3 ms
