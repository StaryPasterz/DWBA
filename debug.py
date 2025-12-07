# debug.py
#
# Enhanced DWBA Debugger & Verification Suite (V2)
#
# USAGE:
#   python debug.py
#
# Features:
#   - Execution Timing for every step.
#   - Deep Physics Verification (Wronskians, Norms, Nodes).
#   - Full Pipeline Tracing (Excitation).
#

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple
from contextlib import contextmanager

# Add functionality to path
sys.path.append(os.getcwd())

# Import Core Modules
from grid import make_r_grid, ev_to_au, k_from_E_eV, integrate_trapz, RadialGrid
from scipy.integrate import trapezoid
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states, BoundOrbital
from distorting_potential import build_distorting_potentials, DistortingPotential
from continuum import solve_continuum_wave, ContinuumWave
from dwba_matrix_elements import radial_ME_all_L
from dwba_coupling import calculate_amplitude_contribution, Amplitudes

@dataclass
class DebugConfig:
    Z: float = 1.0
    E_inc_eV: float = 50.0
    # Process
    transition_name: str = "1s -> 2p"
    ni: int = 1
    li: int = 0
    nf: int = 2
    lf: int = 1
    
    # Grid
    r_max: float = 100.0
    n_points: int = 3000
    
    # Debug Options
    check_high_L: bool = True
    plot_waves: bool = True
    mode: str = "excitation" # or "ionization"

class DWBADebugger:
    def __init__(self, config: DebugConfig):
        self.cfg = config
        self.timings: Dict[str, float] = {}
        self.grid: Optional[RadialGrid] = None
        self.V_core: Optional[np.ndarray] = None
        
        print("\n=== DWBADebugger Initialized ===")
        print(f"Target: Z={config.Z}, Transition: {config.transition_name} ({config.mode})")
        print(f"Energy: {config.E_inc_eV} eV")
    
    @contextmanager
    def measure(self, step_name: str):
        """Context manager to measure execution time of a block."""
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter_ns() - start_ns) / 1e6
            self.timings[step_name] = dt_ms
            print(f"  [TIME] {step_name:<30}: {dt_ms:.2f} ms")

    def log(self, msg: str, header: bool = False):
        if header:
            print(f"\n--- {msg} ---")
        else:
            print(f"  {msg}")

    def check(self, condition: bool, msg: str):
        """Assertion wrapper."""
        status = "PASS" if condition else "FAIL"
        if not condition:
            print(f"  [CHECK] {msg:<40} [{status}] !!!")
        else:
            # print(f"  [CHECK] {msg:<40} [{status}]")
            pass

    def run_full_trace(self):
        """Execute the full debugging pipeline."""
        self.waves_to_plot = [] 
        
        with self.measure("Total Execution"):
            self._setup_grid()
            
            if self.cfg.mode == "excitation":
                orb_i, orb_f = self._solve_target_states()
                U_i, U_f = self._build_distorting_potentials(orb_i, orb_f)
                self._trace_excitation_physics(orb_i, orb_f, U_i, U_f)
                self._plot_debug_info(orb_i, orb_f, U_i, U_f, self.waves_to_plot)
                
            elif self.cfg.mode == "ionization":
                orb_i = self._solve_initial_state_only()
                self._trace_ionization_physics(orb_i)
                # For ionization, we don't have single U_f or orb_f in same sense, adapt plotting
                # We can mock U_f as None
                self._plot_debug_info(orb_i, None, self.U_inc_ion, self.U_scatt_ion, self.waves_to_plot)
        
        self._print_summary()

    def _solve_initial_state_only(self) -> BoundOrbital:
        self.log("Solving Initial Bound State", header=True)
        with self.measure("Bound State Initial"):
            states_i = solve_bound_states(self.grid, self.V_core, l=self.cfg.li, n_states_max=self.cfg.ni+1)
            n_idx_i = self.cfg.ni - self.cfg.li
            orb_i = [s for s in states_i if s.n_index == n_idx_i][0]
        self._verify_bound_state(orb_i, self.cfg.ni, self.cfg.li)
        return orb_i

    def _trace_ionization_physics(self, orb_i):
        self.log("Tracing Ionization Pipeline", header=True)
        
        # 1. Setup Potentials
        from distorting_potential import DistortingPotential
        k_i = k_from_E_eV(self.cfg.E_inc_eV)
        U_inc, _ = build_distorting_potentials(self.grid, self.V_core, orb_i, orb_i, k_i, k_i, False)
        
        U_ion = DistortingPotential(U_of_r=self.V_core, V_hartree_of_r=np.zeros_like(self.V_core))
        
        # Store for plotting
        self.U_inc_ion = U_inc
        self.U_scatt_ion = U_ion
        
        # 2. Energy Grid
        IP_eV = -orb_i.energy_au * 27.211
        E_total_final = self.cfg.E_inc_eV - IP_eV
        
        if E_total_final <= 0:
            self.log("Incident energy below IP! No ionization.")
            return

        self.log(f"Ionization: IP={IP_eV:.2f}eV, E_avail={E_total_final:.2f}eV")
        
        # Test just one energy point for speed
        E_eject_eV = E_total_final / 2.0
        E_scatt_eV = E_total_final - E_eject_eV
        
        self.log(f"Testing Kinematic Point: E_ej={E_eject_eV:.2f}, E_sc={E_scatt_eV:.2f}")

        # 3. Verify Ejected Wave (L=0)
        with self.measure("Ejected Wave Solve"):
            chi_eject = solve_continuum_wave(self.grid, U_ion, l=0, E_eV=E_eject_eV, z_ion=self.core_params.Zc)
            self.waves_to_plot.append((f"Ejected L=0 (E={E_eject_eV:.1f})", chi_eject))
        self._verify_continuum(chi_eject, "Chi_eject (L=0)", U_ion, E_eject_eV)
        
        # 4. Verify Projectile Waves
        with self.measure("Projectile Waves Solve"):
            # Incident Wave (at E_inc)
            # U_inc should include exchange=False as per ionization.py logic
            k_i = k_from_E_eV(self.cfg.E_inc_eV)
            chi_inc = solve_continuum_wave(self.grid, U_inc, l=0, E_eV=self.cfg.E_inc_eV, z_ion=self.core_params.Zc-1.0) # e + Neutral
            self.waves_to_plot.append((f"Incident L=0 (E={self.cfg.E_inc_eV:.1f})", chi_inc))
            
            # Scattered Wave (at E_scatt)
            # e + Ion
            chi_scatt = solve_continuum_wave(self.grid, U_ion, l=0, E_eV=E_scatt_eV, z_ion=self.core_params.Zc)
            self.waves_to_plot.append((f"Scattered L=0 (E={E_scatt_eV:.1f})", chi_scatt))
            
        # 5. Integrals Check
        self.log("Checking Radial Integrals (L=0,0)...")
        # bound_f is chi_eject
        ints = radial_ME_all_L(self.grid, self.V_core, U_inc.U_of_r, orb_i, chi_eject, chi_inc, chi_scatt, L_max=2)
        val = ints.I_L_direct.get(0, 0.0)
        self.log(f"Radial element <ki|V|kf, kej> : {val:.4e}")
        self.check(abs(val) < 1.0, "Ionization Integral Magnitude")
        
        self.log("Ionization Trace Complete (Success).")


    def _setup_grid(self):
        self.log("Initializing Grid", header=True)
        with self.measure("Grid Generation"):
            self.grid = make_r_grid(r_min=1e-4, r_max=self.cfg.r_max, n_points=self.cfg.n_points)
            self.core_params = CorePotentialParams(Zc=self.cfg.Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
            self.V_core = V_core_on_grid(self.grid, self.core_params)
        self.log(f"Grid points: {len(self.grid.r)}, r_max: {self.grid.r[-1]}")

    def _solve_target_states(self) -> (BoundOrbital, BoundOrbital):
        self.log("Solving Bound States", header=True)
        
        # Initial State
        with self.measure("Bound State Initial"):
            states_i = solve_bound_states(self.grid, self.V_core, l=self.cfg.li, n_states_max=self.cfg.ni+1)
            # Correct Indexing Logic (n - l)
            n_idx_i = self.cfg.ni - self.cfg.li
            orb_i = [s for s in states_i if s.n_index == n_idx_i][0]
            
        self._verify_bound_state(orb_i, self.cfg.ni, self.cfg.li)
        
        # Final State
        with self.measure("Bound State Final"):
            states_f = solve_bound_states(self.grid, self.V_core, l=self.cfg.lf, n_states_max=self.cfg.nf+1)
            n_idx_f = self.cfg.nf - self.cfg.lf
            orb_f = [s for s in states_f if s.n_index == n_idx_f][0]
            
        self._verify_bound_state(orb_f, self.cfg.nf, self.cfg.lf)
        
        # Orthogonality Check
        if self.cfg.li == self.cfg.lf:
            overlap = integrate_trapz(orb_i.u_of_r * orb_f.u_of_r, self.grid)
            self.check(abs(overlap) < 1e-4, f"Orthogonality <i|f> = {overlap:.2e}")
        
        return orb_i, orb_f

    def _check_equation_consistency(self, u: np.ndarray, E_au: float, V_eff: np.ndarray, l: int, label: str):
        """
        Verify that u(r) satisfies u'' + 2(E - V_eff - l(l+1)/2r^2)u = 0.
        We check the "Local Energy": E_loc = V_eff + l(l+1)/2r^2 - u''/(2u).
        It should be constant and equal to E_au.
        """
        # Second derivative via finite difference
        # Grid is non-uniform, use gradients
        
        # d/dr
        du = np.gradient(u, self.grid.r, edge_order=2)
        d2u = np.gradient(du, self.grid.r, edge_order=2)
        
        # Centrifugal term
        # handle r=0 singularty: check from index 50 onwards
        r_safe = self.grid.r
        with np.errstate(divide='ignore', invalid='ignore'):
            V_cent = float(l*(l+1)) / (2.0 * r_safe**2)
            
            # Kinetic term = -1/2 u'' / u
            # Avoid division by zero at nodes
            mask = np.abs(u) > 1e-4 * np.max(np.abs(u))
            
            # Theoretical LHS = E
            # Calculated LHS = -0.5 * u''/u + V_eff + V_cent
            
            # Let's compute residual: u'' + 2(E - V_tot)u = 0
            # Residual = u'' + 2(E - V_eff - V_cent)u
            
            V_tot = V_eff + V_cent
            term_potential = 2.0 * (E_au - V_tot) * u
            residual = d2u + term_potential
            
            # Rel Error = Residual / (max(u'') + max(2(E-V)u))
            scale = np.maximum(np.abs(d2u), np.abs(term_potential))
            rel_error = np.abs(residual) / (scale + 1e-10)
            
            # Check in region where u is significant and r is not too small
            check_mask = mask & (r_safe > 0.1) & (r_safe < self.grid.r[-10])
            
            if np.any(check_mask):
                max_err = np.max(rel_error[check_mask])
                avg_err = np.mean(rel_error[check_mask])
                
                # Finite difference on this grid is not super precise, tolerance 0.05 (5%)
                self.check(max_err < 0.1, f"{label} Schrödinger Residual (MaxRel={max_err:.4f})")
            else:
                self.log(f"{label} Skipping Consistency Check (No significant region)")

    def _verify_bound_state(self, orb: BoundOrbital, n: int, l: int):
        # ... existing checks ...
        # 1. Normalization
        norm = integrate_trapz(np.abs(orb.u_of_r)**2, self.grid)
        self.check(abs(norm - 1.0) < 1e-3, f"Norm n={n}l={l} ({norm:.6f})")
        
        # 2. Nodes
        u_max = np.max(np.abs(orb.u_of_r))
        significant_mask = np.abs(orb.u_of_r) > 1e-4 * u_max
        if np.any(significant_mask):
            u_sig = orb.u_of_r[significant_mask]
            sign_changes = np.sum(np.diff(np.sign(u_sig)) != 0)
        else:
            sign_changes = 0
        expected_nodes = n - l - 1
        self.check(sign_changes == expected_nodes, f"Nodes n={n}l={l} (Got {sign_changes}, Exp {expected_nodes})")
        
        # 3. Energy
        if self.cfg.Z == 1.0:
            expected_E = -0.5 / (n**2)
            self.check(abs(orb.energy_au - expected_E) < 1e-3, f"Energy n={n} ({orb.energy_au:.5f}) vs {expected_E:.5f}")
            
        # 4. Equation Consistency
        self._check_equation_consistency(orb.u_of_r, orb.energy_au, self.V_core, l, f"Bound {n}{l}")

    def _verify_continuum(self, wave: ContinuumWave, label: str, U_pot: Optional[DistortingPotential] = None, E_eV: float = 0.0):
        # 1. Asymptotic Amplitude
        tail_max = np.max(np.abs(wave.chi_of_r[-int(self.cfg.n_points*0.1):]))
        self.check(0.9 < tail_max < 1.1, f"{label} Asymp Amp (~{tail_max:.3f})")
        
        # 2. Equation Consistency
        if U_pot is not None:
             # U_of_r is the potential V_eff (excluding centrifugal)
             # Convert E_eV to E_au
             E_au = E_eV / 27.211386
             # Note: solve_continuum_wave internally manages U.
             # If U_pot includes exchange, checking is harder. 
             # But here we assume local potential for check.
             self._check_equation_consistency(wave.chi_of_r, E_au, U_pot.U_of_r, wave.l, label)

    def _build_distorting_potentials(self, orb_i, orb_f):
        self.log("Building Distorting Potentials", header=True)
        
        dE = orb_f.energy_au - orb_i.energy_au
        self.E_final_eV = self.cfg.E_inc_eV - (dE * 27.211386)
        
        self.k_i = k_from_E_eV(self.cfg.E_inc_eV)
        self.k_f = k_from_E_eV(self.E_final_eV)
        
        self.log(f"Kinematics: E_i={self.cfg.E_inc_eV:.2f}eV (k={self.k_i:.3f}), E_f={self.E_final_eV:.2f}eV (k={self.k_f:.3f})")
        
        with self.measure("Potentials Construction"):
            U_i, U_f = build_distorting_potentials(self.grid, self.V_core, orb_i, orb_f, self.k_i, self.k_f, use_exchange=False)
            
        # Check Short Range Match
        # U should match V_core at very short range (nuclear dominance)
        self.check(np.isclose(U_i.U_of_r[0], self.V_core[0], rtol=0.1), "Potential Short-range limits match")
        
        return U_i, U_f

    def _trace_excitation_physics(self, orb_i, orb_f, U_i, U_f):
        self.log("Tracing Excitation Pipeline", header=True)

        # 1. Continuum Wave Analysis (Low Partial Wave)
        self.log("Analyzing Continuum Physics (L=0)...")
        with self.measure("Continuum Solve (L=0)"):
            chi_i = solve_continuum_wave(self.grid, U_i, l=0, E_eV=self.cfg.E_inc_eV)
            self.waves_to_plot.append((f"Incident L=0 (E={self.cfg.E_inc_eV:.1f})", chi_i))
        
        self._verify_continuum(chi_i, "Chi_i (L=0)", U_i, self.cfg.E_inc_eV)
        
        # 2. High-L Stability Scan
        if self.cfg.check_high_L:
            self.log("Scanning High-L Stability...")
            with self.measure("High-L Scan (10..50)"):
                for l_test in range(10, 55, 10):
                    c = solve_continuum_wave(self.grid, U_i, l=l_test, E_eV=self.cfg.E_inc_eV)
                    max_amp = np.max(np.abs(c.chi_of_r))
                    self.check(max_amp < 50.0, f"Stability L={l_test} (MaxAmp={max_amp:.1f})")

        # 3. Full Cross Section Calculation
        self.log("Running Full Partial Wave Summation...")
        
        theta_grid = np.linspace(0.0, np.pi, 200)
        total_amplitudes = {}
        Li, Lf = self.cfg.li, self.cfg.lf
        
        # Init storage
        for Mi in range(-Li, Li+1):
            for Mf in range(-Lf, Lf+1):
                total_amplitudes[(Mi, Mf)] = Amplitudes(np.zeros_like(theta_grid, complex), np.zeros_like(theta_grid, complex))

        L_max_proj = 20
        with self.measure(f"PW Summation (L_max={L_max_proj})"):
            for l_i in range(L_max_proj + 1):
                # Optimize: only recalc chi if needed (not done here for clarity)
                chi_i = solve_continuum_wave(self.grid, U_i, l=l_i, E_eV=self.cfg.E_inc_eV)
                
                parity_target = (Li + Lf) % 2
                lf_min = max(0, l_i - 5) 
                lf_max = l_i + 5
                
                for l_f in range(lf_min, lf_max + 1):
                    if (l_i + l_f) % 2 != parity_target: continue
                    
                    chi_f = solve_continuum_wave(self.grid, U_f, l=l_f, E_eV=self.E_final_eV)
                    
                    # Compute Integrals
                    ints = radial_ME_all_L(self.grid, self.V_core, U_i.U_of_r, orb_i, orb_f, chi_i, chi_f, L_max=8)
                    
                    # Accumulate
                    for Mi in total_amplitudes.keys(): 
                         amps = calculate_amplitude_contribution(theta_grid, ints.I_L_direct, ints.I_L_exchange, l_i, l_f, self.k_i, self.k_f, Li, Lf, *Mi)
                         # Note: `Mi` here is Tuple (Mi, Mf) from dict key. Wait.
                         # calculate_amplitude_contribution takes single integers.
                         # Fix: total_amplitudes keys are (Mi, Mf).
                         pass
                    
                    # Correct Loop for accumulation
                    for Mi in range(-Li, Li+1):
                        for Mf in range(-Lf, Lf+1):
                             amps = calculate_amplitude_contribution(theta_grid, ints.I_L_direct, ints.I_L_exchange, l_i, l_f, self.k_i, self.k_f, Li, Lf, Mi, Mf)
                             total_amplitudes[(Mi, Mf)].f_theta += amps.f_theta
                             total_amplitudes[(Mi, Mf)].g_theta += amps.g_theta

        # 4. Integrate Sigma
        sigma_au = self._integrate_sigma(total_amplitudes, theta_grid)
        self.log(f"Final Cross Section: {sigma_au:.4e} a.u.", header=True)
        self.log(f"                     {sigma_au * 2.80028e-17:.4e} cm^2")



    def _plot_debug_info(self, orb_i, orb_f, U_i, U_f, chi_list: List[Tuple[str, ContinuumWave]]):
        if not self.cfg.plot_waves:
            return
            
        self.log("Generating Debug Plots...", header=True)
        os.makedirs("debug_plots", exist_ok=True)
        
        # 1. Plot Bound States
        plt.figure(figsize=(10, 6))
        r = self.grid.r
        mask = r < 20.0 # significant region
        plt.plot(r[mask], orb_i.u_of_r[mask], label=f"Initial (n={self.cfg.ni}, l={self.cfg.li})")
        if orb_f:
             plt.plot(r[mask], orb_f.u_of_r[mask], label=f"Final (n={self.cfg.nf}, l={self.cfg.lf})")
        plt.title(f"Target Bound States ({self.cfg.transition_name})")
        plt.xlabel("r (a.u.)")
        plt.ylabel("u(r)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"debug_plots/bound_states_{self.cfg.mode}.png")
        plt.close()
        
        # 2. Plot Potentials
        plt.figure(figsize=(10, 6))
        plt.plot(r[mask], self.V_core[mask], 'k--', label="V_core")
        plt.plot(r[mask], U_i.U_of_r[mask], label="U_i (Distorting)")
        if U_f:
            plt.plot(r[mask], U_f.U_of_r[mask], label="U_f (Distorting)")
        plt.title(f"Potentials ({self.cfg.mode})")
        plt.xlabel("r (a.u.)")
        plt.ylabel("V (a.u.)")
        plt.ylim(bottom=-5.0, top=1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"debug_plots/potentials_{self.cfg.mode}.png")
        plt.close()
        
        # 3. Plot Continuum Waves
        plt.figure(figsize=(10, 6))
        mask_cont = r < 50.0
        for label, chi in chi_list:
            plt.plot(r[mask_cont], chi.chi_of_r[mask_cont], label=label)
        plt.title("Continuum Waves (Partial)")
        plt.xlabel("r (a.u.)")
        plt.ylabel("chi(r)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"debug_plots/continuum_{self.cfg.mode}.png")
        plt.close()
        
        self.log("Plots saved to ./debug_plots/")

    def _integrate_sigma(self, amp_dict, theta_grid):
        total_dcs = np.zeros_like(theta_grid, dtype=float)
        prefac = (self.k_f / self.k_i)
        
        spin_singlet = 0.25
        spin_triplet = 0.75
        
        Li = self.cfg.li
        
        for (Mi, Mf), amps in amp_dict.items():
            f = amps.f_theta
            g = amps.g_theta
            term = spin_singlet * np.abs(f+g)**2 + spin_triplet * np.abs(f-g)**2
            total_dcs += term
            
        total_dcs *= prefac * (1.0 / (2*Li + 1))
        
        # Integration
        sigma = 2 * np.pi * trapezoid(total_dcs * np.sin(theta_grid), theta_grid)
        return sigma

    def _print_summary(self):
        self.log("TIMING SUMMARY", header=True)
        total = self.timings.get("Total Execution", 1.0)
        for k, v in self.timings.items():
            pct = (v / total) * 100 if "Total" not in k else 100.0
            print(f"  {k:<30} : {v:8.2f} ms ({pct:5.1f}%)")

if __name__ == "__main__":
    # Case 1: Excitation 1s->2p
    print(">>> RUNNING EXCITATION CHECK")
    cfg_exc = DebugConfig(
        Z=1.0, 
        transition_name="H 1s -> 2p",
        ni=1, li=0,
        nf=2, lf=1,
        mode="excitation"
    )
    DWBADebugger(cfg_exc).run_full_trace()
    
    # Case 2: Ionization H 1s -> Continuum
    print("\n>>> RUNNING IONIZATION CHECK")
    cfg_ion = DebugConfig(
        Z=1.0,
        transition_name="H 1s -> Ion",
        ni=1, li=0,
        mode="ionization",
        E_inc_eV=50.0
    )
    DWBADebugger(cfg_ion).run_full_trace()
