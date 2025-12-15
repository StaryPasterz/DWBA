"""
Quick diagnostic script to compare H and Li potential and bound states.
"""
import numpy as np
from grid import make_r_grid
from potential_core import V_core_on_grid
from bound_states import solve_bound_states
import atom_library

def diagnose_atom(name: str):
    print(f"\n{'='*50}")
    print(f"DIAGNOSTICS: {name}")
    print('='*50)
    
    atom = atom_library.get_atom(name)
    print(f"Z={atom.Z}, Zc={atom.core_params.Zc}")
    print(f"Target state: n={atom.default_n}, l={atom.default_l}")
    print(f"IP (eV): {atom.ip_ev}")
    
    # Check a_params
    p = atom.core_params
    print(f"\nCore params: a1={p.a1:.4f}, a2={p.a2:.4f}, a3={p.a3:.4f}")
    print(f"             a4={p.a4:.4f}, a5={p.a5:.4f}, a6={p.a6:.4f}")
    
    # Compute Z_eff at r=0
    # S(0) = a1*exp(0) + a3*0*exp(0) + a5*exp(0) = a1 + a5
    z_eff_0 = p.Zc + p.a1 + p.a5
    print(f"\nZ_eff(r→0) ≈ {z_eff_0:.2f} (should be ~{atom.Z})")
    print(f"Z_eff(r→∞) = Zc = {p.Zc}")
    
    # Create grid and potential
    grid = make_r_grid(r_min=1e-4, r_max=200.0, n_points=2000)
    V = V_core_on_grid(grid, p)
    
    # Check potential values
    print(f"\nPotential check:")
    print(f"  V(r=0.01) = {np.interp(0.01, grid.r, V):.4f} a.u.")
    print(f"  V(r=1.0)  = {np.interp(1.0, grid.r, V):.4f} a.u.")
    print(f"  V(r=10.0) = {np.interp(10.0, grid.r, V):.4f} a.u.")
    
    # Actual Z_eff = -r*V
    r_test = grid.r[grid.r > 0.001]
    V_test = V[grid.r > 0.001]
    z_eff = -r_test * V_test
    print(f"  Z_eff(r=0.01) = {np.interp(0.01, r_test, z_eff):.2f}")
    print(f"  Z_eff(r=1.0)  = {np.interp(1.0, r_test, z_eff):.2f}")
    print(f"  Z_eff(r=10.0) = {np.interp(10.0, r_test, z_eff):.2f}")
    
    # Solve bound states
    l = atom.default_l
    states = solve_bound_states(grid, V, l=l, n_states_max=5)
    
    print(f"\nBound states for l={l}:")
    for s in states:
        E_eV = s.energy_au * 27.211386
        print(f"  n_index={s.n_index}: E = {E_eV:.4f} eV ({s.energy_au:.4f} a.u.)")
    
    # Expected valence state
    target_n_idx = atom.default_n - atom.default_l
    expected_E_eV = -atom.ip_ev
    
    for s in states:
        if s.n_index == target_n_idx:
            calc_E_eV = s.energy_au * 27.211386
            diff = abs(calc_E_eV - expected_E_eV)
            print(f"\n  Valence state (n_idx={target_n_idx}):")
            print(f"    Calculated: {calc_E_eV:.4f} eV")
            print(f"    Expected:   {expected_E_eV:.4f} eV")
            print(f"    Diff:       {diff:.4f} eV")
            break

if __name__ == "__main__":
    diagnose_atom("H")
    diagnose_atom("Li")
    diagnose_atom("Na")
    print("\n" + "="*50)
    print("DIAGNOSTIC COMPLETE")
