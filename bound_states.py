# bound_states.py
"""
Single-Active-Electron Bound State Solver
==========================================

Solves the radial Schrödinger equation for target bound states in the
single-active-electron (SAE) approximation.

Equation (atomic units)
-----------------------
    [-1/2 d²/dr² + l(l+1)/(2r²) + V_core(r)] u(r) = E·u(r)

where:
- u(r) = r·R(r) is the reduced radial wavefunction
- l = orbital angular momentum
- V_core(r) = effective central potential

Boundary Conditions
-------------------
    u(0) = 0,  u(r_max) = 0  (Dirichlet)

Algorithm
---------
1. Construct finite-difference Hamiltonian on non-uniform grid
2. Enforce Dirichlet BCs by removing endpoints
3. Solve sparse symmetric eigenproblem H·u = E·u
4. Keep E < 0 states (bound), sorted by energy
5. Normalize: ∫|u(r)|² dr = 1

Output
------
BoundOrbital objects containing:
- n, l, E_au (quantum numbers and energy)
- u(r) normalized radial wavefunction  

Logging
-------
Uses logging_config. Set DWBA_LOG_LEVEL=DEBUG for verbose output.
"""



from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from grid import RadialGrid, integrate_trapz
from potential_core import V_core_on_grid
from logging_config import get_logger

# Initialize module logger
logger = get_logger(__name__)


@dataclass(frozen=True)
class BoundOrbital:
    """
    Representation of a single-electron bound state in the SAE potential.

    Attributes
    ----------
    l : int
        Orbital angular momentum quantum number (ℓ).
    energy_au : float
        Eigenenergy E in Hartree atomic units. For a true bound state E < 0.
        This is the binding energy of the active electron in this orbital,
        exactly the quantity that appears in the one-electron Hamiltonian
        used in the article.
    u_of_r : np.ndarray
        Reduced radial wavefunction u(r) = r * R(r),
        on the full grid r (same length as grid.r).
        Dirichlet at both ends: u(0)=0, u(r_max)=0 (numerically).
        Normalized so that:
            ∫ |u(r)|^2 dr = 1
        using the trapezoidal weights defined in grid.py.
    n_index : int
        Index of this bound state within this ℓ, sorted by energy.
        n_index = 1 is the most bound (most negative E),
        n_index = 2 is the next, etc.
        This is NOT automatically the spectroscopic n, but it's useful
        for bookkeeping when selecting "ground" vs "excited" orbitals.
    """
    l: int
    energy_au: float
    u_of_r: np.ndarray
    n_index: int


def _kinetic_second_derivative_tridiag(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the tri-diagonal coefficients (A_i, B_i, C_i) approximating
    the second derivative d^2/dr^2 on a *non-uniform* radial grid.

    For an interior point r[i], with neighbors r[i-1], r[i+1],
    define:
        h_minus = r[i]   - r[i-1]
        h_plus  = r[i+1] - r[i]

    A standard 3-point nonuniform approximation is:
        u''(r_i) ≈ A_i * u_{i-1} + B_i * u_i + C_i * u_{i+1}

    where
        A_i =  2 / (h_minus + h_plus) * ( 1/h_minus )
        C_i =  2 / (h_minus + h_plus) * ( 1/h_plus  )
        B_i = -2 / (h_minus + h_plus) * ( 1/h_minus + 1/h_plus )

    Then the kinetic operator in atomic units is:
        T = -1/2 * d^2/dr^2
    so on the interior points:
        T_{i,i-1} = -1/2 * A_i
        T_{i,i}   = -1/2 * B_i
        T_{i,i+1} = -1/2 * C_i

    We'll return three arrays (subdiag, diag, superdiag) sized for the
    *interior points only*, i.e. i = 1..N-2. When we build the Hamiltonian,
    we'll map interior index (1..N-2) -> matrix index (0..Nint-1).

    Parameters
    ----------
    r : np.ndarray, shape (N,)
        Full radial grid (in bohr), strictly increasing.

    Returns
    -------
    sub : np.ndarray, shape (Nint-1,)
        Elements just below diagonal (lower off-diagonal) for T.
    diag : np.ndarray, shape (Nint,)
        Diagonal elements for T.
    sup : np.ndarray, shape (Nint-1,)
        Elements just above diagonal (upper off-diagonal) for T.

    Notes
    -----
    - We impose Dirichlet boundary conditions u[0]=0, u[N-1]=0 later by
      *excluding* those endpoints from the eigenproblem. So here we only
      consider i = 1..N-2 interior points.
    - This is consistent with how you'd solve the radial equation in a box
      [r_min, r_max] with vanishing wavefunction at the walls.

    """
    N = r.size
    if N < 3:
        raise ValueError("Grid too small: need at least 3 points to define interior.")

    # number of interior points
    Nint = N - 2

    sub = np.zeros(Nint - 1, dtype=float)
    diag = np.zeros(Nint, dtype=float)
    sup = np.zeros(Nint - 1, dtype=float)

    # loop over interior points i = 1..N-2
    # map to matrix index j = i-1  (so j = 0..Nint-1)
    for i in range(1, N - 1):
        j = i - 1
        h_minus = r[i] - r[i - 1]
        h_plus = r[i + 1] - r[i]

        if h_minus <= 0.0 or h_plus <= 0.0:
            raise ValueError("Radial grid must be strictly increasing with positive steps.")

        denom = (h_minus + h_plus)
        A_i =  2.0 / denom * (1.0 / h_minus)
        C_i =  2.0 / denom * (1.0 / h_plus)
        B_i = -2.0 / denom * (1.0 / h_minus + 1.0 / h_plus)

        # Kinetic operator T = -1/2 * d^2/dr^2
        T_im1 = -0.5 * A_i
        T_i   = -0.5 * B_i
        T_ip1 = -0.5 * C_i

        # fill tri-diagonal arrays
        diag[j] = T_i
        if j > 0:
            sub[j - 1] = T_im1  # sub-diagonal below diag[j]
        if j < Nint - 1:
            sup[j] = T_ip1      # super-diagonal above diag[j]

        # IMPORTANT:
        # We have to be careful at the edges j=0 and j=Nint-1:
        # - At j=0 (i=1), there's no i-1 interior index corresponding
        #   to grid point 0, but u[0] is Dirichlet=0, so we do NOT
        #   add anything extra: effectively it's consistent to just
        #   ignore the T_im1 contribution to outside point because
        #   u[0]=0 anyway.
        # - Similarly at j=Nint-1 (i=N-2), there's no i+1 interior index
        #   for grid point N-1 (Dirichlet=0). Ignoring T_ip1 outside
        #   is also consistent.
        #
        # In the assembly below, this is automatically handled, because
        # sub[j-1] and sup[j] only get set when valid.
        #
        # This is equivalent to having Dirichlet BCs u[0]=u[N-1]=0.

        # Now: subtlety
        # At j=0 we still *should* include T_im1*u[0] in diag[j] if
        # T_im1 couples to the boundary? No, because in the discrete
        # tri-diagonal representation we are removing that boundary
        # DOF entirely. The 'lost' coupling term ~ T_im1*u[0] is zero
        # because u[0]=0. So no correction to diag[j] is needed.

        # Same at j=Nint-1 for the N-1 boundary.

    return sub, diag, sup


def _effective_potential_l(V_core_array: np.ndarray, r: np.ndarray, l: int) -> np.ndarray:
    """
    Compute the effective one-electron potential:
        V_eff(r) = l(l+1)/(2 r^2) + V_core(r)

    This is exactly the potential that enters the radial equation
    the way it's written in the article.

    Parameters
    ----------
    V_core_array : np.ndarray, shape (N,)
        The core potential V_{A+}(r) in Hartree.
    r : np.ndarray, shape (N,)
        Radial grid, bohr.
    l : int
        Orbital angular momentum quantum number.

    Returns
    -------
    V_eff : np.ndarray, shape (N,)
        Effective potential in Hartree for this partial wave l.
    """
    if l < 0:
        raise ValueError("Orbital angular momentum l must be >= 0.")
    if V_core_array.shape != r.shape:
        raise ValueError("Shape mismatch in _effective_potential_l.")

    # centrifugal term l(l+1)/(2 r^2)
    # guard against r=0: grid.r[0] > 0 in our construction, so no div-by-zero
    lterm = 0.5 * l * (l + 1) / (r ** 2)

    return lterm + V_core_array


def _assemble_hamiltonian_sparse(
    grid: RadialGrid,
    V_eff: np.ndarray
):
    """
    Build the sparse Hamiltonian matrix H for the interior grid points.

    The continuous operator is:
        H = T + V_eff(r),
    where T = -1/2 d^2/dr^2 in a.u.

    We already built T as tri-diagonal on interior points 1..N-2.
    Now we simply add V_eff(r_i) on the diagonal for those same
    interior points.

    Dirichlet BCs u[0]=u[N-1]=0 are enforced implicitly by restricting
    the basis to interior points.

    Parameters
    ----------
    grid : RadialGrid
        Radial grid definition.
    V_eff : np.ndarray, shape (N,)
        Effective potential [Hartree] at all grid points, as from
        _effective_potential_l(...).

    Returns
    -------
    H : scipy.sparse.csr_matrix, shape (Nint, Nint)
        Sparse symmetric Hamiltonian matrix acting on u_interior.
    """
    r = grid.r
    N = r.size
    Nint = N - 2
    if Nint < 1:
        raise ValueError("Grid too small to form interior points.")

    sub, diag_T, sup = _kinetic_second_derivative_tridiag(r)

    # add potential to diagonal
    # interior indices i=1..N-2 map to j=0..Nint-1
    diag_full = diag_T.copy()
    for i in range(1, N - 1):
        j = i - 1
        diag_full[j] += V_eff[i]

    # build sparse tri-diagonal
    # diags([...], offsets=[-1,0,1])
    H = diags(
        diagonals=[sub, diag_full, sup],
        offsets=[-1, 0, 1],
        shape=(Nint, Nint),
        format="csr"
    )

    return H


def _normalize_radial_u(u_full: np.ndarray, grid: RadialGrid) -> np.ndarray:
    """
    Normalize u(r) so that ∫ |u(r)|^2 dr = 1 using trapezoidal weights
    from the given grid.

    Parameters
    ----------
    u_full : np.ndarray, shape (N,)
        Radial reduced wavefunction on the *full* grid,
        including boundary points u[0] and u[-1].
    grid : RadialGrid

    Returns
    -------
    u_normed : np.ndarray, shape (N,)
        Normalized copy of u_full.

    Raises
    ------
    ValueError
        If norm is ~0 (numerical failure).
    """
    prob_density = np.abs(u_full) ** 2
    norm = np.sqrt(integrate_trapz(prob_density, grid))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Failed to normalize bound state (norm invalid).")
    return u_full / norm


def solve_bound_states(
    grid: RadialGrid,
    V_core_array: np.ndarray,
    l: int,
    n_states_max: int = 5,
    which_solver: str = "eigsh"
) -> List[BoundOrbital]:
    """
    Solve for bound states u(r) for a given angular momentum l in the
    SAE potential V_core(r), following the article's one-electron picture.

    Steps:
    1. Build V_eff(r) = l(l+1)/(2 r^2) + V_core(r).
    2. Assemble the interior-point Hamiltonian H (symmetric sparse).
    3. Solve H u = E u for the lowest eigenvalues.
    4. Reconstruct full u(r) including boundary points u[0]=u[-1]=0.
    5. Keep only eigenpairs with E < 0 (bound states).
    6. Normalize u(r) so that ∫ |u(r)|^2 dr = 1.
    7. Sort by energy (most negative first) and package into BoundOrbital.

    Parameters
    ----------
    grid : RadialGrid
        Radial grid from grid.make_r_grid.
    V_core_array : np.ndarray, shape (N,)
        Core potential V_{A+}(r) [Hartree] evaluated on grid.r,
        from potential_core.V_core_on_grid(grid, params).
    l : int
        Orbital angular momentum ℓ of the electron.
        (E.g. l=0 for s-like, l=1 for p-like, etc.)
    n_states_max : int
        How many lowest eigenstates to request from the solver
        (upper bound). We will throw away positive-E states
        afterwards. For light ions 3-5 is usually enough.
    which_solver : str
        Currently only "eigsh" branch implemented (sparse symmetric
        solver for a few extremal eigenpairs).

    Returns
    -------
    bound_list : list[BoundOrbital]
        List of bound orbitals (E<0), sorted by energy (most negative first).
        Length can be 0 if no bound state with that l exists
        in the given potential + box.

    Notes
    -----
    - The eigenproblem is solved on interior points only (Dirichlet at edges).
    - The finite difference operator for d^2/dr^2 is for a general
      non-uniform radial grid (the exponential grid from grid.py),
      which matches our global approach to resolving short-range
      Coulomb behavior and long-range tails without needing
      separate grids.
    - After we get eigenvectors on interior points, we pad u[0]=0
      and u[-1]=0 and then normalize with respect to the full grid.

    - The n_index in BoundOrbital is assigned AFTER sorting,
      starting from 1 (1 = most bound).

    - These orbitals are exactly what the article treats as Φ_i, Φ_f
      (initial/final target states) later in the DWBA and in the
      distorted-wave potentials U_i, U_f.

    Raises
    ------
    RuntimeError
        If the eigen solver fails.
    ValueError
        If shapes / inputs are inconsistent.
    """
    r = grid.r
    N = r.size
    if V_core_array.shape != r.shape:
        raise ValueError("solve_bound_states: V_core_array and grid.r shape mismatch.")

    if l < 0:
        raise ValueError("solve_bound_states: l must be >= 0.")

    if N < 3:
        raise ValueError("solve_bound_states: radial grid too small.")

    # Build effective potential for this partial wave
    V_eff = _effective_potential_l(V_core_array, r, l)

    # Assemble sparse Hamiltonian on the interior points
    # This matrix H is non-symmetric on non-uniform grid regarding standard dot product.
    H_nonsym = _assemble_hamiltonian_sparse(grid, V_eff)
    Nint = N - 2
    
    # Symmetrization via weights (w_trapz on interior):
    # Symmetric matrix S = W * H_nonsym
    # Eigenproblem: S u = E W u
    # Where W = diag(w_i)
    # The weights from grid.w_trapz correspond exactly to (h_minus + h_plus)/2.
    # We use weights for interior points (indices 1..N-2).
    
    weights_inner = grid.w_trapz[1:-1]
    W_mat = diags(weights_inner, 0, shape=(Nint, Nint), format='csr')
    
    # Multiply H_nonsym by weights row-wise
    H_sym = W_mat @ H_nonsym
    
    # Ensure H_sym is numerically symmetric (remove roundoff errors)
    # H_sym = (H_sym + H_sym.T) / 2.0  <-- just to be sure, though mathematically it should be.
    # In practice scipy.sparse product might not yield perfect symmetry structure.
    # Theoretically for this 3-point scheme it is symmetric.
    
    k_req = min(n_states_max, Nint - 1)
    if k_req < 1:
        return []

    try:
        # Solve generalized eigenproblem A x = lambda M x
        # Use shift-invert mode (sigma) for better convergence near bound states
        evals, evecs = eigsh(H_sym, k=k_req, M=W_mat, sigma=-2.0, which='LM')
    except Exception as exc:
        raise RuntimeError(f"eigsh failed for l={l}: {exc}")

    # Sort eigenpairs by energy E ascending (i.e. most negative first)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    bound_states: List[BoundOrbital] = []

    # Reconstruct full u(r): interior eigenvector -> pad with zeros at boundaries.
    # Then normalize and keep only E<0.
    n_index_counter = 1
    for idx_state in range(evals.size):
        E = float(evals[idx_state])  # Hartree

        if E >= 0.0:
            # Not bound (box continuum), skip
            continue

        # evecs shape = (Nint, k_req); interior points correspond to r[1..N-2]
        u_interior = evecs[:, idx_state]

        # pad boundaries with 0
        u_full = np.zeros(N, dtype=float)
        u_full[1:-1] = u_interior

        # enforce consistent phase: make u positive near origin for readability
        # This doesn't change physics.
        if u_full[1] < 0.0:
            u_full *= -1.0

        # normalize ∫ |u(r)|^2 dr = 1
        u_norm = _normalize_radial_u(u_full, grid)

        # package
        bo = BoundOrbital(
            l=l,
            energy_au=E,
            u_of_r=u_norm,
            n_index=n_index_counter
        )
        bound_states.append(bo)
        n_index_counter += 1

        # stop if we reached n_states_max physically bound states
        if n_index_counter > n_states_max:
            break

    return bound_states
