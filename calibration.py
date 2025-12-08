"""
calibration.py

This module implements the Empirical Calibration Method described in the article
"Distorted-wave Born approximation for electron-atom collisions".

The calibration (Section 4.2) corrects the known deficiency of the standard DWBA
magnitude at low energies (near threshold) by scaling the results using an
empirical BE-scaled cross section formula (Tong et al.).

Key Concepts:
- Tong Model: An analytical formula for TCS (Total Cross Section).
- Calibration Factor C(E): The ratio of Empirical TCS to Calculated DWBA TCS.
- Alpha Matching: The prefactor 'alpha' in the Tong model is determined by
  matching the empirical TCS to the DWBA TCS at high energies (e.g. 1000 eV),
  where DWBA is known to be accurate.

classes:
    TongModel
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

# Constants
PI = np.pi
A0_SQ_CM2 = 2.8002852e-17 # 1 a.u.^2 in cm^2

@dataclass
class TongParams:
    """Parameters for the Tong empirical formula (Eq. 480)."""
    beta: float
    gamma: float
    delta: float

# Predefined parameters from Article Section 2.5
PARAMS_1S_2S = TongParams(beta=0.7638, gamma=1.1759, delta=0.6706)
PARAMS_1S_NP = TongParams(beta=1.32, gamma=-1.08, delta=-0.04)

class TongModel:
    """
    Implements the Tong Empirical Cross Section Model.
    
    Formula (Eq. 493 of Article):
        sigma(E) = alpha * (pi / dE^2) * exp( 1.5 * (dE - epsilon) / E_i ) * f(E_i/dE)
        
    where f(x) (Eq. 480) provides the shape.
    """
    
    def __init__(
        self,
        dE_target_eV: float,
        epsilon_exc_au: float,
        transition_type: str = "1s-2p"
    ):
        """
        Initialize the model for a specific transition.
        
        Args:
            dE_target_eV: Excitation energy (Threshold) in eV.
            epsilon_exc_au: Binding energy (eigenenergy) of the excited state in a.u.
                             (Note: for H, E_2s = -0.125 a.u. = -3.4 eV).
            transition_type: "1s-2s" or "1s-np" (determines fitting parameters).
        """
        self.dE_target_au = dE_target_eV / 27.211386
        self.epsilon_exc_au = epsilon_exc_au
        
        # Select parameters
        # Select parameters
        # Generalized Logic:
        # Article specifies Set 1 for 1s->ns (s-s transitions).
        # Article specifies Set 2 for 1s->np (s-p transitions).
        # We generalize this: 's-s' uses Set 1, others (like 's-p', 's-d') use Set 2.
        
        if "s-s" in transition_type:
            self.params = PARAMS_1S_2S
        else:
            # Default to 1s-np (dipole allowed-like) behavior for others
            self.params = PARAMS_1S_NP
            
        self.alpha: float = 1.0
        self.is_calibrated: bool = False

    def _f_shape(self, x: float) -> float:
        """
        Calculates the shape function f(x) from Eq. 480.
        x = E_i / dE
        """
        if x <= 1.0:
            return 0.0 # Below threshold
            
        beta = self.params.beta
        gamma = self.params.gamma
        delta = self.params.delta
        
        log_x = np.log(x)
        
        term1 = beta * log_x
        term2 = gamma * (1.0 - 1.0/x)
        term3 = delta * log_x / x
        
        return (1.0/x) * (term1 + term2 + term3)

    def calculate_sigma_cm2(self, E_inc_eV: float) -> float:
        """
        Calculate the Total Cross Section (TCS) in cm^2 at a given incident energy.
        Applies the current 'alpha' scaling factor.
        """
        E_inc_au = E_inc_eV / 27.211386
        
        # Threshold check
        if E_inc_au <= self.dE_target_au:
            return 0.0

        dE = self.dE_target_au
        epsilon = self.epsilon_exc_au
        
        x = E_inc_au / dE
        
        # Eq 493 Pre-factor terms
        # (pi / dE^2) * exp(...)
        
        pre_factor = PI / (dE**2)
        exp_arg = 1.5 * (dE - epsilon) / E_inc_au
        
        # Prevent overflow/underflow in exp if E is weird (though E > dE here)
        exponential = np.exp(exp_arg)
        
        shape_val = self._f_shape(x)
        
        sigma_au = self.alpha * pre_factor * exponential * shape_val
        
        return sigma_au * A0_SQ_CM2

    def calibrate_alpha(self, E_ref_eV: float, sigma_ref_cm2: float) -> float:
        """
        Determines the 'alpha' parameter by matching the Tong model to a 
        reference cross section (typically DWBA at high energy, e.g., 1000 eV).
        
        Args:
            E_ref_eV: Reference energy in eV (e.g. 1000).
            sigma_ref_cm2: The computed DWBA TCS at E_ref_eV in cm^2.
            
        Returns:
            The determined alpha.
        """
        # 1. Calculate uncalibrated sigma (alpha=1)
        old_alpha = self.alpha
        self.alpha = 1.0
        sigma_uncalibrated = self.calculate_sigma_cm2(E_ref_eV)
        
        if sigma_uncalibrated < 1e-40:
            # Avoid division by zero if model gives 0 (e.g. invalid E_ref)
            # Revert and warn logic could go here, but raising error is safer
            # or just returning 1.0 if matching fails.
            self.alpha = old_alpha
            return 1.0
            
        # 2. Determine alpha
        # sigma_ref = alpha * sigma_uncalibrated
        self.alpha = sigma_ref_cm2 / sigma_uncalibrated
        self.is_calibrated = True
        
        return self.alpha

    def get_calibration_factor(self, E_inc_eV: float, sigma_dwba_cm2: float) -> float:
        """
        Computes the Calibration Factor C(E) = Sigma_Tong(E) / Sigma_DWBA(E).
        
        This factor should be applied to DWBA partial cross sections or DCS.
        
        Args:
            E_inc_eV: Incident energy.
            sigma_dwba_cm2: The raw DWBA TCS calculated at this energy.
            
        Returns:
            C(E): The scaling factor. Returns 1.0 if not calibrated or 0/0 case.
        """
        if not self.is_calibrated:
            return 1.0
            
        if sigma_dwba_cm2 < 1e-50:
            return 0.0
            
        sigma_tong = self.calculate_sigma_cm2(E_inc_eV)
        return sigma_tong / sigma_dwba_cm2
