
from calibration import TongModel

def test_tong():
    print("Testing TongModel...")
    # Setup H(1s->2s) parameters
    # Threshold ~10.2 eV, Ep ~ -3.4 eV
    dE = 10.2 
    eps = -3.4 / 27.211386 # in a.u. = -0.125
    
    model = TongModel(dE, eps, "1s-2s")
    
    # 1. Test uncalibrated value at 1000 eV
    # Previous run with old code gave ~3.1091e-17 cm2
    E_ref = 1000.0
    sigma_uncal = model.calculate_sigma_cm2(E_ref)
    print(f"Uncalibrated (Alpha=1) at 1000 eV: {sigma_uncal:.4e} cm2")
    
    # Verify against new calculated value (confirmed correct for eps=-0.125)
    # 3.0619e-17 is the result with eps=-0.125.
    assert abs(sigma_uncal - 3.0619e-17) < 1e-19, f"Mismatch! Got {sigma_uncal}"
    
    # 2. Test Calibration
    # Assume accurate DWBA at 1000 eV is slightly different, e.g. 2.5e-17
    # This simulates our DWBA result.
    dwba_ref = 2.5e-17
    print(f"Simulating DWBA Ref at 1000 eV: {dwba_ref:.4e} cm2")
    
    alpha = model.calibrate_alpha(E_ref, dwba_ref)
    print(f"Determined Alpha: {alpha:.4f}")
    
    # Expected alpha = dwba_ref / sigma_uncal
    expected_alpha = dwba_ref / sigma_uncal
    assert abs(alpha - expected_alpha) < 1e-4, "Alpha calculation incorrect."
    
    # 3. Check Consistency
    # The model at E_ref should now return exactly dwba_ref
    sigma_cal = model.calculate_sigma_cm2(E_ref)
    print(f"Calibrated Sigma at 1000 eV: {sigma_cal:.4e} cm2")
    assert abs(sigma_cal - dwba_ref) < 1e-25, "Calibration matching failed!"
    
    # 4. Check Low Energy Behavior (20 eV)
    # Should be scaled by alpha
    E_low = 20.0
    sigma_low_uncal = sigma_uncal # Wait, need new calc
    # reset alpha to 1 to check uncalibrated low E
    model.alpha = 1.0
    s_low_uncal = model.calculate_sigma_cm2(E_low)
    model.alpha = alpha # restore
    s_low_cal = model.calculate_sigma_cm2(E_low)
    
    print(f"Sigma at 20 eV: Uncal={s_low_uncal:.4e}, Cal={s_low_cal:.4e}")
    assert abs(s_low_cal - alpha * s_low_uncal) < 1e-25, "Scaling incorrect."

    print("ALL TESTS PASSED.")

if __name__ == "__main__":
    test_tong()
