"""Debug script to capture full stack trace for LOCAL adaptive grid error."""
import traceback
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable detailed numpy error reporting
import numpy as np
np.seterr(all='raise')

from DW_main import run_from_config

try:
    run_from_config('H2s.yaml')
except Exception as e:
    print("\n" + "="*60)
    print("FULL TRACEBACK:")
    print("="*60)
    traceback.print_exc()
    print("="*60)
