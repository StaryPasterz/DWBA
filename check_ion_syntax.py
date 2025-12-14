import sys
try:
    import ionization
    print("Import successful!")
    print("ionization file:", ionization.__file__)
except Exception as e:
    print("Import failed:", e)
    import traceback
    traceback.print_exc()
