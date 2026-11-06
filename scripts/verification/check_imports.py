
print("Start check")
try:
    import jax
    print("jax imported")
    import numpy
    print("numpy imported")
    import matplotlib
    print("matplotlib imported")
    import equinox
    print("equinox imported")
    import optimistix
    print("optimistix imported")
    from compore.core.networks import SIREN
    print("SIREN imported")
    from compore.core.pinns import SIREN_CSD
    print("SIREN_CSD imported")
except Exception as e:
    print(f"Error: {e}")
print("Done check")
