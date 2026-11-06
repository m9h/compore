import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

print("Starting Import Test")
try:
    import jax
    print("JAX OK")
    import equinox
    print("Equinox OK")
    import optax
    print("Optax OK")
    import corner
    print("Corner OK")
    from compore.core.modeling_framework import JaxMultiCompartmentModel
    print("JaxMultiCompartmentModel OK")
    from compore.distributions.sphere_distributions import SD1Watson
    print("SD1Watson OK")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("Import Test Done")
