
try:
    import compore.simulation
    import compore.simulation.monte_carlo
    import compore.simulation.scanner.bloch
    print("Imports successful")
except Exception as e:
    print(f"Import failed: {e}")
