"""Closed-form kernels lafix needs beyond stick / zeppelin / ball: the
Gaussian-phase-distribution sphere (Murday & Cotts 1968; Balinov et al. 1993)
used by ``FixelConfig(restricted_kernel="sphere")``. Vendored from the
``compore`` platform (``compore.signal_models.sphere_models.g3_sphere``,
``compore.constants.SPHERE_ROOTS``) so that lafix depends on dipy + jax +
optax only. SI units throughout: b in s/m², lengths in m, times in s.
"""
import jax.numpy as jnp
from jax import jit

GYRO_MAGNETIC_RATIO = 267.5152549e6   # rad s⁻¹ T⁻¹

# The 100 roots of (1/x) J_{3/2}(x) = J_{5/2}(x), i.e. of the derivative of the
# order-1 spherical Bessel function, used by the sphere GPD series (copied
# verbatim from compore.constants).
SPHERE_ROOTS = jnp.array([
    2.081575978, 5.940369990, 9.205840145,
    12.40444502, 15.57923641, 18.74264558, 21.89969648,
    25.05282528, 28.20336100, 31.35209173, 34.49951492,
    37.64596032, 40.79165523, 43.93676147, 47.08139741,
    50.22565165, 53.36959180, 56.51327045, 59.65672900,
    62.80000055, 65.94311190, 69.08608495, 72.22893775,
    75.37168540, 78.51434055, 81.65691380, 84.79941440,
    87.94185005, 91.08422750, 94.22655255, 97.36883035,
    100.5110653, 103.6532613, 106.7954217, 109.9375497,
    113.0796480, 116.2217188, 119.3637645, 122.5057870,
    125.6477880, 128.7897690, 131.9317315, 135.0736768,
    138.2156061, 141.3575204, 144.4994207, 147.6413080,
    150.7831829, 153.9250463, 157.0668989, 160.2087413,
    163.3505741, 166.4923978, 169.6342129, 172.7760200,
    175.9178194, 179.0596116, 182.2013968, 185.3431756,
    188.4849481, 191.6267147, 194.7684757, 197.9102314,
    201.0519820, 204.1937277, 207.3354688, 210.4772054,
    213.6189378, 216.7606662, 219.9023907, 223.0441114,
    226.1858287, 229.3275425, 232.4692530, 235.6109603,
    238.7526647, 241.8943662, 245.0360648, 248.1777608,
    251.3194542, 254.4611451, 257.6028336, 260.7445198,
    263.8862038, 267.0278856, 270.1695654, 273.3112431,
    276.4529189, 279.5945929, 282.7362650, 285.8779354,
    289.0196041, 292.1612712, 295.3029367, 298.4446006,
    301.5862631, 304.7279241, 307.8695837, 311.0112420,
    314.1528990
])


@jit
def g3_sphere(bvals, bvecs, diameter, diffusion_constant, big_delta, small_delta):
    """
    Computes signal for restricted diffusion in a Sphere (Soma) using
    the Gaussian Phase Distribution (GPD) approximation (Murday & Cotts, 1968).
    """
    radius = diameter / 2.0
    
    # 1. Gradients
    tau = big_delta - small_delta / 3.0
    G_mag = jnp.sqrt(bvals / (tau + 1e-9)) / (GYRO_MAGNETIC_RATIO * small_delta)
    
    # 2. Roots (Dimensionless mu)
    # alpha_sq_broad = mu^2
    alpha_sq = SPHERE_ROOTS ** 2
    alpha_sq_broad = alpha_sq[None, :] 
    
    # 3. Time Constants (Dm = D * mu^2 / R^2)
    # This part was correct
    Dm_alpha2 = diffusion_constant * alpha_sq_broad / (radius**2)
    
    # 4. The Denominator (The Fix)
    # We use dimensionless mu for the check (mu^2 - 2)
    # The physical 1/alpha^2 term contributes an R^2 factor to the numerator
    # because 1/alpha_phys^2 = R^2 / mu^2
    denom_dimensionless = alpha_sq_broad * (alpha_sq_broad - 2)
    
    # 5. Time Terms
    # Ensure deltas broadcast against roots (axis 1)
    # small_delta/big_delta: (N,) or scalar -> expand to (N, 1) or (1, 1) if not already
    sd_expanded = jnp.expand_dims(small_delta, -1) if jnp.ndim(small_delta) > 0 else small_delta
    bd_expanded = jnp.expand_dims(big_delta, -1) if jnp.ndim(big_delta) > 0 else big_delta
    
    exp_1 = jnp.exp(-Dm_alpha2 * sd_expanded)
    exp_2 = jnp.exp(-Dm_alpha2 * bd_expanded)
    exp_3 = jnp.exp(-Dm_alpha2 * (bd_expanded - sd_expanded))
    exp_4 = jnp.exp(-Dm_alpha2 * (bd_expanded + sd_expanded))
    
    time_term = (
        2 * sd_expanded 
        - (2 + exp_3 - 2 * exp_2 - 2 * exp_1 + exp_4) / Dm_alpha2
    )
    
    # 6. Summation
    # We multiply by radius**2 because of the conversion from physical alpha to mu
    # We also need to divide by Dm_alpha2 ($D \alpha^2$) as per the formula
    sum_term = jnp.sum((time_term / denom_dimensionless / Dm_alpha2), axis=1) * (radius ** 2)
    
    # 7. Final Signal
    prefactor = 2 * (GYRO_MAGNETIC_RATIO * G_mag) ** 2
    log_E = -prefactor * sum_term
    
    return jnp.exp(log_E)
