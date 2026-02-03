import numpy
from src.integrals.python import base_integrals


def overlap_md_3D(centerA: numpy.ndarray, exponentA: numpy.ndarray, shellA: numpy.ndarray,
                  centerB: numpy.ndarray, exponentB: numpy.ndarray, shellB: numpy.ndarray) -> numpy.ndarray:
    """
    Compute overlap integral using McMurchie-Davidson scheme.

    The MD scheme uses Hermite Gaussians as an intermediate expansion.
    """
    # get product gaussians
    gaussian_centers, gaussian_integrals, gaussian_exponents = base_integrals.gaussian_product_theorem(
        centerA, exponentA, centerB, exponentB
    )

    exponentA = numpy.ravel(exponentA)
    exponentB = numpy.ravel(exponentB)
    comb_exp = exponentA[:, None] + exponentB[None, :]

    # Compute Hermite expansion coefficients E^{ij}_t
    Ex = hermite_expansion_coefficients(
        shellA[0], shellB[0],
        centerA[0], centerB[0],
        gaussian_centers[..., 0],
        comb_exp
    )

    Ey = hermite_expansion_coefficients(
        shellA[1], shellB[1],
        centerA[1], centerB[1],
        gaussian_centers[..., 1],
        comb_exp
    )

    Ez = hermite_expansion_coefficients(
        shellA[2], shellB[2],
        centerA[2], centerB[2],
        gaussian_centers[..., 2],
        comb_exp
    )

    # Compute Hermite integrals (for overlap, these are simple)
    # [0]^{(t)} = { sqrt(pi/p)  if t=0
    #             { 0           otherwise

    hermite_x = numpy.zeros_like(Ex)
    hermite_y = numpy.zeros_like(Ey)
    hermite_z = numpy.zeros_like(Ez)

    hermite_x[..., 0] = numpy.sqrt(numpy.pi / comb_exp)
    hermite_y[..., 0] = numpy.sqrt(numpy.pi / comb_exp)
    hermite_z[..., 0] = numpy.sqrt(numpy.pi / comb_exp)

    # Contract: sum over Hermite indices
    overlap_x = numpy.sum(Ex * hermite_x, axis=-1)
    overlap_y = numpy.sum(Ey * hermite_y, axis=-1)
    overlap_z = numpy.sum(Ez * hermite_z, axis=-1)

    # Final overlap integral
    overlap = gaussian_integrals * overlap_x * overlap_y * overlap_z

    return overlap


def hermite_expansion_coefficients(i: int, j: int, XA: float, XB: float, XP: numpy.ndarray, p: numpy.ndarray) -> numpy.ndarray:
    """
    Compute Hermite expansion coefficients E^{ij}_t using McMurchie-Davidson recursion.
    Fully vectorized implementation.
    """
    # Maximum Hermite index needed
    max_t = i + j

    # Initialize array: E[..., angular_A, angular_B, hermite_index]
    E = numpy.zeros(p.shape + (i + 1, j + 1, max_t + 1))

    # Base case: E^{0,0}_0 = 1
    E[..., 0, 0, 0] = 1.0

    # Build up i index (angular momentum on A) - vectorized over t
    for ii in range(i):
        # Vectorize over all t values at once
        t_max_current = ii + j + 1
        t_vals = numpy.arange(t_max_current)

        # Term 1: (1/2p) * E^{i,j}_{t-1} for t > 0
        E[..., ii + 1, :, 1:t_max_current] += (1.0 / (2.0 * p))[..., None, None] * E[..., ii, :, 0:t_max_current - 1]

        # Term 2: (XP - XA) * E^{i,j}_t for all t
        E[..., ii + 1, :, :t_max_current] += (XP - XA)[..., None, None] * E[..., ii, :, :t_max_current]

        # Term 3: (t+1) * E^{i,j}_{t+1} for t < max_t
        if t_max_current <= max_t:
            t_coeffs = numpy.arange(1, t_max_current + 1)
            E[..., ii + 1, :, :t_max_current] += t_coeffs[None, None, :] * E[..., ii, :, 1:t_max_current + 1]

    # Build up j index (angular momentum on B) - vectorized over t
    for jj in range(j):
        t_max_current = i + jj + 1

        # Term 1: (1/2p) * E^{i,j}_{t-1} for t > 0
        E[..., :, jj + 1, 1:t_max_current] += (1.0 / (2.0 * p))[..., None, None] * E[..., :, jj, 0:t_max_current - 1]

        # Term 2: (XP - XB) * E^{i,j}_t for all t
        E[..., :, jj + 1, :t_max_current] += (XP - XB)[..., None, None] * E[..., :, jj, :t_max_current]

        # Term 3: (t+1) * E^{i,j}_{t+1} for t < max_t
        if t_max_current <= max_t:
            t_coeffs = numpy.arange(1, t_max_current + 1)
            E[..., :, jj + 1, :t_max_current] += t_coeffs[None, None, :] * E[..., :, jj, 1:t_max_current + 1]

    # Return the coefficients for the requested (i, j) with all t values
    return E[..., i, j, :]
