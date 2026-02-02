import numpy as np


def gaussiantheorem(centerA: np.ndarray, exponentA: np.ndarray,
                    centerB: np.ndarray, exponentB: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Gaussian product theorem quantities for two 3D centers with sets of exponents.

    This function evaluates the effective Gaussian center, the overlap integral prefactor,
    and the combined exponent for all pairwise combinations of exponents from two centers.
    It is fully vectorized using NumPy broadcasting, producing results for an (N × M) grid
    of exponent pairs.

    Parameters
    ----------
    centerA : ndarray of shape (3,)
        Cartesian coordinates of the first Gaussian center.
    exponentA : ndarray of shape (N,)
        Exponents associated with the first center.
    centerB : ndarray of shape (3,)
        Cartesian coordinates of the second Gaussian center.
    exponentB : ndarray of shape (M,)
        Exponents associated with the second center.

    Returns
    -------
    gaussiancenters : ndarray of shape (N, M, 3)
        Effective Gaussian centers for each exponent pair (i, j).
    gaussianintegrals : ndarray of shape (N, M)
        Overlap integral prefactors for each exponent pair (i, j),
        given by exp(-γ * |centerA - centerB|²).
    gaussianexponents : ndarray of shape (N, M)
        Combined exponents γ = (α * β) / (α + β) for each pair (i, j),
        where α ∈ exponentA and β ∈ exponentB.

    Notes
    -----
    - The function assumes centers are fixed 3D vectors, while exponents vary.
    - Broadcasting ensures efficient SIMD‑friendly evaluation across all pairs.
    - This is a core building block for molecular integral evaluation in
      computational chemistry.
    """

    # Reshape for broadcasting
    e1 = exponentA[:, None]        # (N,1)
    e2 = exponentB[None, :]        # (1,M)

    # Combined centers (broadcast over 3D vector)
    gaussiancenters = (e1[..., None] * centerA + e2[..., None] * centerB) / (e1 + e2)[..., None]

    # Combined exponents
    gaussianexponents = (e1 * e2) / (e1 + e2)

    # Distance squared between centers
    diff = centerA - centerB
    squared_dist = np.dot(diff, diff)

    # Integrals
    gaussianintegrals = np.exp(-gaussianexponents * squared_dist)

    return gaussiancenters, gaussianintegrals, gaussianexponents
