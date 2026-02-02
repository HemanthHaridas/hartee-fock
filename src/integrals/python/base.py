import numpy as np


def gaussian_products(centerA: np.ndarray, exponentA: np.ndarray, centerB: np.ndarray, exponentB: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Gaussian product centers, exponents, and overlap prefactors.

    Parameters
    ----------
    centerA : np.ndarray
        Centers of Gaussian A (shape: [3,]).
    exponentA : np.ndarray
        Exponents of Gaussian A (shape: [n]).
    centerB : np.ndarray
        Centers of Gaussian B (shape: [3,]).
    exponentB : np.ndarray
        Exponents of Gaussian B (shape: [m]).

    Returns
    -------
    centers : np.ndarray
        Product centers (shape: [3, 3, dim]).
    exponents : np.ndarray
        Product exponents (shape: [n, m]).
    prefactors : np.ndarray
        Overlap prefactors (shape: [n, m]).
    """
    # Pairwise sums of exponents
    exponents = exponentA[:, None] + exponentB[None, :]

    # Weighted centers
    centers = (centerA[None, None, :] * exponentA[:, None, None] + centerB[None, None, :] * exponentB[None, :, None]) / exponents[:, :, None]

    # Distance squared between centers
    diff = centerA[None, None, :] - centerB[None, None, :]
    dist2 = np.sum(diff**2, axis=-1)

    # Prefactor
    prefactors = np.exp(- (exponentA[:, None] * exponentB[None, :]) / exponents * dist2)

    return centers, exponents, prefactors
