import numpy as np
import abc
from scr.basis import base


class IntegralEngine(abc.ABC):
    """
    Abstract base class for Gaussian integral engines.
    Provides a consistent interface for different algorithms
    (e.g., McMurchie-Davidson, Obara-Saika).
    """

    def __init__(self, basis_set: list[base.BaseShell] = None, use_optimized: str = False):
        """
        Initialize the integral engine.

        Parameters
        ----------
        basis_set : object, optional
            Basis set information (exponents, coefficients, angular momentum).
        """
        self.basis_set = basis_set

    @abc.abstractmethod
    def compute_overlap(self, shell_a, shell_b):
        """
        Compute overlap integrals between two shells.

        Parameters
        ----------
        shell_a : object
            First shell (contains exponents, coefficients, angular momentum).
        shell_b : object
            Second shell.

        Returns
        -------
        numpy.ndarray
            Overlap integral matrix.
        """
        pass

    @abc.abstractmethod
    def compute_eri(self, shell_a, shell_b, shell_c, shell_d):
        """
        Compute electron repulsion integrals (ERIs).

        Parameters
        ----------
        shell_a, shell_b, shell_c, shell_d : object
            Shells involved in the integral.

        Returns
        -------
        numpy.ndarray
            ERI tensor.
        """
        pass

    def compute_kinetic(self, shell_a, shell_b):
        """
        Optional: Compute kinetic energy integrals.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError("Kinetic integrals not implemented.")

    def compute_nuclear_attraction(self, shell_a, shell_b, center):
        """
        Optional: Compute nuclear attraction integrals.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError("Nuclear attraction integrals not implemented.")


# Utility Functions
def gaussian_product_theorem(centerA: np.ndarray, exponentA: np.ndarray,
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
    gaussian_centers = (e1[..., None] * centerA + e2[..., None] * centerB) / (e1 + e2)[..., None]

    # Combined exponents
    gaussian_exponents = (e1 * e2) / (e1 + e2)

    # Distance squared between centers
    diff = centerA - centerB
    squared_dist = np.dot(diff, diff)

    # Integrals
    gaussian_integrals = np.exp(-gaussian_exponents * squared_dist)

    return gaussian_centers, gaussian_integrals, gaussian_exponents
