import numpy
import abc
import typing
from src.basis import base_basis


class IntegralEngine(abc.ABC):
    """
    Abstract base class for Gaussian integral engines.
    Provides a consistent interface for different algorithms
    (e.g., McMurchie-Davidson, Obara-Saika).
    """

    def __init__(self, molecule: typing.Dict[str, typing.Any], basis_set: list[base_basis.BaseShell] = None, use_optimized: bool = False, checkpoint: bool = True):
        """
        Initialize the integral engine.

        Parameters
        ----------
        basis_set : object, optional
            Basis set information (exponents, coefficients, angular momentum).
        """
        self.basis_set        : list[base_basis.BaseShell] = basis_set
        self.integral_backend : typing.Any = None
        self.molecule         : typing.Dict[str, typing.Any] = molecule
        self.use_optimized    : bool = use_optimized

        if self.use_optimized:
            try:
                from src.integrals.cpp import integrals
                self.integral_backend = integrals
            except ImportError as e:
                raise RuntimeError("Optimized C++ backend not found.") from e

        # if checkpoint:
        #     self.checkpoint : typing.TextIO = open("checkpoint.xml", "a", encoding="utf-8")

    @abc.abstractmethod
    def compute_overlap(self):
        """
        Compute overlap integrals.

        Returns
        -------
        numpy.ndarray
            Overlap integral matrix.
        """
        pass

    # @abc.abstractmethod
    # def compute_eri(self, shell_a, shell_b, shell_c, shell_d):
    #     """
    #     Compute electron repulsion integrals (ERIs).

    #     Parameters
    #     ----------
    #     shell_a, shell_b, shell_c, shell_d : object
    #         Shells involved in the integral.

    #     Returns
    #     -------
    #     numpy.ndarray
    #         ERI tensor.
    #     """
    #     pass

    @abc.abstractmethod
    def compute_kinetic(self):
        """
        Optional: Compute kinetic energy integrals.
        Default implementation raises NotImplementedError.
        """

    def compute_nuclear_attraction(self, shell_a, shell_b, center):
        """
        Optional: Compute nuclear attraction integrals.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError("Nuclear attraction integrals not implemented.")


# Utility Functions
def gaussian_product_theorem(centerA: numpy.ndarray, exponentA: numpy.ndarray,
                             centerB: numpy.ndarray, exponentB: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
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
    squared_dist = numpy.dot(diff, diff)

    # Integrals
    gaussian_integrals = numpy.exp(-gaussian_exponents * squared_dist)

    return gaussian_centers, gaussian_integrals, gaussian_exponents


def double_factorial_array(values: numpy.ndarray) -> numpy.ndarray:
    """
    Create a 2D array where each row starts at a value from `values`
    and descends to 1 with step size 2, then compute double factorials.

    Parameters
    ----------
    values : ndarray of int
        Input array of non-negative integers.

    Returns
    -------
    ndarray of int
        2D array of double factorials.
    """
    # Maximum row length (largest value down to 1, step 2)
    max_len = (values.max() // 2) + 1
    mat = numpy.zeros((len(values), max_len), dtype=int)

    for i, v in enumerate(values):
        row = numpy.arange(v, 0, -2)
        mat[i, :len(row)] = row

    # Vectorized double factorial using gamma function generalization
    # For integers: n!! = 2^(n/2) * (n/2)! if n even, else (n)! / (2^( (n-1)/2 ) * ((n-1)/2)! )
    def _double_factorial(n: int):
        if n == 0 or n == -1:
            return 1
        elif n < -1:
            return 0
        result = 1
        while n > 0:
            result *= n
            n -= 2
        return result

    vec_df = numpy.vectorize(_double_factorial)

    # results are the first columns
    return vec_df(mat)[:, 0]
