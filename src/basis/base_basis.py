import os
import math
import numpy
import abc
import enum


class AngularMomentum(enum.Enum):
    """
    Enumeration mapping angular momentum labels to Cartesian tuples.
    """
    S = [(0, 0, 0)]
    P = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    D = [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    F = [(3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 1, 1), (1, 0, 2), (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)]
    G = [(4, 0, 0), (3, 1, 0), (3, 0, 1), (2, 2, 0), (2, 1, 1), (2, 0, 2), (1, 3, 0), (1, 2, 1), (1, 1, 2), (1, 0, 3), (0, 4, 0), (0, 3, 1), (0, 2, 2), (0, 1, 3), (0, 0, 4)]
    H = [(5, 0, 0), (4, 1, 0), (4, 0, 1), (3, 2, 0), (3, 1, 1), (3, 0, 2), (2, 3, 0), (2, 2, 1), (2, 1, 2), (2, 0, 3), (1, 4, 0),
         (1, 3, 1), (1, 2, 2), (1, 1, 3), (1, 0, 4), (0, 5, 0), (0, 4, 1), (0, 3, 2), (0, 2, 3), (0, 1, 4), (0, 0, 5)]

    @classmethod
    def from_string(cls, label: str):
        """Convert string label (e.g. 'S', 'P') to AngularMomentum enum."""
        label = label.strip().upper()
        try:
            return cls[label]
        except KeyError:
            raise ValueError(f"Unknown angular momentum label: {label}")


def double_factorial(n: int) -> int:
    """
    Compute double factorial (2n-1)!!.
    """
    if n <= -2:
        return 0
    result = 1
    for k in range(n, 0, -2):
        result *= k
    return result


class BaseShell(abc.ABC):
    def __init__(self, angular_momentum: AngularMomentum = None):
        self.angular_momentum: AngularMomentum = angular_momentum
        self.exponents: numpy.ndarray = numpy.array([])
        self.coefficients: numpy.ndarray = numpy.array([])
        self.location: numpy.ndarray = numpy.zeros(3)
        self.atom: str = None
        # dictionary: tuple -> normalized coefficients
        self.normalized_coeffs: dict[tuple[int, int, int], numpy.ndarray] = {}

    def normalize_tuple(self, lx: int, ly: int, lz: int) -> numpy.ndarray:
        """Normalize contracted Gaussian for a given angular momentum tuple."""
        L = lx + ly + lz
        norm_constants = []
        for alpha in self.exponents:
            prefactor = (2**L * (2 * alpha)**(L + 1.5)) / (
                math.pi**1.5
                * double_factorial(2 * lx - 1)
                * double_factorial(2 * ly - 1)
                * double_factorial(2 * lz - 1)
            )
            N = math.sqrt(prefactor)
            norm_constants.append(N)
        prim_normed = self.coefficients * numpy.array(norm_constants)

        # contraction normalization
        S = numpy.zeros((len(self.exponents), len(self.exponents)))
        for i, ai in enumerate(self.exponents):
            for j, aj in enumerate(self.exponents):
                S[i, j] = (math.pi / (ai + aj))**1.5
        norm = math.sqrt(prim_normed @ S @ prim_normed)
        return prim_normed / norm

    def normalize(self):
        """Normalize all angular momentum tuples in this shell."""
        self.normalized_coeffs = {}
        for (lx, ly, lz) in self.angular_momentum.value:
            self.normalized_coeffs[(lx, ly, lz)] = self.normalize_tuple(lx, ly, lz)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(angular_momentum={self.angular_momentum.name}, "
            f"exponents={self.exponents}, coefficients={self.coefficients}, "
            f"normalized={list(self.normalized_coeffs.keys())})"
        )
