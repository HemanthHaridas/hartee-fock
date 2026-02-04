import numpy
import typing
from src.basis import base_basis
from src.integrals.python import base_integrals, one_electron


class McMurchieDavidson(base_integrals.IntegralEngine):
    def __init__(self, molecule: typing.Dict[str, typing.Any], basis_set: list[base_basis.BaseShell], use_optimized: bool = False):
        super().__init__(molecule, basis_set, use_optimized)

    def compute_overlap(self) -> numpy.ndarray:
        """
        Compute the symmetric overlap matrix using the McMurchie–Davidson algorithm.

        Returns:
            numpy.ndarray: (num_basis × num_basis) symmetric overlap matrix.
        """
        num_basis = len(self.basis_set)
        shell_idx = numpy.arange(num_basis)

        # Build index grids
        ix_grid, iy_grid = numpy.meshgrid(shell_idx, shell_idx, indexing="ij")

        # Define scalar overlap function
        def overlap_func(ix: int, iy: int) -> float:
            if ix > iy:
                return 0.0

            integral = one_electron.overlap_md_3D(
                centerA=self.basis_set[ix].location,
                exponentA=self.basis_set[ix].exponents,
                shellA=self.basis_set[ix].angular_momentum.value[0],
                centerB=self.basis_set[iy].location,
                exponentB=self.basis_set[iy].exponents,
                shellB=self.basis_set[iy].angular_momentum.value[0]
            )

            contracted = numpy.sum(
                integral
                * self.basis_set[ix].normalized_coeffs[self.basis_set[ix].angular_momentum.value[0]][..., None]
                * self.basis_set[iy].normalized_coeffs[self.basis_set[iy].angular_momentum.value[0]][None, ...]
            )
            return contracted

        # Vectorize the scalar function
        vectorized_overlap = numpy.vectorize(overlap_func, otypes=[float])
        overlaps = vectorized_overlap(ix_grid, iy_grid)

        # Enforce symmetry
        overlaps = overlaps + overlaps.T - numpy.diag(numpy.diag(overlaps))

        return overlaps

    def compute_kinetic(self) -> numpy.ndarray:
        """
        Compute the symmetric kinetic matrix using the McMurchie–Davidson algorithm.

        Returns:
            numpy.ndarray: (num_basis × num_basis) symmetric overlap matrix.
        """
        num_basis = len(self.basis_set)
        shell_idx = numpy.arange(num_basis)

        # Build index grids
        ix_grid, iy_grid = numpy.meshgrid(shell_idx, shell_idx, indexing="ij")

        # Define scalar overlap function
        def kinetic_func(ix: int, iy: int) -> float:
            if ix > iy:
                return 0.0

            integral = one_electron.kinetic_md_3D(
                centerA=self.basis_set[ix].location,
                exponentA=self.basis_set[ix].exponents,
                shellA=self.basis_set[ix].angular_momentum.value[0],
                centerB=self.basis_set[iy].location,
                exponentB=self.basis_set[iy].exponents,
                shellB=self.basis_set[iy].angular_momentum.value[0]
            )

            contracted = numpy.sum(
                integral
                * self.basis_set[ix].normalized_coeffs[self.basis_set[ix].angular_momentum.value[0]][..., None]
                * self.basis_set[iy].normalized_coeffs[self.basis_set[iy].angular_momentum.value[0]][None, ...]
            )
            return contracted

        # Vectorize the scalar function
        vectorized_kinetic = numpy.vectorize(kinetic_func, otypes=[float])
        kinetics = vectorized_kinetic(ix_grid, iy_grid)

        # Enforce symmetry
        kinetics = kinetics + kinetics.T - numpy.diag(numpy.diag(kinetics))

        return kinetics
