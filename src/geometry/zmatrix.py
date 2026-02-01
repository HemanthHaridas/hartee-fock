import typing
import numpy
from src.geometry import base
from src.helpers import tables


class ZMatrix(base.BaseGeometry):
    def __init__(self):
        super().__init__()

    @property
    def geometry(self) -> typing.Dict[str, typing.Any]:
        """
        Get the molecular geometry in Cartesian representation.

        Returns:
            dict: A dictionary containing:
                - atoms (List[str]): Atomic symbols.
                - coords (numpy.ndarray): Cartesian coordinates (N, 3).
                - charge (int): Molecular charge.
                - multi (int): Spin multiplicity.
                - atomicnumbers (numpy.ndarray): Atomic numbers.
                - atomicmasses (numpy.ndarray): Atomic masses.
        """
        return {
            "atoms"        : self.atoms,
            "coords"       : self.coords,
            "charge"       : self.charge,
            "multi"        : self.multi,
            "atomicnumbers": self.atomicnumbers,
            "atomicmasses" : self.atomicmasses,
        }
