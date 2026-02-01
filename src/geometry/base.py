import numpy
import typing
import abc


class BaseGeometry(abc.ABC):
    """
    Abstract Base Class (ABC) for molecular geometry representations.

    This class enforces the implementation of a `geometry` property in subclasses.
    Subclasses must define how molecular geometry information is stored and accessed.
    """

    def __init__(self) -> None:
        """
        Initialize the base geometry object with default molecular attributes.

        Attributes:
            atoms (List[str]): List of atomic symbols.
            coords (numpy.ndarray): Array of Cartesian coordinates with shape (N, 3).
            atomicnumbers (numpy.ndarray): 1D array of atomic numbers.
            atomicmasses (numpy.ndarray): 1D array of atomic masses.
            coords_internal (numpy.ndarray): Flattened representation of coords.
            natoms (Optional[int]): Number of atoms in the system.
            charge (int): Total molecular charge (default: 0).
            multi (int): Spin multiplicity (default: 1).
        """
        self.atoms          : typing.List[str] = []
        self.coords         : numpy.ndarray = numpy.zeros((0, 3), dtype=float)
        self.atomicnumbers  : numpy.ndarray = numpy.array([], dtype=int)
        self.atomicmasses   : numpy.ndarray = numpy.array([], dtype=float)
        self.coords_internal: numpy.ndarray = numpy.array([], dtype=float)
        self.natoms         : typing.Optional[int] = None
        self.charge         : int = 0
        self.multi          : int = 1

    @property
    @abc.abstractmethod
    def geometry(self) -> typing.Any:
        pass

    @geometry.setter
    @abc.abstractmethod
    def geometry(self, value: typing.Any) -> None:
        """
        Abstract setter for the molecular geometry.

        Args:
            value: The new geometry representation to assign.
        """
        pass
