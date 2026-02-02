import os
import typing
import numpy
from src.basis import base, gaussian


class BasisSetLoader:
    """
    Loader for basis sets stored in the 'basis' folder.
    User provides basis set name and type (gaussian or molpro).
    Automatically normalizes all shells after parsing.
    """

    def __init__(self, basis_folder: str = "basis"):
        self.basis_folder = basis_folder

    def load(self, basis_name: str, basis_type: str, atoms: list[str]):
        filepath = os.path.join(self.basis_folder, basis_name)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Basis set file '{filepath}' not found in {self.basis_folder}")

        with open(filepath, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        basis_type = basis_type.lower()
        if basis_type == ("gaussian94"):
            self.shells_by_atom = gaussian.Gaussian94Shell.parse_file(lines, atoms)
        else:
            raise ValueError("basis_type must be 'gaussian94'")

        # Normalize all shells
        for atom_shells in self.shells_by_atom.values():
            for sh in atom_shells:
                sh.normalize()

        return self.shells_by_atom

    def set_basis_functions(self, molecule: typing.Dict[str, typing.Any], basis_name: str, basis_type: str) -> list[base.BaseShell]:
        """
        Construct and assign basis functions to each atom in a molecule.

        This method loads the specified basis set, parses it into atomic shells,
        normalizes the shells, and then attaches them to the atoms in the given
        molecule. Each shell is annotated with its atomic symbol and Cartesian
        coordinates, producing a complete list of basis functions for the system.

        Parameters
        ----------
        molecule : dict[str, Any]
            A dictionary describing the molecule, with at least:
            - "atoms" : list[str]
                List of atomic symbols (e.g., ["H", "O", "H"]).
            - "coords" : numpy.ndarray
                Array of Cartesian coordinates with shape (N, 3), where N is
                the number of atoms.
        basis_name : str
            Name of the basis set file to load (e.g., "sto-3g.gbs").
        basis_type : str
            Type of basis set format (currently only "gaussian94" supported).

        Returns
        -------
        list[base.BaseShell]
            A flat list of all basis functions (shells) in the molecule,
            each annotated with its atom type and spatial location.

        Raises
        ------
        FileNotFoundError
            If the specified basis set file cannot be found.
        ValueError
            If the basis_type is unsupported.
        """

        _atoms      : list[str] = molecule["atoms"]
        _coords     : numpy.ndarray = molecule["coords"]
        _basis      : dict[str, list[base.BaseShell]] = self.load(basis_name, basis_type, _atoms)
        _full_basis : list[base.BaseShell] = []

        for _index, _atom in enumerate(_atoms):
            for _sh in _basis[_atom]:
                _sh.location = _coords[_index]
                _sh.atom = _atom
                _full_basis.append(_sh)

        return _full_basis
