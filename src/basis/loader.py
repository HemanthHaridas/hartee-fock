import os
from src.basis import gaussian, molpro


class BasisSetLoader:
    """
    Loader for basis sets stored in the 'basis' folder.
    User provides basis set name and type (gaussian or molpro).
    Automatically normalizes all shells after parsing.
    """

    def __init__(self, basis_folder: str = "basis"):
        self.basis_folder = basis_folder

    def load(self, basis_name: str, basis_type: str):
        """
        Load and normalize a basis set by name and type.

        Parameters
        ----------
        basis_name : str
            Name of the basis set file (e.g. 'sto-3g', 'cc-pVDZ').
        basis_type : str
            Type of basis set ('gaussian' or 'molpro').

        Returns
        -------
        shells : list[BaseShell]
            Parsed and normalized shells from the basis set file.
        """
        filepath = os.path.join(self.basis_folder, basis_name)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Basis set file '{filepath}' not found in {self.basis_folder}")

        with open(filepath, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        basis_type = basis_type.lower()
        if basis_type == "gaussian94":
            shells = gaussian.Gaussian94Shell.parse_file(lines)
        elif basis_type == "molpro":
            shells = molpro.MolproShell.parse_file(lines)
        else:
            raise ValueError("basis_type must be either 'gaussian' or 'molpro'")

        # Automatically normalize all shells
        for sh in shells:
            sh.normalize()

        return shells
