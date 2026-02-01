import os
from src.basis import gaussian


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
            shells_by_atom = gaussian.Gaussian94Shell.parse_file(lines, atoms)
        else:
            raise ValueError("basis_type must be 'gaussian94'")

        # Normalize all shells
        for atom_shells in shells_by_atom.values():
            for sh in atom_shells:
                sh.normalize()

        return shells_by_atom
