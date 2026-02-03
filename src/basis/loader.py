import xml.etree.ElementTree as ET
import os
import typing
import numpy
from src.basis import base_basis, gaussian
from xml.dom import minidom


class BasisSetLoader:
    """
    Loader for basis sets stored in the 'basis' folder.
    User provides basis set name and type (gaussian or molpro).
    Automatically normalizes all shells after parsing.
    """

    def __init__(self, basis_folder: str = "basis"):
        self.basis_folder = basis_folder

    def _load(self, basis_name: str, basis_type: str, atoms: list[str]):
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

    def load(self, molecule: typing.Dict[str, typing.Any], basis_name: str, basis_type: str) -> list[base_basis.BaseShell]:
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
        list[base_basis.BaseShell]
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
        _basis      : dict[str, list[base_basis.BaseShell]] = self._load(basis_name, basis_type, _atoms)
        self.full_basis : list[base_basis.BaseShell] = []

        for _index, _atom in enumerate(_atoms):
            for _sh in _basis[_atom]:
                _sh.location = _coords[_index] * 1.8897259885789  # Need to convert the coordinates to bohr before calculations
                _sh.atom = _atom
                self.full_basis.append(_sh)

        return self.full_basis

    def _write_xml(self, checkpoint: str = "checkpoint.xml") -> None:
        """
        Write the current basis set (self.full_basis) to an XML file.

        Parameters
        ----------
        checkpoint : str, optional
            Path to the XML file to write. Defaults to "checkpoint.xml".
        """
        if not hasattr(self, "full_basis") or self.full_basis is None:
            raise ValueError("No basis set loaded. Call load() before writing XML.")

        root = ET.Element("BasisSet")

        for sh in self.full_basis:
            shell_elem = ET.SubElement(root, "Shell")
            shell_elem.set("atom", sh.atom if sh.atom else "")
            shell_elem.set("location", " ".join(map(str, sh.location.tolist())))
            shell_elem.set("angular_momentum", str(sh.angular_momentum))

            # Exponents
            exps_elem = ET.SubElement(shell_elem, "Exponents")
            for exp in sh.exponents:
                exp_elem = ET.SubElement(exps_elem, "Exponent")
                exp_elem.text = str(exp)

            # Coefficients
            coeffs_elem = ET.SubElement(shell_elem, "Coefficients")
            for coeff in sh.coefficients:
                coeff_elem = ET.SubElement(coeffs_elem, "Coefficient")
                coeff_elem.text = str(coeff)

            # Normalized coefficients dictionary
            norm_elem = ET.SubElement(shell_elem, "NormalizedCoeffs")
            for key, arr in sh.normalized_coeffs.items():
                entry_elem = ET.SubElement(norm_elem, "Entry")
                entry_elem.set("lx", str(key[0]))
                entry_elem.set("ly", str(key[1]))
                entry_elem.set("lz", str(key[2]))
                entry_elem.text = " ".join(map(str, arr.tolist()))

            # Pretty-print using minidom
            rough_string = ET.tostring(root, encoding="utf-8")
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="\t")
            with open(checkpoint, "w", encoding="utf-8") as f:
                f.write(pretty_xml)

    def _read_xml(self, checkpoint: str = "checkpoint.xml") -> list[base_basis.BaseShell]:
        """
        Read basis set information from an XML file and set self.full_basis.

        Parameters
        ----------
        checkpoint : str, optional
            Path to the XML file to read. Defaults to "checkpoint.xml".

        Returns
        -------
        list[base_basis.BaseShell]
            List of reconstructed shells.
        """
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"XML checkpoint '{checkpoint}' not found")

        tree = ET.parse(checkpoint)
        root = tree.getroot()

        shells: list[base_basis.BaseShell] = []
        for shell_elem in root.findall("Shell"):
            atom = shell_elem.get("atom")
            location = numpy.array(list(map(float, shell_elem.get("location").split())))
            ang_mom = int(shell_elem.get("angular_momentum"))

            # Exponents
            exps = [float(exp_elem.text) for exp_elem in shell_elem.find("Exponents").findall("Exponent")]
            exps = numpy.array(exps)

            # Coefficients
            coeffs = [float(coeff_elem.text) for coeff_elem in shell_elem.find("Coefficients").findall("Coefficient")]
            coeffs = numpy.array(coeffs)

            # Normalized coefficients
            norm_coeffs: dict[tuple[int, int, int], numpy.ndarray] = {}
            norm_elem = shell_elem.find("NormalizedCoeffs")
            if norm_elem is not None:
                for entry_elem in norm_elem.findall("Entry"):
                    lx = int(entry_elem.get("lx"))
                    ly = int(entry_elem.get("ly"))
                    lz = int(entry_elem.get("lz"))
                    arr = numpy.array(list(map(float, entry_elem.text.split())))
                    norm_coeffs[(lx, ly, lz)] = arr

            # Construct shell
            sh = base_basis.BaseShell(ang_mom)
            sh.atom = atom
            sh.location = location
            sh.exponents = exps
            sh.coefficients = coeffs
            sh.normalized_coeffs = norm_coeffs

            shells.append(sh)

        self.full_basis = shells
        return self.full_basis
