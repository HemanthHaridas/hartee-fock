import numpy
import os
import xml.etree.ElementTree as ET
from src.basis import base_basis


def _get_or_create(root: ET.Element, tag: str) -> ET.Element:
    """
    Utility: return existing subelement or create a new one.
    """
    elem = root.find(tag)
    if elem is None:
        elem = ET.SubElement(root, tag)
    return elem


def _write_molecule_xml(root: ET.Element, molecule: dict) -> None:
    """
    Write or update molecule information.
    """
    mol_elem = _get_or_create(root, "Molecule")
    mol_elem.set("charge", str(molecule.get("charge", 0)))
    mol_elem.set("multiplicity", str(molecule.get("multiplicity", 1)))

    atoms_elem = _get_or_create(mol_elem, "Atoms")
    atoms_elem.clear()

    _atoms = molecule.get("atoms")
    _coords = molecule.get("coords")

    for symbol, coords in zip(_atoms, _coords):
        atom_elem = ET.SubElement(atoms_elem, "Atom")
        atom_elem.set("symbol", symbol)
        atom_elem.set("coords", " ".join(map(str, coords)))


def _write_basis_xml(root: ET.Element, basis_objects: list[base_basis.BaseShell]) -> None:
    """
    Write or update basis set information.
    """
    basis_elem = _get_or_create(root, "BasisSet")
    basis_elem.clear()

    for sh in basis_objects:
        shell_elem = ET.SubElement(basis_elem, "Shell")
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

        # Normalized coefficients
        norm_elem = ET.SubElement(shell_elem, "NormalizedCoeffs")
        for key, arr in sh.normalized_coeffs.items():
            entry_elem = ET.SubElement(norm_elem, "Entry")
            entry_elem.set("lx", str(key[0]))
            entry_elem.set("ly", str(key[1]))
            entry_elem.set("lz", str(key[2]))
            entry_elem.text = " ".join(map(str, arr.tolist()))


def _write_integrals_xml(root: ET.Element, integrals: dict[str, numpy.ndarray]) -> None:
    """
    Write or update integrals.
    """
    ints_elem = _get_or_create(root, "Integrals")
    ints_elem.clear()

    for name, mat in integrals.items():
        mat_elem = ET.SubElement(ints_elem, "Matrix")
        mat_elem.set("type", name)
        mat_elem.set("shape", f"{mat.shape[0]} {mat.shape[1]}")

        for row in mat:
            row_elem = ET.SubElement(mat_elem, "Row")
            row_elem.text = " ".join(map(str, row.tolist()))


def write_checkpoint(molecule: dict | None = None,
                     basis_objects: list[base_basis.BaseShell] | None = None,
                     integrals: dict[str, numpy.ndarray] | None = None,
                     checkpoint: str = "checkpoint.xml") -> None:
    """
    Unified driver to write/update molecule, basis, and integrals to a checkpoint XML.
    Any of the inputs may be None, in which case that section is skipped.
    """
    if os.path.exists(checkpoint):
        tree = ET.parse(checkpoint)
        root = tree.getroot()
    else:
        root = ET.Element("Checkpoint")
        tree = ET.ElementTree(root)

    if molecule is not None:
        _write_molecule_xml(root, molecule)
    if basis_objects is not None:
        _write_basis_xml(root, basis_objects)
    if integrals is not None:
        _write_integrals_xml(root, integrals)

    # Pretty-print using ElementTree.indent (Python 3.9+)
    ET.indent(tree, space="\t", level=0)
    tree.write(checkpoint, encoding="utf-8", xml_declaration=True)


def _read_molecule_xml(root: ET.Element) -> dict | None:
    """
    Read molecule information from XML.
    Returns dict with keys {"atoms", "charge", "multiplicity"} or None if missing.
    """
    mol_elem = root.find("Molecule")
    if mol_elem is None:
        return None

    molecule = {
        "charge": int(mol_elem.get("charge", 0)),
        "multiplicity": int(mol_elem.get("multiplicity", 1)),
        "atoms": []
    }

    atoms_elem = mol_elem.find("Atoms")
    if atoms_elem is not None:
        for atom_elem in atoms_elem.findall("Atom"):
            symbol = atom_elem.get("symbol")
            coords = list(map(float, atom_elem.get("coords").split()))
            molecule["atoms"].append((symbol, coords))

    return molecule


def _read_basis_xml(root: ET.Element) -> list[base_basis.BaseShell] | None:
    """
    Read basis set information from XML.
    Returns list of BaseShell objects or None if missing.
    """
    basis_elem = root.find("BasisSet")
    if basis_elem is None:
        return None

    shells: list[base_basis.BaseShell] = []
    for shell_elem in basis_elem.findall("Shell"):
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

    return shells


def _read_integrals_xml(root: ET.Element) -> dict[str, numpy.ndarray] | None:
    """
    Read integrals from XML.
    Returns dict of matrices keyed by type or None if missing.
    """
    ints_elem = root.find("Integrals")
    if ints_elem is None:
        return None

    integrals: dict[str, numpy.ndarray] = {}
    for mat_elem in ints_elem.findall("Matrix"):
        name = mat_elem.get("type")
        shape = tuple(map(int, mat_elem.get("shape").split()))
        rows = []
        for row_elem in mat_elem.findall("Row"):
            row = list(map(float, row_elem.text.split()))
            rows.append(row)
        mat = numpy.array(rows).reshape(shape)
        integrals[name] = mat

    return integrals


def read_checkpoint(checkpoint: str = "checkpoint.xml") -> tuple[dict | None,
                                                                 list[base_basis.BaseShell] | None,
                                                                 dict[str, numpy.ndarray] | None]:
    """
    Unified driver to read molecule, basis, and integrals from a checkpoint XML.
    Returns (molecule, basis_objects, integrals), each may be None if missing.
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint}' not found")

    tree = ET.parse(checkpoint)
    root = tree.getroot()

    molecule = _read_molecule_xml(root)
    basis_objects = _read_basis_xml(root)
    integrals = _read_integrals_xml(root)

    return molecule, basis_objects, integrals
