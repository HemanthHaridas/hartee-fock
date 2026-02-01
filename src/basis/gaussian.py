import numpy
from src.basis import base


class Gaussian94Shell(base.BaseShell):
    @classmethod
    def parse_file(cls, lines: list[str], atoms: list[str]) -> dict[str, list[base.BaseShell]]:
        """
        Parse a Gaussian94 .gbs file and return shells only for requested atoms.

        Parameters
        ----------
        lines : list[str]
            Contents of the basis set file.
        atoms : list[str]
            Atom symbols present in the molecule (e.g. ["H", "O", "C"]).

        Returns
        -------
        dict[str, list[BaseShell]]
            Mapping from atom symbol to list of shells.
        """
        shells_by_atom = {}
        it = iter(lines)
        current_element = None

        for line in it:
            if not line or line.startswith("!"):
                continue
            if line.startswith("****"):
                current_element = None
                continue

            parts = line.split()
            # element header
            if len(parts) == 2 and parts[0].isalpha():
                current_element, charge = parts[0], int(parts[1])
                if current_element not in atoms:
                    current_element = None  # skip this block
                else:
                    shells_by_atom[current_element] = []
                continue

            # shell header
            if current_element and len(parts) >= 3:
                label, nprim, scale = parts[0], int(parts[1]), float(parts[2])
                am = base.AngularMomentum.from_string(label)
                exps, coeffs = [], []
                for _ in range(nprim):
                    exp_line = next(it).replace("D", "E")
                    exp, coeff = exp_line.split()
                    exps.append(float(exp))
                    coeffs.append(float(coeff) * scale)
                shell = cls(angular_momentum=am)
                shell.exponents = numpy.array(exps)
                shell.coefficients = numpy.array(coeffs)
                shells_by_atom[current_element].append(shell)

        return shells_by_atom
