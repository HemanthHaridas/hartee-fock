import numpy as np
from src.basis import base


class MolproShell(base.BaseShell):
    """
    Shell parser for Molpro basis set files.
    Each shell block looks like:
        s, C, 34.0613410, 0.0060680;
        s, C, 5.1235746, 0.0453080;
        s, C, 1.1646626, 0.2028220;
    """

    def parse(self, lines: list[str]):
        exps, coeffs = [], []
        am_label = None
        for line in lines:
            if not line.strip() or line.strip().startswith("basis"):
                continue
            parts = line.replace(";", "").split(",")
            am_label = parts[0].strip().upper()
            exps.append(float(parts[2]))
            coeffs.append(float(parts[3]))
        self.angular_momentum = base.AngularMomentum.from_string(am_label)
        self.exponents = np.array(exps)
        self.coefficients = np.array(coeffs)
        self.shell = np.array(self.angular_momentum.value[0])

    @staticmethod
    def parse_file(lines: list[str]) -> list["MolproShell"]:
        """Parse an entire Molpro basis set file into multiple shells."""
        shells = []
        current_block = []
        for line in lines:
            if line.strip().startswith("basis") or not line.strip():
                continue
            current_block.append(line)
            # End of a shell block marked by semicolon
            if line.strip().endswith(";"):
                shell = MolproShell()
                shell.parse(current_block)
                shells.append(shell)
                current_block = []
        return shells
