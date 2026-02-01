import numpy as np
from src.basis import base


class Gaussian94Shell(base.BaseShell):
    """
    Shell parser for Gaussian94-type basis set files.
    Each shell block looks like:
        S   3  1.00
          34.0613410   0.0060680
           5.1235746   0.0453080
           1.1646626   0.2028220
    """

    def parse(self, lines: list[str]):
        header = lines[0].split()
        self.angular_momentum = base.AngularMomentum.from_string(header[0])
        nprimitives = int(header[1])
        scale = float(header[2]) if len(header) > 2 else 1.0

        exps, coeffs = [], []
        for line in lines[1:1 + nprimitives]:
            parts = line.split()
            exps.append(float(parts[0]))
            coeffs.append(float(parts[1]) * scale)

        self.exponents = np.array(exps)
        self.coefficients = np.array(coeffs)
        self.shell = np.array(self.angular_momentum.value[0])

    @staticmethod
    def parse_file(lines: list[str]) -> list["Gaussian94Shell"]:
        """
        Parse an entire Gaussian94 basis set file into multiple shells.
        """
        shells = []
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            header = lines[i].split()
            if len(header) < 2:
                i += 1
                continue
            nprimitives = int(header[1])
            block = lines[i:i + 1 + nprimitives]
            shell = Gaussian94Shell()
            shell.parse(block)
            shells.append(shell)
            i += 1 + nprimitives
        return shells
