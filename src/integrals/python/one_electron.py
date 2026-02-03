import numpy
from scr.integrals.python import base


class McMurchieDavidson(base.IntegralEngine):
    def __init__(self, basis_set: list[base.BaseShell]):
        super().__init__(basis_set)


class ObaraSaika(base.IntegralEngine):
    def __init__(self, basis_set: list[base.BaseShell]):
        super().__init__(basis_set)
