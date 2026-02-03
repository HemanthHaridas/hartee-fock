import numpy
from src.basis import base
from src.integrals.python import base as baseIntegrals


class McMurchieDavidson(baseIntegrals.IntegralEngine):
    def __init__(self, basis_set: list[base.BaseShell], use_optimized: bool = False):
        super().__init__(basis_set, use_optimized)
        if use_optimized:
            pass


class ObaraSaika(baseIntegrals.IntegralEngine):
    def __init__(self, basis_set: list[base.BaseShell], use_optimized: bool = False):
        super().__init__(basis_set, use_optimized)
        if use_optimized:
            pass
