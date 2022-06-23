"""
This Kernel stems from the paper by Havlíček et al
Havlíček, V., Córcoles, A.D., Temme, K. et al.
Supervised learning with quantum-enhanced feature spaces.
Nature 567, 209–212 (2019). https://doi.org/10.1038/s41586-019-0980-2
"""

from .zz_kernel import ZZKernel
import numpy as np


class HavlicekKernel(ZZKernel):

    def __init__(self, backend, n_qbits, reps, entanglement_pattern_enum):
        super().__init__(backend, n_qbits, reps, entanglement_pattern_enum)

    def feature_map(self, x) -> float:
        result = 1
        if len(x) == 1:
            return x[0]
        for x_i in x:
            result *= (np.pi - x_i)
        return result
