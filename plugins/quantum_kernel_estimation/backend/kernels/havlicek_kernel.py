# Copyright 2022 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This Kernel stems from the paper by Havlíček et al.
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
            result *= np.pi - x_i
        return result
