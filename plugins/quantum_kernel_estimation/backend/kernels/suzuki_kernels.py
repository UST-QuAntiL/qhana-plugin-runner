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
The Kernels stem from the paper by Suzuki et al.
Suzuki, Y., Yano, H., Gao, Q. et al.
Analysis and synthesis of feature map for kernel-based quantum classifier.
Quantum Mach. Intell. 2, 9 (2020). https://doi.org/10.1007/s42484-020-00020-y

The Kernels actually make most sense, when n_qbits=2
"""

from .zz_kernel import ZZKernel
import numpy as np


class SuzukiKernelEq8(ZZKernel):
    def __init__(self, backend, n_qbits, reps, entanglement_pattern):
        super().__init__(backend, n_qbits, reps, entanglement_pattern)

    def feature_map(self, x) -> float:
        result = np.pi
        if len(x) == 1:
            return x[0]
        for x_i in x:
            result *= x_i
        return result


class SuzukiKernelEq9(ZZKernel):
    def __init__(self, backend, n_qbits, reps, entanglement_pattern):
        super().__init__(backend, n_qbits, reps, entanglement_pattern)

    def feature_map(self, x) -> float:
        result = np.pi / 2.0
        if len(x) == 1:
            return x[0]
        for x_i in x:
            result *= 1 - x_i
        return result


class SuzukiKernelEq10(ZZKernel):
    def __init__(self, backend, n_qbits, reps, entanglement_pattern):
        super().__init__(backend, n_qbits, reps, entanglement_pattern)

    def feature_map(self, x) -> float:
        if len(x) == 1:
            return x[0]
        square_diff = []
        for i, x_i in enumerate(x):
            for j, x_j in enumerate(x):
                if i != j:
                    square_diff.append((x_i - x_j) * (x_j - x_i))
        mean = np.array(square_diff).mean()
        return np.pi * np.exp(mean / 8.0)


class SuzukiKernelEq11(ZZKernel):
    def __init__(self, backend, n_qbits, reps, entanglement_pattern):
        super().__init__(backend, n_qbits, reps, entanglement_pattern)

    def feature_map(self, x) -> float:
        if len(x) == 1:
            return x[0]
        x = 1.0 / np.cos(x)
        return np.pi / 3.0 * np.product(x)


class SuzukiKernelEq12(ZZKernel):
    def __init__(self, backend, n_qbits, reps, entanglement_pattern):
        super().__init__(backend, n_qbits, reps, entanglement_pattern)

    def feature_map(self, x) -> float:
        if len(x) == 1:
            return x[0]
        x = np.cos(x)
        return np.pi * np.product(x)
