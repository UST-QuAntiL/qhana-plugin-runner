# Copyright 2021 QHAna plugin runner contributors.
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
from abc import abstractmethod
from typing import List
from pennylane import Device
import numpy as np
import enum
from celery.utils.log import get_task_logger

TASK_LOGGER = get_task_logger(__name__)


class EntanglementPatternEnum(enum.Enum):
    """
    Enum listing all available entanglement patterns
    """
    full = "full"
    linear = "linear"
    circular = "circular"

    def get_pattern(self, n_qbits: int) -> List[List[int]]:
        if self == EntanglementPatternEnum.full:
            return [[i] for i in range(n_qbits)] + [[i, j] for i in range(n_qbits) for j in range(i + 1, n_qbits)]
        elif self == EntanglementPatternEnum.linear:
            return [[i] for i in range(n_qbits)] + [[i, i+1] for i in range(n_qbits - 1)]
        elif self == EntanglementPatternEnum.circular:
            return [[i] for i in range(n_qbits)] + [[n_qbits - 1, 0]] + [[i, i+1] for i in range(n_qbits - 1)]
        else:
            raise ValueError(f"Unkown entanglement pattern is not implemented!")


class KernelEnum(enum.Enum):
    """
    Enum listing all available kernels
    """
    havlicek_kernel = 'Havlíček Kernel'
    suzuki_kernel8 = 'Suzuki Kernel 8'
    suzuki_kernel9 = 'Suzuki Kernel 9'
    suzuki_kernel10 = 'Suzuki Kernel 10'
    suzuki_kernel11 = 'Suzuki Kernel 11'
    suzuki_kernel12 = 'Suzuki Kernel 12'

    def get_kernel(self,
                   backend: Device,
                   n_qbits: int,
                   reps: int,
                   entanglement_pattern_enum: EntanglementPatternEnum):
        if self == KernelEnum.havlicek_kernel:
            from .havlicek_kernel import HavlicekKernel
            return HavlicekKernel(backend, n_qbits, reps, entanglement_pattern_enum)

        elif self == KernelEnum.suzuki_kernel8:
            from .suzuki_kernels import SuzukiKernelEq8
            return SuzukiKernelEq8(backend, n_qbits, reps, entanglement_pattern_enum)

        elif self == KernelEnum.suzuki_kernel9:
            from .suzuki_kernels import SuzukiKernelEq9
            return SuzukiKernelEq9(backend, n_qbits, reps, entanglement_pattern_enum)

        elif self == KernelEnum.suzuki_kernel10:
            from .suzuki_kernels import SuzukiKernelEq10
            return SuzukiKernelEq10(backend, n_qbits, reps, entanglement_pattern_enum)

        elif self == KernelEnum.suzuki_kernel11:
            from .suzuki_kernels import SuzukiKernelEq11
            return SuzukiKernelEq11(backend, n_qbits, reps, entanglement_pattern_enum)

        elif self == KernelEnum.suzuki_kernel12:
            from .suzuki_kernels import SuzukiKernelEq12
            return SuzukiKernelEq12(backend, n_qbits, reps, entanglement_pattern_enum)

        else:
            raise ValueError(f"Unkown kernel!")


class Kernel:

    def __init__(self, backend: Device, n_qbits: int, reps: int, entanglement_pattern_enum: EntanglementPatternEnum):
        self.backend = backend
        self.n_qbits = n_qbits
        self.max_qbits = backend.num_wires
        self.reps = reps
        self.entanglement_pattern_enum = entanglement_pattern_enum

    @abstractmethod
    def execute_circuit(self, X, Y, to_calculate, entanglement_pattern) -> List[float]:
        raise NotImplementedError("Method evaluate is not implemented yet!")

    @abstractmethod
    def get_qbits_needed(self, X, y) -> int:
        """
        Returns the number of qbits needed, to compute one quantum circuit
        """

    def evaluate(self, X, y):
        """
        This function computes the kernel matrix between input X and y.
        If possible, it evaluates multiple entries at once, e.g. if we have 5 qubits available and need 2 qubits to
        evaluate on entry, then the circuits of two entries will run in parallel.
        :param X: A list of data points
        :param y: A list of data points
        :return: kernel-matrix of size len(y) x len(X)
        """
        needed_qbits = self.get_qbits_needed(X, y)
        entanglement_pattern = self.entanglement_pattern_enum.get_pattern(needed_qbits)
        amount_of_circuits = 0

        is_symmetric = True
        if not np.array_equal(X, y):
            is_symmetric = False

        kernel_matrix = np.zeros((len(y), len(X)))
        start = 0
        to_calculate = []
        next_qbit = 0
        if needed_qbits > self.max_qbits:
            raise ValueError(f"The number of needed qubits exceeds the number given qubits.")
        for idx_x in range(len(X)):
            if is_symmetric:
                start = idx_x + 1
                kernel_matrix[idx_x, idx_x] = 1
            for idx_y in range(start, len(y)):
                if next_qbit + needed_qbits > self.max_qbits:
                    result = self.execute_circuit(X, y, to_calculate, entanglement_pattern)
                    amount_of_circuits += 1
                    for i in range(len(result)):
                        result_idx_x = to_calculate[i][0]
                        result_idx_y = to_calculate[i][1]
                        kernel_matrix[result_idx_y, result_idx_x] = result[i]
                    next_qbit = 0
                    to_calculate = []

                wires_to_use = [i + next_qbit for i in range(needed_qbits)]
                to_calculate.append([idx_x, idx_y, wires_to_use])
                next_qbit += needed_qbits

        if next_qbit != 0:
            result = self.execute_circuit(X, y, to_calculate, entanglement_pattern)
            amount_of_circuits += 1
            for i in range(len(result)):
                result_idx_x = to_calculate[i][0]
                result_idx_y = to_calculate[i][1]
                kernel_matrix[result_idx_y, result_idx_x] = result[i]
        # TASK_LOGGER.info(qml.draw(circuit)(X[1], y[2]))  # DEBUG: draw circuit for X[1] and y[2]
        if is_symmetric:
            for idx_x in range(len(X)):
                for idx_y in range(idx_x):
                    kernel_matrix[idx_y, idx_x] = kernel_matrix[idx_x, idx_y]
        TASK_LOGGER.info(f"It took {amount_of_circuits} quantum circuits")
        return kernel_matrix
