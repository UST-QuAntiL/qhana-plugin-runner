from abc import abstractmethod
from typing import List
from pennylane import Device
from pennylane import numpy as pnp
from pennylane.qnode import QNode
import numpy as np
import enum
# from celery.utils.log import get_task_logger
#
# TASK_LOGGER = get_task_logger(__name__)


class EntanglementPatternEnum(enum.Enum):
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
                   entanglement_pattern: List[List[int]]):
        if self == KernelEnum.havlicek_kernel:
            from .havlicek_kernel import HavlicekKernel
            return HavlicekKernel(backend, n_qbits, reps, entanglement_pattern)

        elif self == KernelEnum.suzuki_kernel8:
            from .suzuki_kernels import SuzukiKernelEq8
            return SuzukiKernelEq8(backend, n_qbits, reps, entanglement_pattern)

        elif self == KernelEnum.suzuki_kernel9:
            from .suzuki_kernels import SuzukiKernelEq9
            return SuzukiKernelEq9(backend, n_qbits, reps, entanglement_pattern)

        elif self == KernelEnum.suzuki_kernel10:
            from .suzuki_kernels import SuzukiKernelEq10
            return SuzukiKernelEq10(backend, n_qbits, reps, entanglement_pattern)

        elif self == KernelEnum.suzuki_kernel11:
            from .suzuki_kernels import SuzukiKernelEq11
            return SuzukiKernelEq11(backend, n_qbits, reps, entanglement_pattern)

        elif self == KernelEnum.suzuki_kernel12:
            from .suzuki_kernels import SuzukiKernelEq12
            return SuzukiKernelEq12(backend, n_qbits, reps, entanglement_pattern)

        else:
            raise ValueError(f"Unkown kernel!")


class Kernel:

    def __init__(self, backend: Device, n_qbits: int, reps: int, entanglement_pattern: List[List[int]]):
        self.backend = backend
        self.n_qbits = n_qbits
        self.reps = reps
        self.entanglement_pattern = entanglement_pattern

    def get_projector_to_zero(self, n_qbits: int) -> pnp.array:
        """
        Returns projector to the zero state, i.e. density operator |0...0><0...0| of size 2^n_qbits x 2^n_qbits.
        """
        projector = pnp.array(np.zeros((2 ** n_qbits, 2 ** n_qbits)))
        projector[0, 0] = 1
        return projector

    @abstractmethod
    def get_circuit(self) -> QNode:
        raise NotImplementedError("Method evaluate is not implemented yet!")

    """
    This function returns a kernel-matrix with size len(y) x len(X)
    """
    def evaluate(self, X, y):
        is_symmetric = True
        if not np.array_equal(X, y):
            is_symmetric = False

        kernel_matrix = np.zeros((len(y), len(X)))
        start = 0
        circuit = self.get_circuit()
        for idx_x in range(len(X)):
            if is_symmetric:
                start = idx_x + 1
                kernel_matrix[idx_x, idx_x] = 1
            for idx_y in range(start, len(y)):
                kernel_matrix[idx_y, idx_x] = circuit(X[idx_x], y[idx_y])
        # TASK_LOGGER.info(qml.draw(circuit)(X[1], y[2]))  # DEBUG: draw circuit for X[1] and y[2]
        if is_symmetric:
            for idx_x in range(len(X)):
                for idx_y in range(idx_x):
                    kernel_matrix[idx_y, idx_x] = kernel_matrix[idx_x, idx_y]
        return kernel_matrix
