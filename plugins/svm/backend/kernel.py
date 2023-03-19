from enum import Enum
from typing import List, Callable, Optional
import numpy as np
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel


class EntanglementPatternEnum(Enum):
    full = "Full"
    linear = "Linear"
    circular = "Circular"

    def get_pattern(self) -> str:
        return self.name


class KernelEnum(Enum):
    rbf = "Radial Basis Function"
    linear = "Linear"
    poly = "Polynomial"
    sigmoid = "Sigmoid"
    precomputed = "Precomputed"
    z_kernel = "Z Kernel"
    zz_kernel = "ZZ Kernel"
    pauli_kernel = "Pauli Kernel"

    def is_classical(self):
        if (
            self.name == "rbf"
            or self.name == "linear"
            or self.name == "poly"
            or self.name == "sigmoid"
            or self.name == "precomputed"
        ):
            return True
        else:
            return False

    def get_kernel(
        self, **kwargs
    ) -> str | Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
        if self.is_classical():
            return self.name
        else:
            return self.get_quantum_kernel(**kwargs)

    def get_quantum_kernel(
        self,
        backend,
        n_qbits: int,
        paulis: List[str],
        reps: int,
        entanglement_pattern: str,
        data_map_func: Callable[[np.array | List[float]], float],
    ) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
        if self == KernelEnum.z_kernel:
            feature_map = ZFeatureMap(
                feature_dimension=n_qbits,
                reps=reps,
                data_map_func=data_map_func,
            )  # This FeatureMap has no entanglement

        elif self == KernelEnum.zz_kernel:
            feature_map = ZZFeatureMap(
                feature_dimension=n_qbits,
                entanglement=entanglement_pattern,
                reps=reps,
                data_map_func=data_map_func,
            )

        elif self == KernelEnum.pauli_kernel:
            feature_map = PauliFeatureMap(
                feature_dimension=n_qbits,
                paulis=paulis,
                entanglement=entanglement_pattern,
                reps=reps,
                data_map_func=data_map_func,
            )

        else:
            raise NotImplementedError("Unkown kernel!")

        return QuantumKernel(feature_map=feature_map, quantum_instance=backend).evaluate
